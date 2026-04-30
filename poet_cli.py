import typer
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
from datetime import datetime, timedelta

# ============================================
# CONFIGURATION
# ============================================
DATA_RAW = Path("data/raw")
DATA_TRAIN = Path("data/training")
DATA_MODELS = Path("data/models")
VIRGIN_PATH = DATA_MODELS / "virgin_mold.pth"

ALPHABET = ['B', 'I', 'U', 'D', 'X', 'W', 'w']
STOI = {s: i for i, s in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET)
SEQ_LEN = 64

app = typer.Typer(add_completion=False)

# ============================================
# 1. THE FETCHER (DUKASCOPY / WRAPPER)
# ============================================
class DataFetcher:
    """
    Handles fetching data.
    Note: Replace the 'fetch' method with your specific Dukascopy API logic.
    """
    def __init__(self, asset: str, timeframe: str, days: int):
        self.asset = asset.upper()
        self.tf = timeframe
        self.days = days

    def fetch(self, save_path: Path):
        typer.echo(f" Attempting to fetch {self.asset} ({self.tf}) for {self.days} days...")
        
        # --- REAL DUKASCOPY IMPLEMENTATION (Pseudo-code) ---
        # from dukascopy import Dukascopy
        # api = Dukascopy()
        # api.login("user", "pass") # If needed
        # df = api.get_data(self.asset, self.tf, start_date, end_date)
        # -------------------------------------------------
        
        # --- DEMO IMPLEMENTATION (Using yfinance for demonstration) ---
        # Remove this block when you implement real API
        try:
            import yfinance as yf
            # Map standard TFs to Yahoo TFs
            tf_map = {'M1': '1m', 'H1': '1h', 'D': '1d'}
            yf_tf = tf_map.get(self.tf, '1m')
            
            ticker = f"{self.asset[:3]}{self.asset[3:]}=X"
            start = (datetime.now() - timedelta(days=self.days)).strftime('%Y-%m-%d')
            
            typer.echo(f"   [Demo Mode] Using yfinance for {ticker}...")
            df = yf.download(ticker, start=start, interval=yf_tf, progress=False)
            
            # Standardize columns for our parser
            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'UTC', 'Datetime': 'UTC'}, inplace=True)
            
            # Ensure standard OHLCV
            df = df[['UTC', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            typer.echo(f" Fetch failed: {e}")
            raise e
        # -------------------------------------------------

        # Save to RAW
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        typer.echo(f" Saved raw data to {save_path}")

# ============================================
# 2. THE PARSER (From Previous Logic)
# ============================================
class SmartPoetEncoder:
    def __init__(self):
        pass
        
    def encode_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Normalize Headers
        df.columns = df.columns.str.strip().str.title()
        
        # ATR
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        df['ATR'] = atr
        
        # Logic
        tokens = []
        for i in range(len(df)):
            row = df.iloc[i]
            curr_atr = row['ATR']
            
            if pd.isna(curr_atr) or curr_atr <= 0:
                tokens.append(STOI['X'])
                continue
            
            o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
            body = abs(c - o)
            rng = h - l
            
            u_wick = h - max(o, c)
            l_wick = min(o, c) - l
            
            # Decision Logic
            if (u_wick / rng > 0.5 and u_wick > body*1.5): tok = 'W'
            elif (l_wick / rng > 0.5 and l_wick > body*1.5): tok = 'w'
            elif (body/rng < 0.1 and body/curr_atr < 0.3): tok = 'X'
            elif c > o:
                tok = 'B' if (body/curr_atr > 0.8) else 'U'
            else:
                tok = 'I' if (body/curr_atr > 0.8) else 'D'
                
            tokens.append(STOI[tok])
            
        df['TokenID'] = tokens
        df['Letter'] = [ALPHABET[t] for t in tokens]
        return df

    def to_windows(self, df: pd.DataFrame, seq_len=64) -> pd.DataFrame:
        seq = df['TokenID'].tolist()
        windows = []
        for i in range(len(seq) - seq_len):
            inp = seq[i : i + seq_len]
            tgt = seq[i + seq_len]
            windows.append({
                'input': ''.join([ALPHABET[t] for t in inp]),
                'target': ALPHABET[tgt]
            })
        return pd.DataFrame(windows)

# ============================================
# 3. THE MODEL (Nano Mold)
# ============================================
class NanoMold(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, 64)
        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, 64))
        
        layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=64*2, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=4)
        self.ln_f = nn.LayerNorm(64)
        self.head = nn.Linear(64, VOCAB_SIZE, bias=False)

    def forward(self, x):
        B, T = x.size()
        x = self.tok_emb(x) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.transformer(x, x, mask=mask)
        x = self.ln_f(x)
        return self.head(x)[:, -1, :]

# ============================================
# 4. MAIN CLI LOGIC
# ============================================

@app.command()
def run(
    asset: str = typer.Option(..., help="Asset (e.g., EURUSD)"),
    timeframe: str = typer.Option("M1", help="Timeframe (M1, H1, D)"),
    days: int = typer.Option(365, help="Number of days back to fetch"),
    mode: str = typer.Option("train", help="'train' (new) or 'retrain' (from scratch)")
):
    """ The full workflow: Download -> Parse -> Train"""
    
    # 1. Setup Paths
    raw_file = DATA_RAW / f"{asset}_{timeframe}.csv"
    train_file = DATA_TRAIN / f"{asset}_{timeframe}.csv"
    model_out = DATA_MODELS / f"vessels" / f"{asset}_{timeframe}.pth"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. FETCH
    typer.secho(f"\n[Phase 1] Fetching Data", fg=typer.colors.BRIGHT_CYAN)
    fetcher = DataFetcher(asset, timeframe, days)
    fetcher.fetch(raw_file)
    
    # 3. PARSE
    typer.secho(f"\n[Phase 2] Encoding & Generating Windows", fg=typer.colors.BRIGHT_CYAN)
    try:
        df_raw = pd.read_csv(raw_file)
    except FileNotFoundError:
        typer.echo(" Raw file not found!")
        raise typer.Exit()
        
    encoder = SmartPoetEncoder()
    df_encoded = encoder.encode_df(df_raw)
    df_windows = encoder.to_windows(df_encoded)
    df_windows.to_csv(train_file, index=False)
    typer.echo(f" Training data created: {len(df_windows)} samples")
    
    # 4. PREPARE MODEL
    typer.secho(f"\n[Phase 3] Loading Mold", fg=typer.colors.BRIGHT_CYAN)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"   Device: {device}")
    
    model = NanoMold().to(device)
    
    if mode == "train":
        if not VIRGIN_PATH.exists():
            typer.secho(f"  Virgin Mold not found at {VIRGIN_PATH}!", fg=typer.colors.YELLOW)
            typer.echo("   Train without it? (This is hard mode)")
            typer.echo("   Continuing...")
            # If we wanted to enforce it, we would raise typer.Abort()
        else:
            typer.echo(f" Loading Virgin Grammar from {VIRGIN_PATH}...")
            state = torch.load(VIRGIN_PATH, map_location=device)
            model.load_state_dict(state)
    elif mode == "retrain":
        typer.echo(" Initializing fresh model (Retrain mode)")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 5. TRAIN
    typer.secho(f"\n[Phase 4] Training Vessel", fg=typer.colors.BRIGHT_CYAN)
    
    # Simple Training Loop
    epochs = 10
    # Convert data to tensors (In real app, use DataLoader)
    inputs = torch.tensor([ [STOI[c] for c in s] for s in df_windows['input'] ]).long().to(device)
    targets = torch.tensor([ STOI[c] for c in df_windows['target'] ]).long().to(device)
    
    # Shuffle
    perm = torch.randperm(len(inputs))
    inputs, targets = inputs[perm], targets[perm]
    
    typer.echo("   Training started...")
    with typer.progressbar(range(epochs)) as progress:
        for epoch in progress:
            model.train()
            
            # Mini-batch (simplified)
            start_idx = 0
            batch_size = 32
            epoch_loss = 0
            
            for i in range(0, len(inputs), batch_size):
                batch_x = inputs[i:i+batch_size]
                batch_y = targets[i:i+batch_size]
                
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(inputs)//batch_size)
            progress.update(epoch, postfix={f"Loss: {avg_loss:.4f}"})

    # 6. SAVE
    typer.secho(f"\n[Phase 5] Saving Vessel", fg=typer.colors.BRIGHT_CYAN)
    torch.save(model.state_dict(), model_out)
    typer.echo(f" Model saved to: {model_out}")
    typer.echo(f"\n Process Complete!")

if __name__ == "__main__":
    app()
