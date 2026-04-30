import typer
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime, timedelta
import time

# Import Core Components
from nano_market_poets.core.model import NanoMold
from nano_market_poets.core.encoder import PoetEncoder

# ============================================
# CONFIGURATION
# ============================================
DATA_RAW = Path("data/raw")
DATA_TRAIN = Path("data/training")
DATA_MODELS = Path("data/models")
VIRGIN_PATH = DATA_MODELS / "virgin_mold.pth"

ALPHABET = PoetEncoder.ALPHABET
STOI = PoetEncoder.STOI
VOCAB_SIZE = len(ALPHABET)
SEQ_LEN = 64

app = typer.Typer(add_completion=False)

# ============================================
# 1. FIXED FETCHER (Handles YF Limits)
# ============================================
class DataFetcher:
    def __init__(self, asset: str, timeframe: str, days: int):
        self.asset = asset.upper()
        self.tf = timeframe
        self.days = days


    def fetch(self, save_path: Path):
        """Universal Fetcher using Dukascopy"""
        typer.echo(f"🔌 Fetching {self.asset} ({self.tf}) from Dukascopy...")

        try:
            # NOTE: You must have installed dukascopy-python
            # pip install dukascopy-python
            import dukascopy_python
            from dukascopy_python import instruments
            
            # --- Mapping Logic (CLI Arg -> Dukascopy Constant) ---
            asset_map = {
                'EURUSD': instruments.FX_MAJORS_EUR_USD,
                'GBPUSD': instruments.FX_MAJORS_GBP_USD,
                'USDJPY': instruments.FX_MAJORS_USD_JPY,
                # Add others here...
            }
            
            tf_map = {
                'M1': dukascopy_python.INTERVAL_MINUTE_1,
                'M5': dukascopy_python.INTERVAL_MINUTE_5,
                'M15': dukascopy_python.INTERVAL_MINUTE_15,
                'H1': dukascopy_python.INTERVAL_HOUR_1,
                'H4': dukascopy_python.INTERVAL_HOUR_4,
                'D': dukascopy_python.INTERVAL_DAY_1,
            }

            instrument_class = asset_map.get(self.asset.upper())
            if not instrument_class:
                typer.secho(f"❌ Asset {self.asset} not configured in patch.", fg=typer.colors.RED)
                typer.secho(f"   Add it to asset_map in patch.py", fg=typer.colors.RED)
                return

            interval_class = tf_map.get(self.tf.upper())
            if not interval_class:
                typer.secho(f"❌ Timeframe {self.tf} not configured in patch.", fg=typer.colors.RED)
                return

            # --- Date Logic ---
            end = datetime.now()
            start = end - timedelta(days=self.days)
            
            # Fix system time issue (if clock is set to future)
            if end > datetime.utcnow() + timedelta(days=1):
                end = datetime.utcnow()
                start = end - timedelta(days=self.days)

            # --- Fetch ---
            typer.echo(f"   Range: {start.date()} to {end.date()}")
            
            df = dukascopy_python.fetch(
                instrument=instrument_class,
                interval=interval_class,
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=start,
                end=end,
            )
            
            # --- Formatting for Poet Parser ---
            df.reset_index(inplace=True)
            # Standardize columns to match parser expectation
            df.rename(columns={'Date': 'UTC', 'Datetime': 'UTC', 'timestamp': 'UTC'}, inplace=True)
            df = df[['UTC', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # --- Save ---
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            typer.secho(f"✅ Saved {len(df)} bars to {save_path}", fg=typer.colors.GREEN)

        except ImportError:
            typer.secho(f"❌ 'dukascopy-python' library not found.", fg=typer.colors.RED)
            typer.secho(f"   Run: pip install dukascopy-python", fg=typer.colors.YELLOW)
        except Exception as e:
            typer.secho(f"❌ Dukascopy fetch failed: {e}", fg=typer.colors.RED)
            raise e
    def __init__(self):
        self.core_encoder = PoetEncoder()
        
    def encode_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Normalize Headers
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.title()
        
        # 2. Force Numeric Conversion
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaNs in price data
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        # 3. Use core encoder for tokenization
        tokens = []
        for i in range(len(df)):
            row = df.iloc[i]
            tok_id = self.core_encoder.encode_candle(
                row['Open'], row['High'], row['Low'], row['Close']
            )
            tokens.append(tok_id)
            
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
# 3. MAIN CLI
# ============================================

@app.command()
def run(
    asset: str = typer.Option(..., help="Asset (e.g., EURUSD)"),
    timeframe: str = typer.Option("M1", help="Timeframe (M1, H1, D)"),
    days: int = typer.Option(365, help="Number of days back to fetch"),
    mode: str = typer.Option("train", help="'train' (new) or 'retrain' (from scratch)")
):
    """🚀 The full workflow: Download -> Parse -> Train"""
    
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
        typer.echo("❌ Raw file not found!")
        raise typer.Exit()
        
    encoder = PoetCLIEncoder()
    df_encoded = encoder.encode_df(df_raw)
    df_windows = encoder.to_windows(df_encoded)
    df_windows.to_csv(train_file, index=False)
    typer.echo(f"✅ Training data created: {len(df_windows)} samples")
    
    # 4. PREPARE MODEL
    typer.secho(f"\n[Phase 3] Loading Mold", fg=typer.colors.BRIGHT_CYAN)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"   Device: {device}")
    
    # Use standardized d_model=128 from Core
    model = NanoMold(vocab_size=VOCAB_SIZE, d_model=128).to(device)
    
    if mode == "train":
        if not VIRGIN_PATH.exists():
            typer.secho(f"⚠️  Virgin Mold not found. Training from scratch (Hard Mode).", fg=typer.colors.YELLOW)
        else:
            typer.echo(f"📜 Loading Virgin Grammar...")
            state = torch.load(VIRGIN_PATH, map_location=device)
            model.load_state_dict(state)
    elif mode == "retrain":
        typer.echo("🆕 Initializing fresh model")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 5. TRAIN
    typer.secho(f"\n[Phase 4] Training Vessel", fg=typer.colors.BRIGHT_CYAN)
    
    epochs = 10
    if len(df_windows) == 0:
        typer.secho("❌ No training samples generated!", fg=typer.colors.RED)
        raise typer.Exit()

    inputs = torch.tensor([ [STOI[c] for c in s] for s in df_windows['input'] ]).long().to(device)
    targets = torch.tensor([ STOI[c] for c in df_windows['target'] ]).long().to(device)
    
    perm = torch.randperm(len(inputs))
    inputs, targets = inputs[perm], targets[perm]
    
    typer.echo("   Training started...")
    with typer.progressbar(range(epochs)) as progress:
        for epoch in progress:
            model.train()
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
            
            avg_loss = epoch_loss / (len(inputs)//batch_size if (len(inputs)//batch_size) > 0 else 1)
            progress.update(epoch, postfix={f"Loss: {avg_loss:.4f}"})

    # 6. SAVE
    typer.secho(f"\n[Phase 5] Saving Vessel", fg=typer.colors.BRIGHT_CYAN)
    torch.save(model.state_dict(), model_out)
    typer.echo(f"✅ Model saved to: {model_out}")
    typer.echo(f"\n🎉 Process Complete!")

if __name__ == "__main__":
    app()
