import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# ============================================
# 1. SMART ENCODER (Copied here for standalone use)
# ============================================
class SmartPoetEncoder:
    ALPHABET = ['B', 'I', 'U', 'D', 'X', 'W', 'w']
    STOI = {s: i for i, s in enumerate(ALPHABET)}
    ITOS = {i: s for s, i in STOI.items()}
    
    def __init__(self, atr_period=14):
        self.atr_period = atr_period

    def compute_atr(self, df):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        return atr

    def encode_candle(self, row, current_atr):
        if pd.isna(current_atr) or current_atr <= 0:
            return self.STOI['X'], 'X'

        open_p, high, low, close = row['Open'], row['High'], row['Low'], row['Close']
        
        body = close - open_p
        body_size = abs(body)
        total_range = high - low
        
        if total_range < 1e-10: 
            return self.STOI['X'], 'X'

        u_wick = high - max(open_p, close)
        l_wick = min(open_p, close) - low
        
        u_wick_ratio = u_wick / total_range
        l_wick_ratio = l_wick / total_range
        body_ratio = body_size / total_range
        
        strength_factor = body_size / current_atr
        
        # Decision Tree
        is_upper_rejection = (u_wick_ratio > 0.5) and (u_wick > body_size * 1.5)
        is_lower_rejection = (l_wick_ratio > 0.5) and (l_wick > body_size * 1.5)
        
        if is_upper_rejection: return self.STOI['W'], 'W'
        if is_lower_rejection: return self.STOI['w'], 'w'

        if body_ratio < 0.1 and strength_factor < 0.3:
            return self.STOI['X'], 'X'
            
        if body > 0: 
            return (self.STOI['B'], 'B') if strength_factor > 0.8 else (self.STOI['U'], 'U')
        else: 
            return (self.STOI['I'], 'I') if strength_factor > 0.8 else (self.STOI['D'], 'D')

    def process_dataframe(self, df):
        df = df.copy()
        # Robust column normalization (Case insensitive)
        col_map = {c: c.title() for c in df.columns}
        df.rename(columns=col_map, inplace=True)
        
        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close']
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}. Found: {df.columns}")
        
        # Calculate ATR
        df['ATR'] = self.compute_atr(df)
        
        results = []
        for i in range(len(df)):
            row = df.iloc[i]
            tid, letter = self.encode_candle(row, row['ATR'])
            results.append(letter)
        
        df['Letter'] = results
        return df

# ============================================
# 2. DATA BUILDER (Sliding Windows)
# ============================================
def build_poem_windows(token_sequence, window_size=64):
    """
    Converts a sequence of letters into training examples.
    """
    windows = []
    seq_len = len(token_sequence)
    
    # We iterate until we can fit a full window + 1 target
    for i in range(seq_len - window_size):
        inputs = token_sequence[i : i + window_size]
        target = token_sequence[i + window_size]
        
        windows.append({
            'input': ''.join(inputs),
            'target': target
        })
    return windows

# ============================================
# 3. MAIN PARSER
# ============================================
def parse_raw_csv(input_path, output_path, window_size=64, dayfirst=True):
    print(f"🔨 Parsing {input_path}...")
    
    # 1. Read CSV
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    # 2. Clean Headers and Dates
    # Strip whitespace (e.g. ' UTC')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.title()
    
    # Parse Date
    # Format: 05.12.2025 12:00:00.000 UTC
    # We remove "UTC" and parse the string
    if 'Utc' in df.columns:
        df['Utc'] = df['Utc'].str.replace(' UTC', '')
        
        # Try to infer format. 
        # If dayfirst=True (05.12 is Dec 5th). 
        # If your data is US format (05.12 is May 12th), set dayfirst=False
        try:
            df['Time'] = pd.to_datetime(df['Utc'], dayfirst=dayfirst, utc=True)
        except Exception:
            # Fallback inference
            df['Time'] = pd.to_datetime(df['Utc'], utc=True)
            
        df.set_index('Time', inplace=True)
    else:
        print("⚠️  No 'UTC' column found. Assuming default index.")

    # 3. Encode (Raw Data -> Poem)
    print("📝 Encoding Candles (Applying ATR & Logic)...")
    encoder = SmartPoetEncoder()
    df_encoded = encoder.process_dataframe(df)
    
    # 4. Generate Windows (Poem -> Training Data)
    print(f"📦 Generating {window_size}-bar Windows...")
    sequence = df_encoded['Letter'].tolist()
    
    # We must skip the first N bars because ATR is NaN during warmup
    # The encoder returns 'X' for ATR warmup, but we can drop those if we want strict data.
    # Here we keep them as 'X' (Neutral), which is safe.
    
    windows = build_poem_windows(sequence, window_size)
    
    # 5. Save Output
    out_df = pd.DataFrame(windows)
    out_df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Saved {len(out_df)} samples to {output_path}")
    
    # Quick stats
    print(f"   Sequence Length: {len(sequence)} candles")
    print(f"   Window Size: {window_size}")
    
    dist = out_df['target'].value_counts(normalize=True)
    print("   Target Distribution:")
    for token in encoder.ALPHABET:
        print(f"      {token}: {dist.get(token, 0):.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse OHLCV to Nano-Poet Training Data")
    parser.add_argument('input_file', help="Raw OHLCV CSV file")
    parser.add_argument('-o', '--output', default='training_data.csv', help="Output training CSV")
    parser.add_argument('-w', '--window', type=int, default=64, help="Sequence window length")
    parser.add_argument('--dayfirst', action='store_true', help="Treat date as DD.MM.YYYY instead of MM.DD.YYYY")
    
    args = parser.parse_args()
    
    parse_raw_csv(args.input_file, args.output, args.window, args.dayfirst)
