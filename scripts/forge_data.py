#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.append('..') # Import the repo package

from nano_market_poets.core.smart_encoder import SmartPoetEncoder

class DatasetBuilder:
    def __init__(self, encoder, window_size=64, lookahead=1):
        self.encoder = encoder
        self.window_size = window_size
        self.lookahead = lookahead

    def sliding_windows(self, token_array):
        """
        Converts a sequence of tokens into training samples.
        Input: [T1, T2, ... Tn]
        Output: List of {input: [...Tn-1], target: [Tn]}
        """
        windows = []
        seq_len = len(token_array)
        
        for i in range(seq_len - self.window_size - self.lookahead + 1):
            inputs = token_array[i : i + self.window_size]
            targets = token_array[i + self.window_size : i + self.window_size + self.lookahead]
            
            windows.append({
                'input_sequence': inputs.tolist(),
                'target_sequence': targets.tolist()
            })
        return windows

    def process_files(self, input_path, output_dir):
        p = Path(input_path)
        files = []
        
        if p.is_file() and p.suffix == '.csv':
            files = [p]
        elif p.is_dir():
            files = sorted(p.glob('*.csv'))
        else:
            print(f" Invalid path: {input_path}")
            return

        all_tokens = []
        total_candles = 0
        
        print(f" Found {len(files)} file(s) to forge...")
        
        # 1. Merge and Encode
        for f in files:
            try:
                print(f"  Reading: {f.name}")
                df = pd.read_csv(f)
                
                # Encode
                df_enc = self.encoder.process_dataframe(df)
                
                # Append tokens (skip NaN rows from ATR init)
                valid_tokens = df_enc['TokenID'].dropna().astype(int).tolist()
                all_tokens.extend(valid_tokens)
                total_candles += len(df_enc)
                
            except Exception as e:
                print(f"    Error reading {f.name}: {e}")

        print(f"\n Stats: {total_candles} raw candles -> {len(all_tokens)} valid tokens")
        
        # 2. Generate Training Windows
        print(f" Constructing Sliding Windows (Size={self.window_size})...")
        training_data = self.sliding_windows(np.array(all_tokens))
        
        # 3. Save Outputs
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for easy Python loading
        import json
        with open(out / 'training_windows.json', 'w') as f:
            json.dump(training_data, f)
            
        # Save as text for human reading
        letters = [self.encoder.ITOS[t] for t in all_tokens]
        with open(out / 'pattern_sequence.txt', 'w') as f:
            f.write(''.join(letters))
            
        # Save metadata
        meta = {
            'total_tokens': len(all_tokens),
            'training_samples': len(training_data),
            'vocab_size': 7,
            'window_size': self.window_size,
            'unique_patterns': len(set(''.join(letters))) # Rough estimate
        }
        with open(out / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
            
        print(f" Forging Complete.")
        print(f"   Output: {out}")
        print(f"   Samples: {meta['training_samples']}")

def main():
    parser = argparse.ArgumentParser(description="Forge Training Data for Nano Poets")
    parser.add_argument('input', help="CSV file or directory containing CSVs")
    parser.add_argument('-o', '--output', default='./data/forge_output', help="Output directory")
    parser.add_argument('-w', '--window', type=int, default=64, help="Sequence length")
    parser.add_argument('-l', '--lookahead', type=int, default=1, help="Prediction horizon")
    
    args = parser.parse_args()
    
    # Initialize the Smart Encoder
    encoder = SmartPoetEncoder(atr_period=14)
    
    # Initialize Builder
    builder = DatasetBuilder(encoder, args.window, args.lookahead)
    
    # Run
    builder.process_files(args.input, args.output)

if __name__ == "__main__":
    main()
