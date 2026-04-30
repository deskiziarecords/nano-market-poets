import pandas as pd
import numpy as np
from collections import Counter
import sys

# ============================================
# CONFIGURATION
# ============================================
INPUT_CSV = "training_data.csv" # The file you provided
NEW_WINDOW_SIZE = 64          # Scale up from 10 to 64
NEW_OUTPUT_CSV = "training_data_fixed.csv"

# Mapping (must match your SmartEncoder)
ALPHABET = ['B', 'I', 'U', 'D', 'X', 'W', 'w']
STOI = {s: i for i, s in enumerate(ALPHABET)}

# ============================================
# 1. RECONSTRUCT SEQUENCE
# ============================================
print("🔍 Reading input CSV...")
df = pd.read_csv(INPUT_CSV)

# Reconstruct the full sequence from the first row's input + all targets
# Note: This assumes the CSV contains the entire continuous sequence
first_input = df.iloc[0]['input']
all_targets = df['target'].tolist()

# Stitch them together
full_sequence_string = first_input + ''.join(all_targets)
tokens = [STOI[char] for char in full_sequence_string]

print(f"   Total Sequence Length: {len(tokens)} candles")

# ============================================
# 2. AUDIT DISTRIBUTION (The "Boring" Check)
# ============================================
dist = Counter(full_sequence_string)
print("\n📊 Token Distribution:")
for char in ALPHABET:
    count = dist.get(char, 0)
    pct = (count / len(full_sequence_string)) * 100
    print(f"   {char}: {count:6} ({pct:5.2f}%)")

if dist['X'] > 0.6 * len(full_sequence_string):
    print("⚠️  WARNING: Market is too quiet (Too many X's). The model might be useless.")
else:
    print("✅ Distribution looks healthy.")

# ============================================
# 3. REGENERATE WINDOWS WITH LARGER CONTEXT
# ============================================
print(f"\n🔨 Regenerating windows with size {NEW_WINDOW_SIZE}...")

new_rows = []
num_samples = len(tokens) - NEW_WINDOW_SIZE - 1

for i in range(num_samples):
    input_seq = tokens[i : i + NEW_WINDOW_SIZE]
    target_seq = tokens[i + NEW_WINDOW_SIZE] # Predict next 1
    
    # Convert back to string for readability (optional)
    input_str = ''.join([ALPHABET[t] for t in input_seq])
    target_str = ALPHABET[target_seq]
    
    new_rows.append({
        'position': i,
        'input': input_str,
        'target': target_str
    })

new_df = pd.DataFrame(new_rows)
new_df.to_csv(NEW_OUTPUT_CSV, index=False)

print(f"✅ Saved {len(new_df)} training samples to {NEW_OUTPUT_CSV}")
print("   Ready for training NanoMold.")
