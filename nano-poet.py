import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

from nano_market_poets.core.model import NanoMold
from nano_market_poets.core.encoder import PoetEncoder
from nano_market_poets.core.memory import PoetryAnthology

# ============================================
# CONFIGURATION
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHABET = PoetEncoder.ALPHABET
VOCAB_SIZE = len(ALPHABET)
STOI = PoetEncoder.STOI
ITOS = PoetEncoder.ITOS

# Standardized Core Dimensions
D_MODEL = 128
CTX_LEN = 64

# ============================================
# 1. MOLD MANAGER (Integrated)
# ============================================
class MoldManager:
    def __init__(self):
        self.virgin_mold = NanoMold(vocab_size=VOCAB_SIZE, d_model=D_MODEL).to(DEVICE)
        
    def save_virgin(self, path="data/models/virgin_mold.pth"):
        torch.save(self.virgin_mold.state_dict(), path)
        print(f"Virgin Mold saved to {path}")

    def load_virgin(self, path="data/models/virgin_mold.pth"):
        self.virgin_mold.load_state_dict(torch.load(path, map_location=DEVICE))
        self.virgin_mold.eval()
        print("Virgin Mold loaded.")

    def clone_for_asset(self, asset_name="EURUSD_1M"):
        new_model = NanoMold(vocab_size=VOCAB_SIZE, d_model=D_MODEL).to(DEVICE)
        new_model.load_state_dict(self.virgin_mold.state_dict())
        print(f"Cloned Mold for: {asset_name}")
        return new_model, PoetryAnthology(new_model)

# ============================================
# 2. TRAINING LOOP
# ============================================
def fine_tune_model(model, data_file, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    try:
        with open(data_file, 'rb') as f:
            sequences = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Data file {data_file} not found.")
        return
        
    print(f"Fine-tuning on {len(sequences)} sequences...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq in sequences:
            # seq is list of ints
            if len(seq) < 2: continue

            x = torch.tensor([seq[:-1]]).long().to(DEVICE)
            y = torch.tensor([seq[1:]]).long().to(DEVICE)
            
            # Note: Core NanoMold forward returns logits for the NEXT token of the sequence.
            # If we want to train on the whole sequence, we might need a slightly different forward
            # or call it multiple times. For simplicity in this demo, let's assume we train
            # to predict the very last token from the preceding sequence.

            logits = model(x) # [1, VOCAB_SIZE]
            target = y[:, -1] # [1] - The last token of the sequence

            loss = F.cross_entropy(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss/max(1, len(sequences)):.4f}")

    model.eval()
    print("Fine-tuning complete.")

# ============================================
# 3. LIVE EXECUTION STRATEGY
# ============================================
def live_step(model, memory, current_sequence):
    # 1. Pure Model Prediction
    t_seq = torch.tensor([current_sequence[-CTX_LEN:]]).long().to(DEVICE)
    with torch.no_grad():
        logits = model(t_seq)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    
    pred_token_id = np.argmax(probs)
    pred_letter = ITOS[pred_token_id]
    
    # 2. RAG Check (using core PoetryAnthology)
    matches, outcome_dist = memory.find_rhymes(current_sequence[-CTX_LEN:], top_k=10)
    
    print(f"Model Sees: {pred_letter} (Conf: {probs[pred_token_id]:.2f})")
    
    if matches:
        print(f"RAG found {len(matches)} similar patterns in history.")
        # Check for consensus
        if outcome_dist.get('B', 0) > 7:
            print(">>> RAG OVERRIDE: STRONG BUY")
            return "BUY"
        elif outcome_dist.get('D', 0) > 7:
            print(">>> RAG OVERRIDE: STRONG SELL")
            return "SELL"
            
    # 3. Final Decision Logic
    if pred_letter in ['B', 'w'] and probs[pred_token_id] > 0.6:
        return "BUY"
    elif pred_letter in ['I', 'W'] and probs[pred_token_id] > 0.6:
        return "SELL"
    
    return "HOLD"

# ============================================
# USAGE EXAMPLE
# ============================================
if __name__ == "__main__":
    manager = MoldManager()
    
    # Clone and train for EURUSD 1M
    eurusd_model, eurusd_memory = manager.clone_for_asset("EURUSD_1M")
    
    # Mock sequence for demo
    live_seq = [STOI['D'], STOI['D'], STOI['X'], STOI['w'], STOI['U'], STOI['B']]
    
    # Populate memory for demo
    eurusd_memory.write_poem(live_seq[:3], STOI['D'])
    
    decision = live_step(eurusd_model, eurusd_memory, live_seq)
    print(f"Final Action: {decision}")
