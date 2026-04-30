import sys
sys.path.append('..')
import torch
import numpy as np

from nano_market_poets.core.mold import MoldManager
from nano_market_poets.core.memory import PoetryAnthology
from nano_market_poets.core.encoder import PoetEncoder

def live_trading():
    # 1. Setup
    manager = MoldManager()
    manager.load_virgin("data/molds/virgin_mold.pth")
    
    # 2. Load Fine-tuned Vessel (Assume you trained EURUSD already)
    vessel = manager.forge_vessel("EURUSD")
    vessel.load_state_dict(torch.load("data/molds/eurusd_vessel.pth"))
    vessel.eval()
    
    # 3. Setup Memory
    anthology = PoetryAnthology(vessel)
    # anthology.load("data/memory/eurusd_memory.pkl") # Load past RAG data
    
    encoder = PoetEncoder()
    
    # 4. Live Loop (Mock)
    # In reality, this loops with OnTick
    current_stanza = [0] * 64 # Last 64 tokens
    
    print("Nano Poet is listening...")
    
    # Simulate a new candle coming in
    new_candle_token = 1 # Random token for demo
    
    current_stanza.append(new_candle_token)
    current_stanza.pop(0)
    
    # 5. Generate Verse (Prediction)
    t_stanza = torch.tensor([current_stanza])
    with torch.no_grad():
        logits = vessel(t_stanza)
        probs = torch.softmax(logits, dim=-1).numpy()[0]
    
    pred_token = np.argmax(probs)
    pred_letter = encoder.ITOS[pred_token]
    
    # 6. Check for Rhymes (RAG)
    matches, outcomes = anthology.find_rhymes(current_stanza, top_k=10)
    
    signal = "HOLD"
    if matches:
        print(f"Rhyme found: {outcomes}")
        # Logic: If 8 out of 10 similar patterns resulted in 'B', Override model
        if outcomes.get('B', 0) >= 8:
            signal = "STRONG BUY"
        elif outcomes.get('D', 0) >= 8:
            signal = "STRONG SELL"
    elif probs[pred_token] > 0.8:
        # High confidence from model alone
        signal = "BUY" if pred_token < 3 else "SELL"
        
    print(f"Verse: {pred_letter} | Signal: {signal}")

if __name__ == "__main__":
    live_trading()
