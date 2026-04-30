import sys
sys.path.append('..')

from nano_market_poets.core.mold import MoldManager
from nano_market_poets.core.model import NanoMold
from nano_market_poets.core.encoder import PoetEncoder
import torch
import torch.nn.functional as F

def train(model, data_path, epochs=10):
    # Load generic market data (Mixed Forex, Crypto, etc.)
    # Assuming you have a pre-generated list of sequences
    # sequences = [..., [0, 1, 2], [3, 4, 5]...]
    
    # Mock Data for example
    sequences = [torch.randint(0, 7, (64,)) for _ in range(1000)]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    for epoch in range(epochs):
        for seq in sequences:
            x = seq[:-1].unsqueeze(0)
            y = seq[1:].unsqueeze(0)
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Virgin Mold Epoch {epoch}: Loss {loss.item():.4f}")

if __name__ == "__main__":
    manager = MoldManager()
    train(manager.virgin_mold, "data/market_mixed.pkl")
    manager.save_virgin()
