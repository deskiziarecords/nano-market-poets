import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# CONFIGURATION - THE "PURIST" SETTINGS
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The 7 "Letters" of the market
TOKENS = ['B', 'I', 'U', 'D', 'X', 'W', 'w']
VOCAB_SIZE = len(TOKENS)
STOI = {s:i for i,s in enumerate(TOKENS)}
ITOS = {i:s for s,i in STOI.items()}

# Model Dimensions (Tiny - fits in RAM easily)
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
CTX_LEN = 64   # How far back the "Eye" looks
DROPOUT = 0.1

# ============================================
# 1. THE NANO-TRANSFORMER (GPT-Style Decoder)
# ============================================
class NanoMoldModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. The "Word" Embeddings
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        
        # 2. The "Position" Awareness (Order matters)
        self.pos_emb = nn.Parameter(torch.zeros(1, CTX_LEN, D_MODEL))
        
        # 3. The Brain (Transformer Decoder)
        # Using Decoder because we want to predict the NEXT token based on PAST tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 2,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=N_LAYERS)
        
        # 4. The Voice (Linear projection to vocab)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, idx):
        """
        idx: [Batch, SeqLen]
        """
        b, t = idx.size()
        assert t <= CTX_LEN, "Sequence too long"

        # Embedding + Position
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :t, :]
        x = tok_emb + pos_emb
        
        # Standard GPT causal mask (can't see the future)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(DEVICE)
        
        # Transformer Pass
        # Note: We pass x as both target and memory because it's a decoder-only 
        # architecture with masking, effectively acting as a decoder-only GPT.
        x = self.transformer(x, x, mask=mask) # Self-attention with mask
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits[:, -1, :] # Return logits for the PREDICTED next token

# ============================================
# 2. VECTOR MEMORY (The "RAG" System)
# ============================================
class MarketMemory:
    """
    Stores sequences of tokens and their subsequent outcomes.
    Allows the model to 'remember' historical probability of success.
    """
    def __init__(self, model):
        self.model = model.eval()
        self.memory_db = [] # List of { 'vec': np.array, 'outcome': 'B'/'D' }
    
    def remember(self, sequence, next_token):
        """
        sequence: list of ints (the context)
        next_token: the actual token that happened (ground truth)
        """
        # 1. Get the internal representation (Vector) of this sequence
        # We extract the last hidden state before the head
        with torch.no_grad():
            t_seq = torch.tensor([sequence]).long().to(DEVICE)
            tok_emb = self.model.tok_emb(t_seq)
            pos_emb = self.model.pos_emb[:, :len(sequence), :]
            x = tok_emb + pos_emb
            
            # Run through transformer
            mask = nn.Transformer.generate_square_subsequent_mask(len(sequence)).to(DEVICE)
            x = self.model.transformer(x, x, mask=mask)
            x = self.model.ln_f(x)
            
            # Take the last token's vector as the "Fingerprint"
            vec = x[0, -1, :].cpu().numpy()
            
        self.memory_db.append({'vec': vec, 'outcome': next_token})
        
    def recall(self, sequence, top_k=5):
        """
        Given a current sequence, search memory for similar patterns in history.
        """
        if len(self.memory_db) == 0:
            return None, {}

        # 1. Vectorize current sequence
        with torch.no_grad():
            t_seq = torch.tensor([sequence]).long().to(DEVICE)
            tok_emb = self.model.tok_emb(t_seq)
            pos_emb = self.model.pos_emb[:, :len(sequence), :]
            x = tok_emb + pos_emb
            mask = nn.Transformer.generate_square_subsequent_mask(len(sequence)).to(DEVICE)
            x = self.model.transformer(x, x, mask=mask)
            x = self.model.ln_f(x)
            query_vec = x[0, -1, :].cpu().numpy().reshape(1, -1)

        # 2. Compute Cosine Similarity against DB
        all_vecs = np.array([m['vec'] for m in self.memory_db])
        similarities = cosine_similarity(query_vec, all_vecs)[0]
        
        # 3. Get Top K matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        matches = []
        outcome_dist = {'B':0, 'I':0, 'U':0, 'D':0, 'X':0, 'W':0, 'w':0}
        
        for idx in top_indices:
            matches.append(self.memory_db[idx])
            outcome = self.memory_db[idx]['outcome']
            outcome_dist[ITOS[outcome]] += 1
            
        return matches, outcome_dist

# ============================================
# 3. MOLD MANAGER
# ============================================
class MoldManager:
    def __init__(self):
        self.virgin_mold = NanoMoldModel().to(DEVICE)
        
    def save_virgin(self, path="market_virgin_mold.pth"):
        torch.save(self.virgin_mold.state_dict(), path)
        print(f"Virgin Mold saved to {path}")

    def load_virgin(self, path="market_virgin_mold.pth"):
        self.virgin_mold.load_state_dict(torch.load(path))
        self.virgin_mold.eval()
        print("Virgin Mold loaded.")

    def clone_for_asset(self, asset_name="EURUSD_1M"):
        """
        Creates a fresh copy of the mold for a specific asset.
        This preserves the grammar, but clears the fine-tuned weights.
        """
        new_model = NanoMoldModel().to(DEVICE)
        new_model.load_state_dict(self.virgin_mold.state_dict())
        print(f"Cloned Mold for: {asset_name}")
        return new_model, MarketMemory(new_model)

# ============================================
# 4. TRAINING LOOP (For Fine-Tuning)
# ============================================
def fine_tune_model(model, data_file, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Assuming data is a list of integer sequences
    # data = [[0, 1, 2, ...], [3, 4, 5...]]
    with open(data_file, 'rb') as f:
        sequences = pickle.load(f)
        
    print(f"Fine-tuning on {len(sequences)} sequences...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq in sequences:
            # Input: seq[:-1], Target: seq[1:]
            x = torch.tensor([seq[:-1]]).long().to(DEVICE)
            y = torch.tensor([seq[1:]]).long().to(DEVICE)
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss/len(sequences):.4f}")

    model.eval()
    print("Fine-tuning complete.")

# ============================================
# 5. LIVE EXECUTION STRATEGY
# ============================================
def live_step(model, memory, current_sequence):
    """
    1. Model predicts based on probability.
    2. RAG searches history for similar patterns.
    3. Combine both to decide.
    """
    # 1. Pure Model Prediction
    t_seq = torch.tensor([current_sequence[-CTX_LEN:]]).long().to(DEVICE)
    with torch.no_grad():
        logits = model(t_seq)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    
    pred_token_id = np.argmax(probs)
    pred_letter = ITOS[pred_token_id]
    
    # 2. RAG Check
    matches, outcome_dist = memory.recall(current_sequence[-CTX_LEN:], top_k=10)
    
    print(f"Model Sees: {pred_letter} (Conf: {probs[pred_token_id]:.2f})")
    
    if matches:
        print(f"RAG found {len(matches)} similar patterns in history.")
        # Check for consensus
        if outcome_dist['B'] > 7: # 70% of matches resulted in Bull
            print(">>> RAG OVERRIDE: STRONG BUY")
            return "BUY"
        elif outcome_dist['D'] > 7:
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
    # 1. Create Manager
    manager = MoldManager()
    
    # A. TRAIN THE VIRGIN MOLD (Do this once on mixed data)
    # (Assuming you have a mixed_data.pkl)
    # fine_tune_model(manager.virgin_mold, "mixed_data.pkl", epochs=10)
    # manager.save_virgin()
    
    # B. CLONE AND TRAIN FOR EURUSD 1M
    eurusd_model, eurusd_memory = manager.clone_for_asset("EURUSD_1M")
    
    # (Assuming you have eurusd_1m_data.pkl)
    # fine_tune_model(eurusd_model, "eurusd_1m_data.pkl", epochs=20)
    
    # C. LIVE SIMULATION
    # Let's say the market sequence was: D, D, X, w, U, B
    live_seq = [STOI['D'], STOI['D'], STOI['X'], STOI['w'], STOI['U'], STOI['B']]
    
    # Before running live, populate memory with known successful/failed trades
    # eurusd_memory.remember([0,1,2], STOI['D']) ... etc ...
    
    decision = live_step(eurusd_model, eurusd_memory, live_seq)
    print(f"Final Action: {decision}")
