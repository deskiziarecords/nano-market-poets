import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from .encoder import PoetEncoder

class PoetryAnthology:
    """
    Stores patterns (Stanzas) and their outcomes (Verses).
    Searches for 'Rhymes' (similar patterns).
    """
    def __init__(self, model):
        self.model = model.eval()
        self.db = [] # List of dicts: {'fingerprint': vec, 'outcome': token_id}

    def _get_fingerprint(self, stanza):
        """Get the hidden state vector of the current stanza"""
        with torch.no_grad():
            t_stanza = torch.tensor([stanza]).long().to(next(self.model.parameters()).device)
            
            # Simplified embedding extraction
            x = self.model.tok_emb(t_stanza) + self.model.pos_emb[:, :len(stanza), :]
            mask = nn.Transformer.generate_square_subsequent_mask(len(stanza)).to(x.device)

            # Note: In our NanoMold, we pass x as both target and memory
            # For TransformerDecoder, mask is passed as tgt_mask
            x = self.model.transformer(x, x, tgt_mask=mask)
            x = self.model.ln_f(x)
            
            return x[0, -1, :].cpu().numpy()

    def write_poem(self, stanza, outcome_token):
        """Store a pattern and what happened next"""
        vec = self._get_fingerprint(stanza)
        self.db.append({'fingerprint': vec, 'outcome': outcome_token})

    def find_rhymes(self, stanza, top_k=10):
        """Search for similar patterns in history"""
        if not self.db: return None, {}
        
        query = self._get_fingerprint(stanza).reshape(1, -1)
        all_vecs = np.array([d['fingerprint'] for d in self.db])
        
        # Cosine Similarity
        sims = cosine_similarity(query, all_vecs)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        matches = [self.db[i] for i in top_indices]
        
        # Count outcomes
        outcome_counts = {}
        for m in matches:
            token = PoetEncoder.ITOS[m['outcome']]
            outcome_counts[token] = outcome_counts.get(token, 0) + 1
            
        return matches, outcome_counts
