import numpy as np

class PoetEncoder:
    ALPHABET = ['B', 'I', 'U', 'D', 'X', 'W', 'w']
    STOI = {s: i for i, s in enumerate(ALPHABET)}
    ITOS = {i: s for s, i in STOI.items()}
    
    def __init__(self, lookback_atr=14):
        self.lookback = lookback_atr

    def encode_candle(self, open_p, high_p, low_p, close_p, atr=None):
        body = abs(close_p - open_p)
        rng = max(high_p - low_p, 1e-9)
        ratio = body / rng
        
        u_wick = high_p - max(open_p, close_p)
        l_wick = min(open_p, close_p) - low_p
        
        if ratio < 0.1:
            token = 'X'
        elif close_p > open_p:
            token = 'B' if ratio > 0.6 else 'U'
        else:
            token = 'I' if ratio > 0.6 else 'D'

        if u_wick > rng * 0.6: token = 'W'
        if l_wick > rng * 0.6: token = 'w'

        return self.STOI[token]

    def encode_dataframe(self, df):
        # Wrapper to encode whole DF
        tokens = []
        # Ensure column names are lowercase as expected by the loop below
        df.columns = [c.lower() for c in df.columns]
        for i in range(len(df)):
            # Assuming df has high/low/open/close
            t = self.encode_candle(
                df['open'].iloc[i], 
                df['high'].iloc[i], 
                df['low'].iloc[i], 
                df['close'].iloc[i]
            )
            tokens.append(t)
        return np.array(tokens)
