import pandas as pd
import numpy as np

class SmartPoetEncoder:
    """
    A robust encoder that classifies candles based on Relative Shape and Volatility (ATR).
    """
    ALPHABET = ['B', 'I', 'U', 'D', 'X', 'W', 'w']
    STOI = {s: i for i, s in enumerate(ALPHABET)}
    ITOS = {i: s for s, i in STOI.items()}
    
    def __init__(self, atr_period=14):
        self.atr_period = atr_period

    def compute_atr(self, df):
        """Calculates ATR and appends to dataframe"""
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
        """
        Encodes a single row based on relative metrics.
        Returns: Token ID (int), Token Letter (str), Debug Dict
        """
        if pd.isna(current_atr) or current_atr <= 0:
            return self.STOI['X'], 'X', {'reason': 'No Vol Data'}

        open_p, high, low, close = row['Open'], row['High'], row['Low'], row['Close']
        
        # 1. Core Geometry
        body = close - open_p
        body_size = abs(body)
        total_range = high - low
        
        # Prevent division by zero
        if total_range < 1e-10: 
            return self.STOI['X'], 'X', {'reason': 'Zero Range'}

        # 2. Wick Analysis (as percentage of total range)
        upper_wick = high - max(open_p, close)
        lower_wick = min(open_p, close) - low
        
        u_wick_ratio = upper_wick / total_range
        l_wick_ratio = lower_wick / total_range
        body_ratio = body_size / total_range
        
        # 3. Volatility Analysis (How "Strong" is the move?)
        # A strong candle should be at least 0.8x the ATR
        strength_factor = body_size / current_atr
        
        # 4. Logic Hierarchy (Decision Tree)
        
        # A. REJECTION PRIORITY (Pinbars)
        # A long wick must be > 50% of range AND > 1.5x the body
        is_upper_rejection = (u_wick_ratio > 0.5) and (upper_wick > body_size * 1.5)
        is_lower_rejection = (l_wick_ratio > 0.5) and (lower_wick > body_size * 1.5)
        
        if is_upper_rejection:
            return self.STOI['W'], 'W', {'desc': 'Bearish Rejection', 'u_ratio': round(u_wick_ratio,2)}
        if is_lower_rejection:
            return self.STOI['w'], 'w', {'desc': 'Bullish Rejection', 'l_ratio': round(l_wick_ratio,2)}

        # B. INDECISION (Doji)
        # Small body relative to range (< 10%) AND small body relative to volatility
        if body_ratio < 0.1 and strength_factor < 0.3:
            return self.STOI['X'], 'X', {'desc': 'Doji/Indecision', 'body_ratio': round(body_ratio,2)}
            
        # C. DIRECTIONAL (Bullish vs Bearish)
        if body > 0: # Bullish
            if strength_factor > 0.8:
                return self.STOI['B'], 'B', {'desc': 'Strong Bull', 'strength': round(strength_factor,2)}
            else:
                return self.STOI['U'], 'U', {'desc': 'Weak Bull', 'strength': round(strength_factor,2)}
        else: # Bearish
            if strength_factor > 0.8:
                return self.STOI['I'], 'I', {'desc': 'Strong Bear', 'strength': round(strength_factor,2)}
            else:
                return self.STOI['D'], 'D', {'desc': 'Weak Bear', 'strength': round(strength_factor,2)}

    def process_dataframe(self, df):
        """
        Runs encoder on the whole dataframe.
        """
        df = df.copy()
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.title()
        
        # Normalize Columns (Robustness)
        col_map = {'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'}
        df.rename(columns={c: col_map[c] for c in col_map if c in df.columns}, inplace=True)
        
        # Calculate ATR
        df['ATR'] = self.compute_atr(df)
        
        results = []
        for i in range(len(df)):
            row = df.iloc[i]
            tid, letter, debug = self.encode_candle(row, row['ATR'])
            results.append({
                'Letter': letter,
                'TokenID': tid,
                'Debug': str(debug)
            })
        
        df_encoded = pd.concat([df, pd.DataFrame(results)], axis=1)
        return df_encoded
