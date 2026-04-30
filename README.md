# nano-market-poets

``` text

nano-market-poets/
│
├── README.md
├── pyproject.toml
│
├── data/                     # THE "DATA" BOX
│   │
│   ├── raw/                 # Input: Fresh from MT5 / Broker
│   │   ├── EURUSD/
│   │   │   ├── 1m.csv
│   │   │   ├── 1d.csv
│   │   │   ├── 1w.csv
│   │   │   └── 1M.csv
│   │   │
│   │   └── USDCAD/
│   │       └── 1m.csv
│   │
│   ├── encoded/             # Intermediate: Just the text ('B', 'X', 'w'...)
│   │   ├── EURUSD/
│   │   │   ├── 1m_sequence.txt       # The full poem
│   │   │   └── 1m_windows.csv        # Sliding windows (optional intermediate)
│   │   │
│   │   └── USDCAD/
│   │       └── 1m_sequence.txt
│   │
│   ├── training/            # THE "TRAINING DATA" BOX (Ready for Model)
│   │   ├── virgin_mold.csv            # Mixed data from all assets
│   │   ├── eurusd_1m.csv            # Specific asset/timeframe datasets
│   │   ├── eurusd_1d.csv
│   │   └── usdcad_1m.csv
│   │
│   └── models/              # THE "OUTPUT" (The Poets)
│       ├── virgin_mold.pth            # The master grammar
│       ├── vessels/
│       │   ├── eurusd_1m.pth
│       │   ├── eurusd_1d.pth
│       │   └── usdcad_1m.pth
│       └── rag_memory/
│           ├── eurusd_1m_memory.pkl
│           └── usdcad_1m_memory.pkl
│
├── scripts/                 # WORKFLOW AUTOMATION
│   ├── 01_download_mt5.py   # Script to fill /data/raw
│   ├── 02_encode_poet.py     # Script to convert /raw -> /encoded -> /training
│   ├── 03_train_virgin.py   # Script to train on mixed /training
│   └── 04_forge_vessel.py  # Script to fine-tune virgin mold
│
├── src/                    # SOURCE CODE
│   ├── encoder.py
│   ├── model.py
│   └── mold.py
│
└── notebooks/               # FOR VISUALIZATION/DEBUGGING
    └── 01_examine_poems.ipynb

```
The `nano-poet.py` script is an implementation of a tiny GPT-style transformer model designed for financial market predictions based on historical sequences of market "letters." It employs a combination of deep learning and a memory-augmented retrieval system to predict potential market actions (BUY, SELL, or HOLD). The code is structured into several key components:

1. **Model Definition**: The `NanoMoldModel` class represents the core transformer model capable of processing sequences and making predictions.
2. **Memory System**: The `MarketMemory` class enables the model to retain and recall historical sequences and their outcomes for better prediction accuracy.
3. **Mold Management**: The `MoldManager` class provides functions to save, load, and clone models for different financial assets.
4. **Training Loop**: The `fine_tune_model` function is used to train the model on specific financial data.
5. **Live Execution**: The `live_step` function handles real-time decision-making based on model predictions and historical data recall.

## Detailed Components

### 1. Configuration

The configuration sets parameters for hardware, model dimensions, and vocabulary:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENS = ['B', 'I', 'U', 'D', 'X', 'W', 'w']
VOCAB_SIZE = len(TOKENS)
STOI = {s:i for i,s in enumerate(TOKENS)}  # String to index mapping
ITOS = {i:s for s,i in STOI.items()}         # Index to string mapping

# Model Dimensions
D_MODEL = 128             # Size of model embeddings
N_HEADS = 4               # Number of self-attention heads
N_LAYERS = 4              # Number of transformer decoder layers
CTX_LEN = 64              # Maximum context length
DROPOUT = 0.1             # Dropout rate
```

### 2. The NanoMoldModel (Transformer Architecture)

The `NanoMoldModel` class defines the transformer architecture used for predicting the next token in a sequence:

```python
class NanoMoldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)  # Token embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, CTX_LEN, D_MODEL))  # Positional embeddings
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 2,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=N_LAYERS)
        self.ln_f = nn.LayerNorm(D_MODEL)  # Layer normalization
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)  # Output layer

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, idx):
        # Forward pass function
        # idx: [Batch, SeqLen]
        ...
```

#### Key Functions:
- **`init_weights`**: Initializes model weights using a normal distribution.
- **`forward`**: Processes input sequences through embeddings, adds positional encodings, and passes through the transformer layers. Outputs the logits for the predicted next token.

### 3. Vector Memory (MarketMemory)

The `MarketMemory` class stores sequences and their outcomes, allowing the model to "remember" past events:

```python
class MarketMemory:
    def __init__(self, model):
        self.model = model.eval()
        self.memory_db = []  # List to store historical data
    
    def remember(self, sequence, next_token):
        # Store vector representation of a given sequence along with the outcome
        ...
        
    def recall(self, sequence, top_k=5):
        # Retrieve similar historical patterns and their outcomes for a given sequence
        ...
```

#### Key Functions:
- **`remember`**: Converts a sequence into its vector representation and stores it with the associated outcome.
- **`recall`**: Computes cosine similarities between the current sequence and stored sequences, returning the most similar matches and their outcome distributions.

### 4. Mold Manager (MoldManager)

The `MoldManager` class manages the creation and manipulation of model instances:

```python
class MoldManager:
    def __init__(self):
        self.virgin_mold = NanoMoldModel().to(DEVICE)  # Base model instance
    
    def save_virgin(self, path="market_virgin_mold.pth"):
        ...
    
    def load_virgin(self, path="market_virgin_mold.pth"):
        ...
    
    def clone_for_asset(self, asset_name="EURUSD_1M"):
        ...
```

#### Key Functions:
- **`save_virgin`**: Saves the initial model state for future cloning.
- **`clone_for_asset`**: Creates a clone of the base model with cleared weights for asset-specific training.

### 5. Training Loop (fine_tune_model)

The `fine_tune_model` function handles the training of models on financial data:

```python
def fine_tune_model(model, data_file, epochs=5):
    ...
```

#### Key Features:
- **Training**: Uses AdamW optimizer and processes sequences of tokens for training.
- **Loss Calculation**: Computes loss using cross-entropy for the predictions made by the model.

### 6. Live Execution Strategy (live_step)

The `live_step` function is used to make live predictions using the model and the memory:

```python
def live_step(model, memory, current_sequence):
    ...
```

#### Key Functions:
- **Model Prediction**: Generates the next predicted token based on the current sequence.
- **RAG (Retrieval-Augmented Generation)**: Searches historical memory for patterns that match the current context.
- **Decision Logic**: Combines predictions from the model and memory to return a final action (BUY, SELL, or HOLD).

## Usage Example

To use the script, you can follow these steps in the `__main__` section:

```python
if __name__ == "__main__":
    manager = MoldManager()
    
    # A. Train the virgin model once and save it
    # fine_tune_model(manager.virgin_mold, "mixed_data.pkl", epochs=10)
    # manager.save_virgin()
    
    # B. Clone and train for a specific asset (e.g., EURUSD)
    eurusd_model, eurusd_memory = manager.clone_for_asset("EURUSD_1M")
    
    # fine_tune_model(eurusd_model, "eurusd_1m_data.pkl", epochs=20)
    
    # C. Run live simulation with historical data
    live_seq = [STOI['D'], STOI['D'], STOI['X'], STOI['w'], STOI['U'], STOI['B']]
    
    # Remember successful/failed trades in memory
    # eurusd_memory.remember([0, 1, 2], STOI['D'])  # Example entry

    decision = live_step(eurusd_model, eurusd_memory, live_seq)
    print(f"Final Action: {decision}")
```

This structure provides a way to train a predictive model, maintain historical context, and make live trade predictions based on learned behaviors.

## Conclusion

The `nano-poet.py` script represents a concise and modular implementation of a market prediction model that leverages the strengths of transformer architectures in combination with a memory-augmented retrieval system. The architecture is designed to be easily extendable for various financial assets and training data, making it suitable for real-world applications in algorithmic trading and financial analysis.

---
