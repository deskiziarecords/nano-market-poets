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
