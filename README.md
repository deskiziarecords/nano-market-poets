# nano-market-poets

``` text

nano-market-poets/
├── nano_market_poets/       # Main Source Code
│   ├── __init__.py          # Package initializer
│   ├── core/                # The "Brain" (Model, RAG, Encoder)
│   │   ├── __init__.py
│   │   ├── encoder.py       # Candlestick -> Letter (The Grammar)
│   │   ├── model.py         # The Nano-Transformer (The Brain)
│   │   ├── memory.py        # Vector DB / RAG (The Wisdom)
│   │   └── mold.py          # Mold Manager (The Factory)
│   ├── strategies/          # Grid & Execution logic
│   │   ├── __init__.py
│   │   └── micro_grid.py
│   └── utils/               # Helpers
│       ├── __init__.py
│       └── config.py
├── scripts/                 # Workflow Automation
│   ├── forge_virgin.py     # Train the master model
│   ├── scribe_vessel.py    # Fine-tune for specific asset
│   └── recite.py           # Live trading loop
├── data/                    # Storage for molds and memory
│   ├── molds/
│   └── memory/
├── pyproject.toml           # Project dependencies
└── README.md


```
