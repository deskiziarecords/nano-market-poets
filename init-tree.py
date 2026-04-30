import os
from pathlib import Path

# Define the structure
structure = [
    "data/raw/EURUSD",
    "data/raw/USDCAD",
    "data/encoded/EURUSD",
    "data/training",
    "data/models/vessels",
    "data/models/rag_memory",
    "scripts",
    "src",
    "notebooks"
]

def create_tree(base_path="."):
    base = Path(base_path)
    print(f"🌱 Planting Nano-Poet Garden at: {base.absolute()}")
    
    for folder in structure:
        p = base / folder
        p.mkdir(parents=True, exist_ok=True)
        # Create a .gitkeep file so empty folders get saved in Git
        (p / ".gitkeep").touch()
        print(f"   📁 {p}")

    print("✅ Directory structure created.")
    
    # Create a README placeholder
    readme = base / "README.md"
    if not readme.exists():
        readme.write_text("# Nano-Market-Poets\n\nAutomated Trading with LLMs.")
        print("   📄 README.md created")

if __name__ == "__main__":
    create_tree()
