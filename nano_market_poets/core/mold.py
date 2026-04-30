import torch
import copy
from .model import NanoMold

class MoldManager:
    def __init__(self, vocab_size=7):
        self.vocab_size = vocab_size
        self.virgin_mold = NanoMold(vocab_size)

    def save_virgin(self, path="data/molds/virgin_mold.pth"):
        torch.save(self.virgin_mold.state_dict(), path)
        print(f"Virgin Mold saved: {path}")

    def load_virgin(self, path):
        self.virgin_mold.load_state_dict(torch.load(path, map_location='cpu'))
        self.virgin_mold.eval()
        print(f"Virgin Mold loaded from: {path}")

    def forge_vessel(self, asset_name="EURUSD_1M"):
        """
        Clone the virgin mold.
        The vessel starts with the 'Universal Grammar' but can be 
        fine-tuned to speak a specific 'Dialect' (Asset).
        """
        vessel = NanoMold(self.vocab_size)
        vessel.load_state_dict(self.virgin_mold.state_dict())
        print(f"Forged new vessel: {asset_name}")
        return vessel
