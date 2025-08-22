import torch
import numpy as np
import pickle
import faiss
import re

from build_index import SpectrumEncoder, parse_nmr_spectrum, pad_features

# -----------------------------
# Load saved data
# -----------------------------
index = faiss.read_index("StateData/nmr_index.faiss")

with open("StateData/smiles_list.pkl", "rb") as f:
    smiles_list = pickle.load(f)

print("✅ Loaded index with", index.ntotal, "spectra")
model = SpectrumEncoder()
model.load_state_dict(torch.load("StateData/spectrum_encoder.pt"))
model.eval()

query_text = "1H NMR (CDCl3, 400 MHz) δH = 8.71 (d, J = 1.2 Hz, 1H), 8.53–8.63 (m, 5H), 7.63–7.67 (m, 5H), 2.24 (s, 1H), 1.72 (s, 6H) ppm"

query_features = parse_nmr_spectrum(query_text)
query_features = pad_features(query_features, max_len=64)
q = torch.tensor(query_features, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    q_emb = model(q).numpy().astype("float32")

# Search top-5
k = 5
distances, indices = index.search(q_emb, k)

print("🔎 Top candidates:")
for rank, idx in enumerate(indices[0]):
    print(f"Rank {rank+1}: SMILES = {smiles_list[idx]}, distance = {distances[0][rank]:.4f}")
