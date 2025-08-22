import streamlit as st
import torch
import numpy as np
import faiss
import pickle
from build_index import SpectrumEncoder, parse_nmr_spectrum, pad_features  # import your functions

# -------------------------
# Load model & data
# -------------------------
@st.cache_resource
def load_model_and_index():
    # Load model
    model = SpectrumEncoder(input_dim=4, embed_dim=128, hidden_dim=256, num_layers=2, nhead=4, max_len=64)
    model.load_state_dict(torch.load("StateData/spectrum_encoder.pt", map_location="cpu"))
    model.eval()

    # Load FAISS index
    index = faiss.read_index("StateData/nmr_index.faiss")

    # Load SMILES list
    with open("StateData/smiles_list.pkl", "rb") as f:
        smiles_list = pickle.load(f)

    return model, index, smiles_list

model, index, smiles_list = load_model_and_index()

# -------------------------
# Streamlit UI
# -------------------------
st.title("NMR Analyzer")
st.write("Enter a **1H NMR spectrum string**, and retrieve the closest SMILES + structure.")

# Input box
nmr_input = st.text_area("Paste your 1H NMR spectrum string:", height=150)

top_k = st.slider("Number of results (k):", 1, 10, 5)

if st.button("🔍 Search") and nmr_input.strip():
    try:
        # 1. Parse spectrum
        features = parse_nmr_spectrum(nmr_input)
        features = pad_features(features, max_len=64)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 4]

        # 2. Encode
        with torch.no_grad():
            q_emb = model(x).numpy().astype("float32")

        # 3. Search FAISS
        distances, indices = index.search(q_emb, top_k)

        st.subheader("Top Matches")
        for rank, idx in enumerate(indices[0]):
            smi = smiles_list[idx]
            dist = distances[0][rank]
            st.write(f"**Rank {rank+1}** — SMILES: `{smi}` (distance={dist:.4f})")

            # Show molecule using RDKit
            if rank == 0:
                try:
                    from rdkit import Chem
                    from rdkit.Chem import Draw
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, caption=f"Structure for {smi}")
                except Exception as e:
                    st.warning(f"Could not render structure for {smi}. Error: {e}")

    except Exception as e:
        st.error(f"❌ Error processing input: {e}")
