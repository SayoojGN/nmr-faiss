# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import faiss
import pickle
import numpy as np
import io
import base64
from build_index import SpectrumEncoder, parse_nmr_spectrum, pad_features

app = FastAPI(title="nmr2smiles API")

# Allow local frontend to fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PredictRequest(BaseModel):
    nmr: str
    top_k: int = 5

# Global loader
def load_resources():
    # instantiate model with same hyperparams used when saved
    model = SpectrumEncoder(input_dim=4, embed_dim=128, hidden_dim=256, num_layers=2, nhead=4, max_len=64)
    model.load_state_dict(torch.load("StateData/spectrum_encoder.pt", map_location="cpu"))
    model.eval()

    index = faiss.read_index("StateData/nmr_index.faiss")
    with open("StateData/smiles_list.pkl", "rb") as f:
        smiles_list = pickle.load(f)

    return model, index, smiles_list

model, index, smiles_list = load_resources()

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.nmr
    k = max(1, min(20, int(req.top_k)))

    try:
        features = parse_nmr_spectrum(text)
        features = pad_features(features, max_len=64)
        q = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 4]

        with torch.no_grad():
            q_emb = model(q).numpy().astype("float32")

        distances, indices = index.search(q_emb, k)
        result_smiles = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(smiles_list):
                result_smiles.append("")
            else:
                result_smiles.append(smiles_list[idx])

        # Try render top-1 with rdkit if available
        image_data = None
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            top_smiles = result_smiles[0] if len(result_smiles) > 0 else None
            if top_smiles:
                mol = Chem.MolFromSmiles(top_smiles)
                if mol:
                    pil_img = Draw.MolToImage(mol, size=(420, 320))
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    image_data = "data:image/png;base64," + b64
        except Exception:
            # RDKit not present or image rendering failed; return None image
            image_data = None

        return {"smiles": result_smiles, "image": image_data}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {e}")
