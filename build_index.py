import torch
import torch.nn as nn
import re
import numpy as np
import pandas as pd
import faiss
import torch
import numpy as np
import pickle

MULTIPLICITY_MAP = {
    "s": 0, "d": 1, "t": 2, "q": 3, "dd": 4, "m": 5
}

def parse_nmr_spectrum(text):
    """
    Parse 1H NMR spectrum string into numerical features.
    Returns: numpy array [num_peaks, 4]
    """
    peaks = []
    
    # Split by spaces before numbers like "δ 3.50" and keep groups
    matches = re.findall(r"([\d\.]+)\s*\(([^)]+)\)", text)
    
    for delta, details in matches:
        delta = float(delta)
        
        # Extract multiplicity
        mult = None
        for m in MULTIPLICITY_MAP:
            if m in details:
                mult = MULTIPLICITY_MAP[m]
                break
        if mult is None:
            mult = -1  # unknown
        
        # Extract integration (#H)
        integ_match = re.search(r"(\d+)H", details)
        integration = int(integ_match.group(1)) if integ_match else 1
        
        # Extract J-coupling values
        J_match = re.findall(r"J\s*=\s*([\d\.]+)", details)
        J_vals = [float(j) for j in J_match]
        J_mean = np.mean(J_vals) if J_vals else 0.0
        
        peaks.append([delta, mult, integration, J_mean])
    
    return np.array(peaks, dtype=np.float32)

class SpectrumEncoder(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, hidden_dim=256, num_layers=2, nhead=4, max_len=20):
        """
        input_dim: number of features per signal (δ, multiplicity, integration, J)
        embed_dim: embedding dimension per signal
        hidden_dim: transformer hidden size
        num_layers: number of transformer encoder layers
        nhead: number of attention heads
        max_len: max number of NMR signals per spectrum
        """
        super(SpectrumEncoder, self).__init__()
        
        self.input_fc = nn.Linear(input_dim, embed_dim)  # project input features
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool across sequence length

    def forward(self, x):
        """
        x: tensor of shape [batch_size, seq_len, input_dim]
        """
        # Project to embedding space
        x = self.input_fc(x)  # [B, seq_len, embed_dim]
        
        # Transformer expects [seq_len, batch, embed_dim]
        x = x.permute(1, 0, 2)
        
        # Pass through transformer
        x = self.transformer(x)  # [seq_len, batch, embed_dim]
        
        # Permute back
        x = x.permute(1, 2, 0)  # [batch, embed_dim, seq_len]
        
        # Pool across sequence length
        x = self.pool(x).squeeze(-1)  # [batch, embed_dim]
        
        return x
    
max_len = 64
def pad_features(features, max_len=64):
    features = np.array(features, dtype=np.float32)
    if features.shape[0] >= max_len:
        return features[:max_len]
    else:
        pad_len = max_len - features.shape[0]
        padding = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
        return np.vstack([features, padding])


if __name__ == "__main__":
    
    model = SpectrumEncoder(input_dim=4, embed_dim=128, hidden_dim=256, num_layers=2, nhead=4, max_len=20)
    torch.save(model.state_dict(), "StateData/spectrum_encoder.pt")

    df = pd.read_excel("sample2.xlsx")

    # creating a list of embeddings of the spectral data in the excel file
    # creating a list of SMILES strings for mapping

    all_embeddings = []
    smiles_list = []

    for i, row in df.iterrows():
        spectrum_text = row["1H NMR chemical shifts"]
        smiles = row["SMILES"]

        try:
            features = parse_nmr_spectrum(spectrum_text)
            features = pad_features(features, max_len=64)
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 4]

            with torch.no_grad():
                embedding = model(x)  # [1, 128]

            all_embeddings.append(embedding.squeeze(0).numpy())
            smiles_list.append(smiles)

        except Exception as e:
            print(f"⚠️ Error at row {i}: {e}")
            continue

    all_embeddings = np.vstack(all_embeddings).astype("float32")

    # build FAISS index
    d = all_embeddings.shape[1]   # embedding dimension (e.g. 128)
    index = faiss.IndexFlatL2(d)  # L2 similarity
    index.add(all_embeddings)

    faiss.write_index(index, "StateData/nmr_index.faiss")

    # Save SMILES mapping
    with open("StateData/smiles_list.pkl", "wb") as f:
        pickle.dump(smiles_list, f)

    # Also save raw embeddings for debugging/rebuilding
    np.save("StateData/all_embeddings.npy", all_embeddings)

    print("✅ Saved index with", index.ntotal, "spectra")
