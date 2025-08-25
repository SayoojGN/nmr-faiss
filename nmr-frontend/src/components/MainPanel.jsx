import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import HowToModal from "./HowToModal";

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000/predict";

export default function MainPanel(){
  const [nmrText, setNmrText] = useState("");
  const [loading, setLoading] = useState(false);
  const [smiles, setSmiles] = useState([]);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const canvasRef = useRef(null);
  const topSmilesRef = useRef("");
  const [image, setImage] = useState(null);


    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        setSmiles([]);
        setImage(null);
        try {
            const resp = await axios.post(API_URL, { nmr: nmrText, top_k: 5 }, { timeout: 120000 });
            const result = resp.data;            
            if (!result || !result.smiles) {
            setError("No results returned from server.");
            } else {
            setSmiles(result.smiles);
            setImage(result.image); // 👈 use backend RDKit image
            }
        } catch (err) {
            console.error(err);
            setError(err?.response?.data?.detail || err.message || "Request failed");
        } finally {
            setLoading(false);
        }
    };

  const copyToClipboard = async (s) => {
    try {
      await navigator.clipboard.writeText(s || "");
      // small feedback could be added
    } catch {}
  };

  return (
  <main className="main-container w-full flex justify-center m-10">
    <section className="hero-card w-full max-w-6xl p-6 flex flex-col gap-6">
      
      {/* Row 1: Input Card */}
      <div className="input-card bg-white p-6 mb-10 rounded-xl shadow">
        <h2 className="text-2xl font-semibold mb-2">Predict SMILES from NMR data</h2>
        <p className="muted mb-4">
          Paste your ¹H NMR peak text. Click <strong>Predict</strong>.
        </p>

        <textarea
          className="nmr-input w-full border rounded-md p-3"
          placeholder="Example: 1H NMR (400 MHz, CDCl3): 2.15 (s, 3H), 2.28 (s, 3H), 2.43 (ABd, 2H, J = 16.5 Hz)"
          value={nmrText}
          onChange={(e) => setNmrText(e.target.value)}
          rows={6}
        />

        <div className="controls mt-4 flex gap-3">
          <button className="btn-primary" onClick={handlePredict} disabled={loading}>
            {loading ? "Predicting…" : "Predict SMILES"}
          </button>
          <button className="btn-ghost" onClick={() => setShowModal(true)}>
            How to format your data?
          </button>
        </div>

        {error && <div className="error mt-2">{error}</div>}
      </div>

      {/* Row 2: Outputs (side by side) */}
      <div className="output-row grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Structure Card */}
        <div className="mol-card bg-white p-4 rounded-xl shadow">
          <div className="mol-title font-medium mb-2">Top predicted structure</div>
          <div className="mol-canvas-wrap flex justify-center">
            {image ? (
                <div className="flex flex-row">                
                <img src={image} alt="molecule" className="rounded-lg border max-w-full h-auto" />
                <text
                    variant="subtitle2" 
                    style={{ color: "#bdbabaff", marginBottom: "8px" }}
                    >
                    {smiles[0]}
                </text>
                </div>
            ) : (
                <div className="muted">No prediction yet</div>
            )}
            </div>
        </div>

        {/* SMILES List */}
        <div className="smiles-list bg-white p-4 rounded-xl shadow">
          <div className="list-title font-medium mb-2">Top 5 predicted SMILES</div>
          {smiles && smiles.length ? (
            smiles.map((smi, idx) => (
              <div
                key={idx}
                className={
                  "smiles-item flex justify-between items-center p-2 border-b " +
                  (idx === 0 ? "bg-gray-50 font-semibold" : "")
                }
              >
                <div className="smi-text break-all">{smi || "(empty)"}</div>
                <button
                  className="copy-btn ml-2 text-sm text-blue-600"
                  onClick={() => copyToClipboard(smi)}
                >
                  Copy
                </button>
              </div>
            ))
          ) : (
            <div className="muted">No predictions yet</div>
          )}
        </div>
      </div>

      {showModal && <HowToModal onClose={() => setShowModal(false)} />}
    </section>
  </main>
);

}