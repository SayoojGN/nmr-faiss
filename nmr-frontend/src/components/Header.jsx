import React from "react";

export default function Header(){
  return (
    <header className="site-header">
      <div className="header-inner">
        <div className="brand">
          <div className="logo">NMR ANALYZER</div>
          <div className="tag">Predict SMILES from NMR spectra</div>
        </div>
        <nav className="nav">
          <a href="#" onClick={(e)=>e.preventDefault()}>Home</a>
          <a href="#" onClick={(e)=>e.preventDefault()}>Docs</a>
          <a href="https://github.com/SayoojGN/nmr-faiss" target="_blank" rel="noreferrer">Repo</a>
        </nav>
      </div>
    </header>
  );
}
