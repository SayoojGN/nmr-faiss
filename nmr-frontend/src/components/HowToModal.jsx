import React from "react";

export default function HowToModal({ onClose }){
  const example = `1H NMR (400 MHz, CDCl3): δ 2.15 (s, 3H), 2.28 (s, 3H), 2.65 (ABd, 2H, JAB = 16.5 Hz)
13C NMR (100 MHz, CDCl3): δ 21.1, 22.5, 34.2, 135.8`;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e)=>e.stopPropagation()}>
        <h3>How to format your data?</h3>
        <p>Copy-paste your ¹H NMR peak listings. The parser looks for patterns like <code>δ 2.15 (s, 3H)</code>.</p>
        <ul>
          <li>Include multiplicity in parentheses if available (s, d, t, q, m, dd ...)</li>
          <li>Integration is useful (eg. <code>3H</code>)</li>
          <li>J-couplings (e.g. <code>J = 7.2 Hz</code>) are optional</li>
        </ul>
        <div className="example">
          <div className="example-label">Example ipenput</div>
          <pre>{example}</pre>
        </div>
        <div className="modal-actions">
          <button className="btn-primary" onClick={onClose}>Got it</button>
        </div>
      </div>
    </div>
  );
}
