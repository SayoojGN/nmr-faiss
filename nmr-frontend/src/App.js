import React from "react";
import Header from "./components/Header";
import Footer from "./components/Footer";
import MainPanel from "./components/MainPanel";

export default function App() {
  return (
    <div className="app-root">
      <Header />
      <MainPanel />
      <Footer />
    </div>
  );
}
