import React from "react";
import { useNavigate } from "react-router-dom"; // <-- import
import MarketDataCard from "../components/MarketDataCard";
import AlgorithmStatusCard from "../components/AlgorithmStatusCard";
import PortfolioCard from "../components/PortfolioCard";
import SystemMonitorCard from "../components/SystemMonitorCard";
import { GlareCard } from "../components/ui/glare-card";
import { TypewriterEffect } from "../components/ui/typewriter";
import logo from "../assets/logo.png";
import "./home.css";

function Home() {
  const navigate = useNavigate(); // <-- define navigate

  return (
    <div className="home-container">
  {/* Logo */}
  <img src={logo} alt="Logo" className="logo" />

  {/* Cards */}
  
 

  {/* Center content */}
  <div className="center-content">
    <TypewriterEffect
      words={[{ text: "COINQUEST", className: "text-white" }]}
      className="coinquest-title"
      cursorClassName="bg-blue-500"
    />

    <div className="button-group">
      <button className="watch-btn" onClick={() => navigate("/crypto")}>Watch</button>
      <button className="trade-btn" onClick={() => navigate("/trade")}>Trade</button>
    </div>
    
    <MarketDataCard title="Top Left" content="This is the top left card" position="top-left" />
     <PortfolioCard title="Bottom Left" content="This is the bottom left card" position="bottom-left" />
  <AlgorithmStatusCard title="Top Right" content="This is the top right card" position="bottom-right" />
  </div>
</div>

  );
}

export default Home;
