import React from "react";
import { useNavigate } from "react-router-dom"; // <-- import
import MarketDataCard from "../components/MarketDataCard";
import AlgorithmStatusCard from "../components/AlgorithmStatusCard";
import PortfolioCard from "../components/PortfolioCard";
import SystemMonitorCard from "../components/SystemMonitorCard";
import { GlareCard } from "../components/ui/glare-card";
import { TypewriterEffect } from "../components/ui/typewriter";
import logo from "../assets/logo.png";

function Home() {
  const navigate = useNavigate(); // <-- define navigate

  return (
    <div className="relative h-screen w-screen">
      <img
        src={logo} // <-- replace with your image path
        alt="Logo"
        className="absolute top-4 left-4 w-16 h-16 sm:w-20 sm:h-20 md:w-30 md:h-30 object-cover rounded-2xl shadow-lg"
      />
      <MarketDataCard title="Top Left" content="This is the top left card" position="top-left" />
      <AlgorithmStatusCard title="Top Right" content="This is the top right card" position="top-right" />
      <PortfolioCard title="Bottom Left" content="This is the bottom left card" position="bottom-left" />

      <div className="absolute inset-0 flex flex-col items-center justify-center space-y-6">
  {/* Title */}
  <TypewriterEffect
        words={[{ text: "COINQUEST", className: "text-white" }]}
        className="text-2xl md:text-7xl lg:text-9xl font-bold"
        cursorClassName="bg-blue-500"
      />

  {/* Buttons */}
  <div className="flex space-x-4">
  <button
    className="bg-red-800 hover:bg-red-700 text-white text-lg px-6 py-3 rounded-xl font-semibold"
    onClick={() => navigate("/crypto")}
  >
    Watch
  </button>
  
  <button
    className="bg-red-950 hover:bg-red-900 text-white text-lg px-6 py-3 rounded-xl font-semibold"
    onClick={() => navigate("/trade")}
  >
    Trade
  </button>
  
  
</div>


      </div>
      
    </div>
  );
}

export default Home;
