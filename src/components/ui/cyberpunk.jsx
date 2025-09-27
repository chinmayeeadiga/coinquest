import React from "react";
import "./cyberpunk.css"; // We'll put animations here

export default function CyberpunkBackground() {
  return (
    <div className="cyber-bg">
      {/* Glowing gradient overlay */}
      <div className="glow-overlay"></div>

      {/* Moving neon lines */}
      {[...Array(20)].map((_, i) => (
        <div
          key={i}
          className="neon-line"
          style={{ top: `${i * 5}%`, left: `${-100 + i * 10}%` }}
        ></div>
      ))}

      {/* Floating neon dots */}
      {[...Array(50)].map((_, i) => (
        <div
          key={i}
          className="neon-dot"
          style={{
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
            width: `${Math.random() * 4 + 2}px`,
            height: `${Math.random() * 4 + 2}px`,
          }}
        ></div>
      ))}
    </div>
  );
}
