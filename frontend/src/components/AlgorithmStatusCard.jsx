import React, { useState, useEffect } from "react";
import "./algo.css";

function AlgorithmStatusCard() {
  const [status, setStatus] = useState({
    core: "Loading...",
    risk: "Loading...",
    dataFeed: "Loading...",
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const apiUrl = import.meta.env.VITE_API_URL;

  const fetchAlgorithmStatus = async () => {
    try {
      setLoading(true);
      setError(null);

      // ✅ Correct endpoint
      const response = await fetch(`${apiUrl}/api/algorithm_status`, {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      setStatus({
        core: result.core_engine || "N/A",
        risk: result.risk_monitor || "N/A",
        dataFeed: result.data_feed || "N/A",
      });
    } catch (err) {
      console.error("Error fetching algorithm status:", err);
      setError("Failed to fetch algorithm status");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlgorithmStatus();
    const interval = setInterval(fetchAlgorithmStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div className="algorithm-card">Loading algorithm status...</div>;
  }

  if (error) {
    return <div className="algorithm-card">{error}</div>;
  }

  return (
    <div className="algorithm-card">
      <h2 className="text-lg font-bold text-white">ALGORITHM STATUS</h2>
      <ul className="text-white">
        <li>Core Engine: {status.core}</li>
        <li>Risk Monitor: {status.risk}</li>
        <li>Data Feed: {status.dataFeed}</li>
      </ul>
    </div>
  );
}

export default AlgorithmStatusCard;