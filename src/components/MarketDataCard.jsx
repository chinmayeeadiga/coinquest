import React, { useState, useEffect } from "react";
import "./market.css";

function MarketDataCard() {
  const [data, setData] = useState({
    price: 0,
    change: 0,
    volume: 0,
  });

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const ngrokUrl = "https://1917286b732f.ngrok-free.app"; // Replace with your ngrok URL

  const fetchMarketData = async () => {
    try {
      setLoading(true);
      setError(null);

      // ✅ Correct backend endpoint
      const response = await fetch(`${ngrokUrl}/api/market_data/btc`, {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // ✅ Map backend fields to frontend state
      setData({
        price: result.current_price,
        change: result.price_change_24h,
        volume: result.volume_24h,
      });
    } catch (err) {
      console.error("Error fetching market data:", err);
      setError("Failed to fetch market data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMarketData();

    // Optional: auto-refresh every 15 seconds
    const interval = setInterval(fetchMarketData, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="market-card">
        Loading market data...
      </div>
    );
  }

  if (error) {
    return (
      <div className="market-card">
        {error}
      </div>
    );
  }

  return (
    <div className="market-card">
      <h2 className="text-lg font-bold text-white">MARKET DATA</h2>
      <p>BTC/USD: ${data.price?.toLocaleString()}</p>
      <p>24h Change: {data.change?.toFixed(2)}%</p>
      <p>Volume: ${data.volume?.toLocaleString()}</p>
    </div>
  );
}

export default MarketDataCard;
