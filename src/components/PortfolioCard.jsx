import React, { useState, useEffect } from "react";

function PortfolioCard() {
  const [portfolio, setPortfolio] = useState({
    balance: 0,
    unrealized: 0,
    positions: 0,
    buyingPower: 0,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const ngrokUrl = "https://de6890d58456.ngrok-free.app"; // Replace with your ngrok URL

  const fetchPortfolio = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${ngrokUrl}/api/portfolio`, {
        headers: { "ngrok-skip-browser-warning": "true" },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      setPortfolio({
        balance: result.total_balance || 0,
        unrealized: result.unrealized_pnl || 0,
        positions: result.positions_count || 0,
        buyingPower: result.buying_power || 0,
      });
    } catch (err) {
      console.error("Error fetching portfolio:", err);
      setError("Failed to fetch portfolio data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="absolute bottom-4 left-4 border border-[#ff6f61] rounded p-4 w-60 shadow-lg text-white bg-gray-900/50">
        Loading portfolio...
      </div>
    );
  }

  if (error) {
    return (
      <div className="absolute bottom-4 left-4 border border-red-500 rounded p-4 w-60 shadow-lg text-white bg-gray-900/50">
        {error}
      </div>
    );
  }

  return (
    <div className="absolute bottom-4 left-4 border border-[#ff6f61] rounded p-4 w-60 shadow-lg text-white bg-gray-900/50">
      <h2 className="text-lg font-bold">PORTFOLIO ANALYSIS</h2>
      <p>Total Balance: ${portfolio.balance?.toLocaleString()}</p>
      <p>Unrealized P&L: ${portfolio.unrealized?.toLocaleString()}</p>
      <p>Open Positions: {portfolio.positions}</p>
      <p>Buying Power: ${portfolio.buyingPower?.toLocaleString()}</p>
    </div>
  );
}

export default PortfolioCard;
