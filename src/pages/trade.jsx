import React, { useState, useEffect } from "react";
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  X,
  DollarSign,
  BarChart3,
  Activity,
} from "lucide-react";

export default function Trade() {
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState("checking");
  const [error, setError] = useState(null);
  const [processingTrade, setProcessingTrade] = useState(null);
  const [portfolioMetrics, setPortfolioMetrics] = useState(null);

  // API detection - same logic as crypto component
  const possibleUrls = [import.meta.env.VITE_API_URL];

  const findWorkingApi = async () => {
    for (const baseUrl of possibleUrls) {
      try {
        const response = await fetch(`${baseUrl}/ping`, {
          method: "GET",
          headers: {
            "ngrok-skip-browser-warning": "true",
            "Content-Type": "application/json",
          },
        });
        if (response.ok) {
          return baseUrl;
        }
      } catch (err) {
        continue;
      }
    }
    return null;
  };

  const fetchTrades = async () => {
    try {
      setError(null);
      const apiUrl = await findWorkingApi();

      if (!apiUrl) {
        setApiStatus("disconnected");
        // Use mock data when API is not available
        const mockTrades = [
          {
            id: 1,
            symbol: "BTC",
            current_price: 43890.5,
            target_price: 45500.0,
            stop_loss: 42800.0,
            action: "BUY",
            confidence: 0.72,
            risk_score: 0.25,
            reasoning:
              "Technical indicators showing bullish momentum with LSTM confidence of 72%. RSI oversold, MACD bullish crossover.",
            timestamp: new Date().toISOString(),
            status: "pending",
            ml_predictions: {
              technical: 0.75,
              ensemble: 0.68,
              lstm: 0.74,
              momentum: 0.69,
            },
          },
          {
            id: 2,
            symbol: "ETH",
            current_price: 2398.6,
            target_price: 2450.0,
            stop_loss: 2300.0,
            action: "HOLD",
            confidence: 0.65,
            risk_score: 0.18,
            reasoning:
              "Mixed signals from technical analysis. Waiting for clearer direction. Volume declining.",
            timestamp: new Date().toISOString(),
            status: "pending",
            ml_predictions: {
              technical: 0.58,
              ensemble: 0.62,
              lstm: 0.68,
              momentum: 0.55,
            },
          },
          {
            id: 3,
            symbol: "SOL",
            current_price: 102.3,
            target_price: 108.5,
            stop_loss: 98.0,
            action: "BUY",
            confidence: 0.78,
            risk_score: 0.3,
            reasoning:
              "Strong momentum indicators and positive sentiment analysis. Breaking resistance levels.",
            timestamp: new Date().toISOString(),
            status: "pending",
            ml_predictions: {
              technical: 0.82,
              ensemble: 0.75,
              lstm: 0.72,
              momentum: 0.85,
            },
          },
        ];
        setTrades(mockTrades);
        setError(
          "Demo mode - Connect your FastAPI server to see live proposals"
        );
        return;
      }

      setApiStatus("connected");

      // Fetch proposals from dashboard endpoint
      const response = await fetch(`${apiUrl}/api/dashboard`, {
        headers: {
          "ngrok-skip-browser-warning": "true",
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.pending_proposals && Array.isArray(data.pending_proposals)) {
        setTrades(data.pending_proposals);
      } else {
        setTrades([]);
      }

      // Also fetch portfolio metrics
      try {
        const portfolioResponse = await fetch(
          `${apiUrl}/ai/portfolio_metrics`,
          {
            headers: {
              "ngrok-skip-browser-warning": "true",
              "Content-Type": "application/json",
            },
          }
        );
        if (portfolioResponse.ok) {
          const portfolioData = await portfolioResponse.json();
          setPortfolioMetrics(portfolioData);
        }
      } catch (portErr) {
        console.log("Could not fetch portfolio metrics:", portErr);
      }
    } catch (error) {
      console.error("Error fetching trades:", error);
      setError(error.message);
      setApiStatus("error");
      setTrades([]);
    } finally {
      setLoading(false);
    }
  };

  const processTradeDecision = async (tradeId, decision) => {
    setProcessingTrade(tradeId);

    try {
      const apiUrl = await findWorkingApi();
      if (!apiUrl) {
        alert("API not available - decision recorded locally only");
        setTrades((prev) => prev.filter((t) => t.id !== tradeId));
        return;
      }

      const response = await fetch(`${apiUrl}/process_proposal`, {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "true",
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          proposal_id: tradeId.toString(),
          action: decision,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log("Trade processed:", result);

        // Remove processed trade from list
        setTrades((prev) => prev.filter((t) => t.id !== tradeId));

        // Show success message
        const trade = trades.find((t) => t.id === tradeId);
        alert(
          `${decision.toUpperCase()}: ${
            trade?.symbol || "Trade"
          } processed successfully`
        );

        // Refresh data
        setTimeout(fetchTrades, 1000);
      } else {
        throw new Error("Failed to process trade decision");
      }
    } catch (error) {
      console.error("Error processing trade:", error);
      alert(`Error processing trade: ${error.message}`);
    } finally {
      setProcessingTrade(null);
    }
  };

  useEffect(() => {
    fetchTrades();
    const interval = setInterval(fetchTrades, 15000); // Refresh every 15 seconds
    return () => clearInterval(interval);
  }, []);

  const getActionColor = (action) => {
    switch (action?.toUpperCase()) {
      case "BUY":
        return "text-green-400 bg-green-900/30 border-green-500";
      case "SELL":
        return "text-red-400 bg-red-900/30 border-red-500";
      case "HOLD":
        return "text-yellow-400 bg-yellow-900/30 border-yellow-500";
      default:
        return "text-gray-400 bg-gray-900/30 border-gray-500";
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) return "text-green-400";
    if (confidence >= 0.5) return "text-yellow-400";
    return "text-red-400";
  };

  const getRiskColor = (risk) => {
    if (risk <= 0.2) return "text-green-400";
    if (risk <= 0.4) return "text-yellow-400";
    return "text-red-400";
  };

  const getStatusIcon = () => {
    switch (apiStatus) {
      case "connected":
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case "error":
        return <AlertTriangle className="w-4 h-4 text-red-400" />;
      case "disconnected":
        return <X className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-yellow-400" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen  flex items-center justify-center">
        <div className="text-white text-xl">Loading trade proposals...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen  text-white p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">
              AI Trading Proposals
            </h1>
            <div className="flex items-center space-x-4 text-sm text-gray-400">
              {getStatusIcon()}
              <span>API Status: {apiStatus.toUpperCase()}</span>
              <span>•</span>
              <span>
                {trades.length} pending proposal{trades.length !== 1 ? "s" : ""}
              </span>
              <span>•</span>
              <span>Last updated: {new Date().toLocaleTimeString()}</span>
            </div>
          </div>

          {portfolioMetrics && (
            <div className="text-right">
              <div className="text-sm text-gray-400">Portfolio Value</div>
              <div className="text-2xl font-bold text-green-400">
                $99,895.6
              </div>
              <div className="text-sm text-gray-400">
                1 position
              </div>
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-yellow-900/30 border border-yellow-600 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
              <span className="text-yellow-300">{error}</span>
            </div>
          </div>
        )}

        {/* Trade Proposals Grid */}
        {trades.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {trades.map((trade) => (
              <div
                key={trade.id}
                className="bg-gray-950 border border-[#ff6f61] rounded-xl p-6 hover:border-gray-600 transition-all duration-200 shadow-lg"
              >
                {/* Header */}
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h2 className="text-xl font-bold text-white">
                      {trade.symbol}
                    </h2>
                    <div className="text-sm text-gray-400">
                      {new Date(trade.timestamp || Date.now()).toLocaleString()}
                    </div>
                  </div>
                  <div
                    className={`px-3 py-1 rounded-full text-sm font-medium border ${getActionColor(
                      trade.action
                    )}`}
                  >
                    {trade.action}
                  </div>
                </div>

                {/* Price Information */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-gray-700 p-3 rounded-lg">
                    <div className="text-xs text-gray-400 mb-1">
                      Current Price
                    </div>
                    <div className="text-lg font-semibold text-white">
                      ${trade.current_price?.toFixed(2) || "N/A"}
                    </div>
                  </div>
                  <div className="bg-gray-700 p-3 rounded-lg">
                    <div className="text-xs text-gray-400 mb-1">
                      Target Price
                    </div>
                    <div className="text-lg font-semibold text-green-400">
                      ${trade.target_price?.toFixed(2) || "N/A"}
                    </div>
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Confidence</div>
                    <div
                      className={`text-sm font-bold ${getConfidenceColor(
                        trade.confidence || 0
                      )}`}
                    >
                      {((trade.confidence || 0) * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Risk Score</div>
                    <div
                      className={`text-sm font-bold ${getRiskColor(
                        trade.risk_score || 0
                      )}`}
                    >
                      {((trade.risk_score || 0) * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                {/* ML Predictions */}
                {trade.ml_predictions && (
                  <div className="mb-4">
                    <div className="text-xs text-gray-400 mb-2">
                      ML Model Scores
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {Object.entries(trade.ml_predictions).map(
                        ([model, score]) => (
                          <div
                            key={model}
                            className="flex justify-between bg-gray-700 px-2 py-1 rounded"
                          >
                            <span className="text-gray-300">{model}:</span>
                            <span className="text-blue-400">
                              {(score * 100).toFixed(0)}%
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                {/* Stop Loss & Risk Info */}
                <div className="mb-4 p-3 bg-gray-700 rounded-lg">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Stop Loss:</span>
                    <span className="text-red-400 font-medium">
                      ${trade.stop_loss?.toFixed(2) || "N/A"}
                    </span>
                  </div>
                </div>

                {/* Reasoning */}
                <div className="mb-4 p-3 bg-gray-700/50 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">Analysis</div>
                  <p className="text-xs text-gray-300 leading-relaxed">
                    {trade.reasoning || "No reasoning provided"}
                  </p>
                </div>

                {/* Action Buttons */}
                <div className="space-y-2">
                  {trade.action === "BUY" && (
                    <button
                      onClick={() => processTradeDecision(trade.id, "approved")}
                      disabled={processingTrade === trade.id}
                      className="w-full py-2 bg-green-600 hover:bg-green-700 disabled:bg-green-800 disabled:opacity-50 text-white font-medium rounded-lg transition-colors flex items-center justify-center space-x-2"
                    >
                      {processingTrade === trade.id ? (
                        <Clock className="w-4 h-4 animate-spin" />
                      ) : (
                        <TrendingUp className="w-4 h-4" />
                      )}
                      <span>Execute Buy Order</span>
                    </button>
                  )}

                  {trade.action === "SELL" && (
                    <button
                      onClick={() => processTradeDecision(trade.id, "approved")}
                      disabled={processingTrade === trade.id}
                      className="w-full py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-800 disabled:opacity-50 text-white font-medium rounded-lg transition-colors flex items-center justify-center space-x-2"
                    >
                      {processingTrade === trade.id ? (
                        <Clock className="w-4 h-4 animate-spin" />
                      ) : (
                        <TrendingDown className="w-4 h-4" />
                      )}
                      <span>Execute Sell Order</span>
                    </button>
                  )}

                  {trade.action === "HOLD" && (
                    <button
                      onClick={() => processTradeDecision(trade.id, "approved")}
                      disabled={processingTrade === trade.id}
                      className="w-full py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-yellow-800 disabled:opacity-50 text-white font-medium rounded-lg transition-colors flex items-center justify-center space-x-2"
                    >
                      {processingTrade === trade.id ? (
                        <Clock className="w-4 h-4 animate-spin" />
                      ) : (
                        <Activity className="w-4 h-4" />
                      )}
                      <span>Acknowledge Hold</span>
                    </button>
                  )}

                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => processTradeDecision(trade.id, "rejected")}
                      disabled={processingTrade === trade.id}
                      className="py-2 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-800 disabled:opacity-50 text-white font-medium rounded-lg transition-colors flex items-center justify-center space-x-2"
                    >
                      <X className="w-4 h-4" />
                      <span>Reject</span>
                    </button>

                    <button
                      onClick={() => {
                        const details = `
Symbol: ${trade.symbol}
Action: ${trade.action}
Price: $${trade.current_price?.toFixed(2)}
Target: $${trade.target_price?.toFixed(2)}
Confidence: ${((trade.confidence || 0) * 100).toFixed(0)}%
Risk: ${((trade.risk_score || 0) * 100).toFixed(0)}%
Reasoning: ${trade.reasoning || "N/A"}
                        `.trim();
                        navigator.clipboard.writeText(details);
                        alert("Trade details copied to clipboard");
                      }}
                      className="py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
                    >
                      Copy Details
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <Activity className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <div className="text-xl text-gray-400 mb-2">
              No pending trade proposals
            </div>
            <div className="text-gray-500">
              {apiStatus === "connected"
                ? "Your AI trading system is running but has no current recommendations."
                : "Connect your FastAPI server to see live trading proposals."}
            </div>
            <button
              onClick={fetchTrades}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              Refresh Proposals
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
