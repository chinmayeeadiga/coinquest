import React, { useEffect, useState, useRef } from "react";
import { Chart } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  TimeScale,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement,
  BarElement,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import "chartjs-adapter-date-fns";

// Financial charts
import {
  CandlestickController,
  OhlcController,
  CandlestickElement,
  OhlcElement,
} from "chartjs-chart-financial";

// Register everything
ChartJS.register(
  CategoryScale,
  LinearScale,
  TimeScale,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement,
  BarElement,
  zoomPlugin,
  CandlestickController,
  OhlcController,
  CandlestickElement,
  OhlcElement
);

export default function CryptoWatch() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState("1d");
  const [chartType, setChartType] = useState("line");
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [proposals, setProposals] = useState([]);
  const [selectedSymbols, setSelectedSymbols] = useState([]);
  const [proposalError, setProposalError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  const chartRef = useRef(null);

  // Chart options
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { 
        display: true, 
        labels: { color: "#fff" }
      },
      tooltip: {
        titleColor: "#fff",
        bodyColor: "#fff",
        backgroundColor: "rgba(0,0,0,0.8)",
      },
      zoom: {
        zoom: { 
          wheel: { enabled: true }, 
          pinch: { enabled: true }, 
          mode: "x" 
        },
        pan: { enabled: true, mode: "x" },
      },
    },
    scales: {
      x: {
        type: "time",
        time: { unit: "day" },
        ticks: { color: "#fff" },
        grid: { color: "rgba(255,255,255,0.06)" },
      },
      y: {
        ticks: { color: "#fff" },
        grid: { color: "rgba(255,255,255,0.06)" },
      },
    },
  };

  // Fetch Binance data
  useEffect(() => {
    const fetchCrypto = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${timeframe}&limit=100`;
        const res = await fetch(url);
        
        if (!res.ok) {
          throw new Error(`Failed to fetch data: ${res.status}`);
        }
        
        const data = await res.json();

        if (!Array.isArray(data)) {
          console.error("API Error:", data);
          throw new Error("Invalid data format received");
        }

        const labels = data.map((d) => new Date(d[0]));
        const prices = data.map((d) => ({
          t: new Date(d[0]),
          o: parseFloat(d[1]),
          h: parseFloat(d[2]),
          l: parseFloat(d[3]),
          c: parseFloat(d[4]),
        }));

        const formattedData =
          chartType === "candlestick" || chartType === "ohlc"
            ? {
                datasets: [
                  {
                    label: `${symbol} ${timeframe}`,
                    data: prices,
                    borderColor: "#ff6f61",
                    backgroundColor: "rgba(255,111,97,0.8)",
                  },
                ],
              }
            : {
                labels,
                datasets: [
                  {
                    label: `${symbol} ${timeframe}`,
                    data: prices.map((p) => p.c),
                    borderColor: "#ff6f61",
                    backgroundColor: "rgba(255,111,97,0.12)",
                    tension: 0.15,
                    pointRadius: 0,
                    fill: true,
                  },
                ],
              };

        setChartData(formattedData);
        setError(null);
      } catch (err) {
        console.error("Fetch error:", err);
        setError(err.message);
        setChartData(null);
      } finally {
        setLoading(false);
      }
    };

    fetchCrypto();
  }, [symbol, timeframe, chartType]);

  // Check API status and fetch proposals
  useEffect(() => {
    const checkApiAndFetchProposals = async () => {
      const possibleUrls = [
        'https://1917286b732f.ngrok-free.app',
        'http://localhost:8080',
        'http://127.0.0.1:8080',
        'http://localhost:8000',
        'http://127.0.0.1:8000',
      ];

      setApiStatus('checking');
      
      for (const baseUrl of possibleUrls) {
        try {
          // First check if API is running
          const pingRes = await fetch(`${baseUrl}/ping`, {
            method: 'GET',
            headers: {
          'ngrok-skip-browser-warning': 'true',
          'Content-Type': 'application/json',
        },

          });

          if (pingRes.ok) {
            setApiStatus('connected');
            console.log(`Connected to API at ${baseUrl}`);
            
            // Now fetch proposals
            try {
              const proposalRes = await fetch(`${baseUrl}/api/dashboard`, {
                method: 'GET',
                headers: {
          'ngrok-skip-browser-warning': 'true',
          'Content-Type': 'application/json',
        },

              });

              if (proposalRes.ok) {
                const data = await proposalRes.json();
                console.log("Dashboard API response:", data);

                if (data.pending_proposals && Array.isArray(data.pending_proposals)) {
                  setProposals(data.pending_proposals);
                  setProposalError(null);
                } else {
                  // Create mock proposals for demonstration
                  const mockProposals = [
                    {
                      id: 1,
                      symbol: 'BTC',
                      action: 'BUY',
                      current_price: 43890.50,
                      target_price: 45500.00,
                      stop_loss: 42800.00,
                      risk_score: 0.25,
                      confidence: 0.72,
                      reasoning: 'Technical indicators showing bullish momentum with ML confidence of 72%'
                    },
                    {
                      id: 2,
                      symbol: 'ETH',
                      action: 'HOLD',
                      current_price: 2398.60,
                      target_price: 2450.00,
                      stop_loss: 2300.00,
                      risk_score: 0.18,
                      confidence: 0.65,
                      reasoning: 'Mixed signals from technical analysis, waiting for clearer direction'
                    },
                    {
                      id: 3,
                      symbol: 'SOL',
                      action: 'BUY',
                      current_price: 102.30,
                      target_price: 108.50,
                      stop_loss: 98.00,
                      risk_score: 0.30,
                      confidence: 0.78,
                      reasoning: 'Strong momentum indicators and positive sentiment analysis'
                    }
                  ];
                  setProposals(mockProposals);
                  setProposalError('Using demo data - API connected but no live proposals');
                }
              } else {
                throw new Error(`Proposals fetch failed: ${proposalRes.status}`);
              }
            } catch (propErr) {
              console.error("Error fetching proposals:", propErr);
              setProposalError(`Proposals error: ${propErr.message}`);
              setProposals([]);
            }
            
            return; // Success, exit the loop
          }
        } catch (err) {
          console.log(`Failed to connect to ${baseUrl}:`, err.message);
          continue;
        }
      }
      
      // If we get here, none of the URLs worked
      setApiStatus('disconnected');
      setProposalError('Trading API not available - is your FastAPI server running?');
      setProposals([]);
    };

    checkApiAndFetchProposals();
    
    // Set up periodic refresh
    const interval = setInterval(checkApiAndFetchProposals, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Filtering proposals by selectedSymbols
  const filteredProposals =
    selectedSymbols.length > 0
      ? proposals.filter((p) => selectedSymbols.includes(p.symbol))
      : proposals;

  const uniqueSymbols = [...new Set(proposals.map((p) => p.symbol))];

  const getStatusColor = (status) => {
    switch(status) {
      case 'connected': return 'text-green-400';
      case 'checking': return 'text-yellow-400';
      case 'disconnected': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getActionColor = (action) => {
    switch(action?.toUpperCase()) {
      case 'BUY': return 'text-green-400 bg-green-900/30';
      case 'SELL': return 'text-red-400 bg-red-900/30';
      case 'HOLD': return 'text-yellow-400 bg-yellow-900/30';
      default: return 'text-gray-400 bg-gray-900/30';
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6  text-white rounded-xl shadow-2xl">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-bold text-white">Crypto Trading Dashboard</h2>
        <div className="flex items-center space-x-4">
          <div className={`text-sm ${getStatusColor(apiStatus)}`}>
            API: {apiStatus.toUpperCase()}
          </div>
          <div className="text-sm text-gray-400">
            {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">Symbol</label>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="w-full px-3 py-2 bg-gray-950 text-white border border-[#ff6f61] rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="BTCUSDT">BTC/USDT</option>
            <option value="ETHUSDT">ETH/USDT</option>
            <option value="BNBUSDT">BNB/USDT</option>
            <option value="ADAUSDT">ADA/USDT</option>
            <option value="SOLUSDT">SOL/USDT</option>
            <option value="DOTUSDT">DOT/USDT</option>
            <option value="LINKUSDT">LINK/USDT</option>
            <option value="MATICUSDT">MATIC/USDT</option>
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">Timeframe</label>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="w-full px-3 py-2 bg-gray-950 text-white border border-[#ff6f61] rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="1m">1 Minute</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
            <option value="1w">1 Week</option>
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">Chart Type</label>
          <select
            value={chartType}
            onChange={(e) => setChartType(e.target.value)}
            className="w-full px-3 py-2 bg-gray-950 text-white border border-[#ff6f61] rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="line">Line Chart</option>
            <option value="bar">Bar Chart</option>
            <option value="candlestick">Candlestick</option>
            <option value="ohlc">OHLC</option>
          </select>
        </div>
      </div>

      {/* Chart */}
      <div className="bg-gray-950 border border-[#ff6f61] p-4 rounded-lg mb-6" style={{ height: '400px' }}>
        {loading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-blue-400">Loading chart data...</div>
          </div>
        )}
        
        {error && (
          <div className="flex items-center justify-center h-full">
            <div className="text-red-400">Error: {error}</div>
          </div>
        )}
        
        {chartData && !loading && !error && (
          <Chart 
            ref={chartRef} 
            type={chartType} 
            data={chartData} 
            options={options}
          />
        )}
      </div>

      {/* Chart Controls */}
      <div className="flex space-x-2 mb-6">
        <button
          onClick={() => chartRef.current?.resetZoom()}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
        >
          Reset Zoom
        </button>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-md transition-colors"
        >
          Refresh Data
        </button>
      </div>

      {/* Trading Proposals Section */}
      <div className="bg-gray-950 p-6 border border-[#ff6f61] rounded-lg">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-bold text-white">AI Trading Proposals</h3>
          <div className="text-sm text-gray-400">
            {proposals.length} active proposal{proposals.length !== 1 ? 's' : ''}
          </div>
        </div>

        {/* API Status */}
        {(apiStatus !== 'connected' || proposalError) && (
          <div className="mb-4 p-3 bg-yellow-900/30 border border-yellow-600 rounded-md">
            <div className="text-yellow-300 text-sm">
              {proposalError || 'Checking API connection...'}
            </div>
          </div>
        )}

        {/* Symbol Filter */}
        {uniqueSymbols.length > 0 && (
          <div className="mb-6">
            <h4 className="text-lg font-semibold mb-3 text-gray-300">Filter by Symbol</h4>
            
            <div className="flex flex-wrap gap-2 mb-3">
              {uniqueSymbols.map((s) => (
                <label key={s} className="flex items-center space-x-2 bg-gray-700 px-3 py-1 rounded-md cursor-pointer hover:bg-gray-600 transition-colors">
                  <input
                    type="checkbox"
                    value={s}
                    checked={selectedSymbols.includes(s)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedSymbols((prev) => [...prev, s]);
                      } else {
                        setSelectedSymbols((prev) => prev.filter((sym) => sym !== s));
                      }
                    }}
                    className="rounded"
                  />
                  <span className="text-sm font-medium">{s}</span>
                </label>
              ))}
            </div>

            <button
              onClick={() => setSelectedSymbols([])}
              className="px-3 py-1 text-sm bg-gray-600 hover:bg-gray-700 text-white rounded-md transition-colors"
            >
              Clear Filters
            </button>
          </div>
        )}

        {/* Proposals Grid */}
        {filteredProposals.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredProposals.map((proposal) => (
              <div
                key={proposal.id}
                className="bg-gray-700 p-4 rounded-lg border border-gray-600 hover:border-gray-500 transition-all"
              >
                <div className="flex justify-between items-start mb-3">
                  <h4 className="text-lg font-bold text-white">
                    {proposal.symbol}
                  </h4>
                  <span className={`px-2 py-1 rounded text-xs font-bold ${getActionColor(proposal.action)}`}>
                    {proposal.action}
                  </span>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Current:</span>
                    <span className="text-white font-medium">
                      ${proposal.current_price?.toFixed(2) || 'N/A'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">Target:</span>
                    <span className="text-green-400 font-medium">
                      ${proposal.target_price?.toFixed(2) || 'N/A'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">Stop Loss:</span>
                    <span className="text-red-400 font-medium">
                      ${proposal.stop_loss?.toFixed(2) || 'N/A'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">Risk:</span>
                    <span className="text-yellow-400 font-medium">
                      {((proposal.risk_score || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">Confidence:</span>
                    <span className="text-blue-400 font-medium">
                      {((proposal.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                
                {proposal.reasoning && (
                  <div className="mt-3 pt-3 border-t border-gray-600">
                    <p className="text-xs text-gray-400 leading-relaxed">
                      {proposal.reasoning}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-gray-400 text-lg">
              {proposals.length === 0 
                ? 'No trading proposals available' 
                : 'No proposals match your current filter'
              }
            </div>
            {selectedSymbols.length > 0 && (
              <button
                onClick={() => setSelectedSymbols([])}
                className="mt-2 text-blue-400 hover:text-blue-300 underline"
              >
                Clear filters to see all proposals
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}