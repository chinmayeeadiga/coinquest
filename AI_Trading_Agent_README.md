# 🧠 AI Trading Agent - Agentic Trading System  

A full-stack **AI-powered autonomous trading system** with **real-time market scanning**, **AI-enhanced technical analysis**, and a **human-in-the-loop approval workflow**.  

👉 **Live Demo:**  
- **Frontend Dashboard:** [coinquest-indol.vercel.app](https://coinquest-indol.vercel.app/)  
- **Backend API:** [coin-pilot-3.onrender.com](https://coin-pilot-3.onrender.com)  

---

## 🚀 Key Features  

### 🤖 AI-Powered Agent  
- **Autonomous Market Scanning**: Stocks & crypto from multiple data sources  
- **Multi-Factor Analysis**: Technicals + ML scoring + volatility models  
- **Smart Proposal Generation**: Top 5 ranked trade opportunities with reasoning  
- **Confidence Scoring**: Combines TA + ML + sentiment + volatility forecasts  
- **Dual Approval Workflow**: Web dashboard + email-based approvals  

### 📊 Technical Analysis  
- **Indicators Used:** SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Volume Analysis  
- **Fallback Support:** Custom technical analysis when `pandas-ta` unavailable  
- **Risk Management:** Auto stop-loss, take-profit, portfolio concentration rules  

### 💹 Trading & Execution  
- **Stock Trading:** [Alpaca Markets Paper Trading API](https://alpaca.markets)  
- **Crypto Trading:** [CCXT](https://github.com/ccxt/ccxt) (Binance, Coinbase sandbox)  
- **Order Types:** Market, Limit, Bracket (stop-loss + take-profit)  
- **Portfolio Tracking:** Live positions, trade history, and performance analytics  

### 🖥️ User Experience  
- **Frontend (React + Vite + Tailwind + shadcn/ui)**  
  - Real-time trade proposals dashboard  
  - Portfolio view with performance metrics  
  - Approval/rejection UI with modern cyberpunk styling  
- **Email Notifications:** Approve or reject trades directly from your inbox  

---

## ⚙️ Tech Stack  

### 🖥️ Frontend (Vercel Deployment)  
- **React + Vite** – Modern frontend framework & bundler  
- **TailwindCSS + shadcn/ui** – Fast, elegant UI design  
- **Lucide Icons** – Beautiful icons  
- **Framer Motion** – Animations & transitions  
- **Deployed on Vercel** – [coinquest-indol.vercel.app](https://coinquest-indol.vercel.app/)  

### 🧩 Backend (Render Deployment)  
- **FastAPI + Uvicorn** – High-performance async API server  
- **SQLite** – Local database integration (preserving your existing schema)  
- **Asyncio** – Concurrent background tasks (market scanning, cleanup, monitoring)  
- **Logging** – System activity, error tracking, and debugging logs  
- **Deployed on Render** – [coin-pilot-3.onrender.com](https://coin-pilot-3.onrender.com)  

### 📚 AI & Data Science Libraries  
- **pandas, numpy** – Data handling and manipulation  
- **scikit-learn** – RandomForest, VotingClassifier, scaling, train/test splits  
- **xgboost** – Gradient boosting ML models for signal classification  
- **tensorflow / keras** – LSTM for deep sequence learning (optional)  
- **arch** – GARCH volatility models (optional)  
- **scipy.stats** – Statistical testing & probability analysis  

### 📈 Technical Analysis Tools  
- **pandas-ta** – Comprehensive TA indicators  
- **Custom TA Engine** – Fallback for SMA, EMA, RSI, MACD, Bollinger, ATR  

### 💹 Trading & Market APIs  
- **Alpaca-trade-api** – Stock & ETF trading (paper only)  
- **alpaca-py (Trading client)** – Bracket order support  
- **CCXT** – Unified crypto trading API (Binance, Coinbase sandbox)  
- **Custom real-time data provider** – For crypto price feeds  

### 📬 Notifications & Workflow  
- **smtplib + email.mime** – Rich HTML emails with secure TLS  
- **EmailNotifier** – Sends approval/rejection trade links  
- **Web Dashboard** – Human-in-the-loop approval system  

---

## 🏗️ System Architecture  

![Architecture Diagram](ai_trading_agent_architecture.png)  

---

## 🔧 Deployment & Configuration  

### Environment Variables  
```env
# Email Configuration
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
USER_EMAIL=your_email@gmail.com

# Alpaca Paper Trading
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret

# Trading Settings
TRADE_ALLOCATION_USD=1000
```

### Run Locally  
```bash
# Frontend
cd frontend
npm install
npm run dev

# Backend
cd backend
pip install -r requirements.txt
uvicorn new_main:app --reload
```

---

## 🔐 Risk Management  

- Max **2% allocation per trade**  
- Automatic **stop-loss** & **take-profit**  
- **Portfolio concentration rules** (no overexposure)  
- **Human approval required** for all trades  
- **Paper trading only** (safe for research & hackathon demo)  

---

## 📊 Monitoring & Maintenance  

- **Web Dashboard** – Real-time portfolio and trade history  
- **Background Tasks** – Automatic cleanup, proposal expiry, log rotation  
- **Logs** – `logs/trading_agent_YYYYMMDD.log`  

---

## 🚀 Future Enhancements  

- ✅ Integration of more advanced strategies (Opening Range Breakout, Bollinger Bands)  
- ✅ Real-time market streaming & WebSocket-based updates  
- ✅ Advanced portfolio optimization (Sharpe ratio, Kelly criterion)  
- ✅ Mobile-first approval app  
- ✅ Backtesting & simulation environment  

---

## 🏆 Hackathon Submission Highlights  

- **AI-driven trade recommendation engine** (ML + TA + volatility modeling)  
- **Human-in-the-loop design** – balances autonomy with safety  
- **Full-stack system** – React frontend + FastAPI backend + email workflows  
- **Live deployment** – Vercel (frontend) + Render (backend)  
- **Risk-aware execution** – Stop-loss, take-profit, allocation rules  
- **Secure notifications** – Gmail App Passwords + TLS  

---

⚡ **Built with care for the hackathon – showcasing AI + Finance + Full-stack deployment** ⚡  
