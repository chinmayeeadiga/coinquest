# AI Trading Agent - Agentic Trading System

A sophisticated AI-powered trading system that combines your existing technical analysis with autonomous market scanning, proposal generation, and human-in-the-loop approval workflows.

## 🚀 Features

### Core Agentic Capabilities
- **Autonomous Market Scanning**: Scans stocks and crypto using your existing database
- **Multi-Factor Analysis**: Combines your technical analysis with AI-enhanced scoring
- **Smart Proposal Generation**: Creates top 5 trade recommendations with reasoning
- **Dual Approval System**: Web dashboard + email approval links
- **Automated Execution**: Executes approved trades via Alpaca Paper Trading

### Technical Analysis Integration
- **Preserves Your Code**: Integrates your existing `technical_analysis.py` perfectly
- **Enhanced Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Volume Analysis
- **Risk Management**: Automatic stop-loss and take-profit calculations
- **Confidence Scoring**: AI-powered confidence levels for each recommendation

### User Interface
- **Modern Web Dashboard**: Real-time proposal approval interface
- **Email Notifications**: Rich HTML emails with approve/reject buttons
     
- **Manual Analysis**: On-demand analysis of any stock or crypto
- **Portfolio Tracking**: View positions, trades, and performance

## 📁 Project Structure
**frontend**
```
coinquest-frontend/
├── public/
│   ├── favicon.ico
│   └── vite.svg
│
├── src/
│   ├── assets/
│   │   ├── images/
│   │   │   ├── logo.png
│   │   │   └── react.svg
│   │
│   ├── components/
│   │   ├── AlgorithmStatusCard.jsx
│   │   ├── MarketDataCard.jsx
│   │   ├── PortfolioCard.jsx
│   │   ├── SystemMonitorCard.jsx
│   │   ├── algo.css
│   │   |── market.css
│   │   │── port.css
│   │   │
│   │   ├── lib/
│   │   │   └── utils.js
│   │   │
│   │   └── ui/
│   │       ├── AuroraBackground.jsx
│   │       ├── Cyberpunk.jsx
│   │       ├── GlareCard.jsx
│   │       ├── HoverCard.jsx
│   │       ├── Typewriter.jsx
│   │       └── cyberpunk.css
│   │
│   ├── layouts/
│   │   └── AppLayout.jsx
│   │
│   ├── pages/
│   │   ├── Home.jsx
│   │   ├── Crypto.jsx
│   │   ├── Trade.jsx
│   │   └── home.css
│   │
│   ├── App.jsx
│   ├── App.css
│   ├── index.css
│   └── main.jsx
│
├── .gitignore
├── package.json
├── vite.config.js
└── README.md

```

__backend__
```
ai-trading-agent/
├── backend/ # AI trading agent backend
│ ├── complete_realtime_main.py 
│ ├── enhanced_crypto_agent.py # Main AI agent orchestrator
│ ├── data_fetcher.py
│ ├── enhanced_crypto_analyzer.py
│ ├── enhanced_crypto_trader.py
│ ├── notifier.py
│ ├── crypto_data_provider.py
│ ├── models.py
│ ├── tasks.py
│ ├── utils.py
│ ├── config.py
│ ├── database.py
│ ├── requirements.txt

```

## 🛠️ Installation & Setup

#frontend
### 1. Navigate to frontend folder
```bash
cd frontend
```

### 2. Install dependencies
```bash
npm install
```

### 3. Start development server
```bash
npm run dev
```



#backend
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Setup Script
```bash
python setup.py
```

### 3. Configure Environment
Copy `.env.template` to `.env` and configure:
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

### 4. Start the Application
```bash
uvicorn new_main:app --reload
```

### 5. Access the Dashboard
Open http://localhost:8000 in your browser

## 🔧 Configuration

### Alpaca Paper Trading Setup
1. Create account at [alpaca.markets](https://alpaca.markets)
2. Generate Paper Trading API keys
3. Add keys to your `.env` file

### Email Notifications Setup
1. Enable 2-factor authentication on Gmail
2. Generate an App Password
3. Use the App Password (not your regular password) in `.env`

### Database Integration
The system automatically integrates with your existing database structure:
- Preserves your `stock`, `stock_price`, `stock_strategy` tables
- Adds new tables for agent proposals and trade execution
- Uses your existing technical indicators (SMA, RSI, etc.)

## 🤖 How the AI Agent Works

### 1. Market Scanning
- Queries your existing stock database for candidates
- Filters using your SMA and RSI indicators
- Adds popular crypto symbols for analysis

### 2. Technical Analysis
- Uses your existing `TechnicalAnalyzer` class
- Enhances with simulated sentiment and ML scoring
- Calculates confidence levels and risk scores

### 3. Proposal Generation
- Ranks all analyzed assets by confidence
- Selects top 5 actionable recommendations
- Balances buy/sell recommendations for diversity

### 4. User Approval
- Saves proposals to database with 2-hour expiry
- Sends HTML email with approve/reject links
- Provides web dashboard for bulk actions

### 5. Trade Execution
- Executes approved trades on Alpaca Paper Trading
- Simulates crypto trades with portfolio tracking
- Sends confirmation emails with results

## 🎯 Usage Examples

### Automatic Analysis Cycle
The agent runs automatically:
- Every 10 minutes (demo mode)
- Pre-market at 9:00 AM ET
- Mid-day at 12:00 PM ET
- Monday through Friday

### Manual Analysis
Use the web dashboard to analyze any symbol:
```python
# Via web interface or API
POST /analyze_manual
{
    "symbol": "AAPL",
    "asset_type": "stock"
}
```

### Email Approval Workflow
1. Agent finds 5 high-confidence opportunities
2. Email sent with proposal details and reasoning
3. Click "Approve & Execute" or "Reject" in email
4. Trade executes automatically on approval
5. Confirmation email sent with results

## 🔐 Security & Risk Management

### Paper Trading Only
- All stock trades via Alpaca Paper Trading
- Crypto trades are simulated locally
- No real money at risk

### Risk Controls
- Maximum 2% allocation per trade
- Automatic stop-losses on all positions
- Position limits and concentration limits
- Human approval required for all trades

### Data Privacy
- All data stored locally in SQLite
- Email notifications only to configured address
- No external AI APIs used for analysis

## 📊 Monitoring & Maintenance

### Web Dashboard Features
- Real-time portfolio status
- Pending approval notifications
- Manual symbol analysis
- Trade history and performance

### Background Tasks
- Automatic proposal cleanup
- System health monitoring
- Database maintenance
- Log rotation

### Logging
All activities logged to:
- `logs/trading_agent_YYYYMMDD.log`
- Console output during development
- Error tracking and debugging info

## 🔄 Integration with Your Existing Code

### Preserved Components
- Your `technical_analysis.py` - Used as-is for all technical analysis
- Your `crypto_data_provider.py` - Integrated for crypto data
- Your database schema - Extended, not replaced
- Your trading strategies - Available for future integration

### Enhanced Components
- **Data Fetching**: Combines your crypto provider with stock data
- **Analysis Engine**: Uses your technical analysis + AI scoring
- **Execution System**: Adds paper trading and risk management
- **User Interface**: Modern web dashboard and email approvals

## 🚀 Future Enhancements

### Planned Features
- Integration of your existing strategies (opening_range_breakout, bollinger_bands)
- Advanced portfolio optimization
- Real-time market data streaming
- Mobile app for approvals
- Advanced backtesting integration

### Customization Options
- Adjustable confidence thresholds
- Custom notification schedules
- Additional technical indicators
- Portfolio risk parameters

## 📞 Support & Development

### Troubleshooting
1. Check logs in `logs/` directory
2. Verify `.env` configuration
3. Ensure all dependencies installed
4. Check Alpaca API key permissions

### Development Mode
```bash
# Start with auto-reload
uvicorn new_main:app --reload --log-level debug

# Trigger manual analysis
curl -X POST http://localhost:8000/api/trigger_analysis
```

### Testing
```bash
# Run basic health check
curl http://localhost:8000/health

# Test manual analysis
curl -X POST http://localhost:8000/analyze_manual \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "asset_type": "stock"}'
```

---

**Note**: This system is designed for educational and research purposes. Paper trading only - no real money at risk. Always do your own research before making investment decisions.
