# utils.py - Utility Functions
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import config

def setup_logging():
    """Configure logging for the application"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/trading_agent_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific log levels
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

def validate_environment():
    """Validate required environment variables and configuration"""
    required_configs = [
        'EMAIL_ADDRESS',
        'EMAIL_PASSWORD', 
        'USER_EMAIL',
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY'
    ]
    
    missing_configs = [
        config_name for config_name in required_configs
        if not getattr(config, config_name, None)
    ]
    
    if missing_configs:
        print(f"❌ Missing required configuration: {', '.join(missing_configs)}")
        print("Please check your .env file or environment variables")
        return False
    
    if not os.path.exists(config.DB_FILE):
        print(f"⚠️ Database file {config.DB_FILE} not found. Will be created on first run.")
    
    print("✅ Environment validation passed")
    return True


def calculate_position_size(price: float, portfolio_value: float = None, risk_per_trade: float = 0.02) -> int:
    """
    Calculate position size based on price and risk management
    This is a more sophisticated version of your helpers.calculate_quantity function
    """
    if price <= 0:
        return 0
    
    # Use allocation from config if no portfolio value provided
    if portfolio_value is None:
        allocation = config.TRADE_ALLOCATION_USD
    else:
        allocation = portfolio_value * risk_per_trade
    
    # Calculate maximum shares/units
    max_units = allocation / price
    
    # For stocks, return whole shares
    if price > 10:  # Assume stocks if price > $10
        return max(1, int(max_units))
    else:  # Crypto or low-price assets
        return max(0.01, round(max_units, 4))

def format_currency(amount: float) -> str:
    """Format currency amounts consistently"""
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage values consistently"""
    return f"{value:.2f}%"

def is_market_hours() -> bool:
    """Check if it's within market trading hours (9:30 AM - 4:00 PM ET)"""
    # This is a simplified check - you might want to use a library like pandas_market_calendars
    # for proper market holiday handling
    now = datetime.now()
    weekday = now.weekday()  # 0 = Monday, 6 = Sunday
    
    # Weekend check
    if weekday > 4:  # Saturday or Sunday
        return False
    
    # Simple hour check (this doesn't account for timezone properly)
    current_hour = now.hour
    return 9 <= current_hour <= 16

def sanitize_symbol(symbol: str) -> str:
    """Sanitize and normalize trading symbols"""
    if not symbol:
        return ""
    
    # Remove common suffixes and normalize
    symbol = symbol.upper().strip()
    symbol = symbol.replace('.', '-')  # Some APIs use dashes instead of dots
    
    return symbol

def get_risk_level(risk_score: float) -> str:
    """Convert risk score to human-readable risk level"""
    if risk_score <= 0.3:
        return "Low"
    elif risk_score <= 0.6:
        return "Medium"
    else:
        return "High"

def get_confidence_level(confidence: float) -> str:
    """Convert confidence score to human-readable level"""
    if confidence >= 0.8:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.6:
        return "Medium"
    else:
        return "Low"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int with fallback"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def parse_timeframe(timeframe: str) -> Dict[str, int]:
    """Parse timeframe string into components"""
    # Simple parser for timeframes like "1d", "1h", "5m"
    if not timeframe:
        return {"amount": 1, "unit": "d"}
    
    import re
    match = re.match(r'(\d+)([dhm])', timeframe.lower())
    if match:
        amount, unit = match.groups()
        return {"amount": int(amount), "unit": unit}
    
    return {"amount": 1, "unit": "d"}

def load_prompt_template() -> str:
    """Load the agent prompt template from file"""
    prompt_file = "prompt.txt"
    
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            return f.read()
    else:
        # Default prompt if file doesn't exist
        return """
You are an AI trading agent assistant. Your role is to:

1. Analyze market data using technical indicators
2. Generate clear, actionable trading recommendations 
3. Explain your reasoning in simple terms
4. Consider risk management in all recommendations
5. Maintain a professional but approachable tone

Focus on providing value through:
- Clear technical analysis explanations
- Risk-adjusted position sizing recommendations
- Transparent reasoning for all decisions
- Appropriate confidence levels based on signal strength

Always prioritize risk management and education over aggressive trading.
        """.strip()

def create_trade_summary(trades: List[Dict]) -> Dict[str, Any]:
    """Create a summary of trading activity"""
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0
        }
    
    total_trades = len(trades)
    executed_trades = [t for t in trades if t.get('status') == 'executed']
    
    # This is simplified - in reality you'd need current prices to calculate PnL
    winning_trades = sum(1 for t in executed_trades if t.get('pnl', 0) > 0)
    losing_trades = sum(1 for t in executed_trades if t.get('pnl', 0) < 0)
    total_pnl = sum(t.get('pnl', 0) for t in executed_trades)
    
    win_rate = (winning_trades / len(executed_trades)) * 100 if executed_trades else 0
    
    return {
        "total_trades": total_trades,
        "executed_trades": len(executed_trades),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl
    }

def validate_proposal_data(proposal: Dict) -> bool:
    """Validate that a proposal contains all required fields"""
    required_fields = [
        'symbol', 'asset_type', 'action', 'confidence',
        'current_price', 'reasoning'
    ]
    
    return all(field in proposal and proposal[field] is not None for field in required_fields)

def generate_trade_id() -> str:
    """Generate a unique trade ID"""
    import uuid
    return f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

# Environment validation on import
if __name__ == "__main__":
    setup_logging()
    validate_environment()