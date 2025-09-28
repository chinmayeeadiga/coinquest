# config.py - FIXED Configuration for MORE Proposals
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FIXED: VERY LOW thresholds to generate MORE proposals
CONFIDENCE_THRESHOLD = 0.15  # LOWERED from 0.30 to 0.15
MIN_CONFIDENCE_THRESHOLD = 0.10  # LOWERED from 0.25 to 0.10
STREAMLIT_PORT = 8501

# FIXED: More aggressive trading parameters  
TRADE_ALLOCATION_USD = float(os.getenv('TRADE_ALLOCATION_USD', '500'))  
MAX_CONCURRENT_POSITIONS = 20  # Increased from 15
MAX_SIGNALS_PER_CYCLE = 15  # Increased from 10
TARGET_PROPOSALS_PER_CYCLE = 8  # NEW: Target at least 8 proposals

# Database
DB_FILE = 'new_db.db'

# Alpaca Paper Trading API
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '').strip()
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '').strip()
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
BASE_URL = 'https://paper-api.alpaca.markets'

# Email Configuration
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', '').strip()
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '').strip()
USER_EMAIL = os.getenv('USER_EMAIL', '').strip()

# FIXED: EXPANDED crypto list for MORE opportunities
ALL_CRYPTO_SYMBOLS = [
    # Tier 1 - Always analyze (high volume, established)
    'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'AVAX',
    'MATIC', 'LINK', 'UNI', 'LTC', 'ALGO', 'ATOM', 'NEAR',
    
    # Tier 2 - Frequently analyze (good volume, popular)
    'FTM', 'ICP', 'APT', 'ARB', 'OP', 'DOGE', 'SHIB', 'SAND',
    'MANA', 'AXS', 'GALA', 'ENJ', 'CHZ', 'BAT', 'ZEC', 'DASH',
    
    # Tier 3 - Regular analysis (medium volume, opportunities)
    'PEPE', 'FLOKI', 'CRV', 'AAVE', 'SUSHI', 'COMP', 'YFI',
    'MKR', 'SNX', '1INCH', 'BAL', 'CAKE', 'RUNE', 'KAVA',
    
    # Tier 4 - Occasional analysis (smaller caps, higher volatility)
    'XLM', 'VET', 'HBAR', 'ETC', 'THETA', 'XTZ', 'FIL', 'EOS',
    'TRX', 'BSV', 'NEO', 'IOTA', 'DASH', 'ZEC', 'XMR', 'DCR'
]

# Alpaca supported (limited but real execution possible)
ALPACA_SUPPORTED_CRYPTOS = ['BTC', 'ETH', 'DOGE', 'LTC', 'BCH', 'AAVE', 'UNI', 'LINK']

# FIXED: VERY AGGRESSIVE analysis parameters for MORE signals
ANALYSIS_SETTINGS = {
    'min_data_points': 20,  # REDUCED from 30
    'confidence_threshold': 0.10,  # VERY LOW threshold
    'volatility_threshold': 0.20,  # HIGHER volatility allowed  
    'position_size_range': (0.01, 0.10),  # 1-10% positions
    'force_signals': True,  # Force signal generation
    'mock_data_enhanced': True,  # Use enhanced mock data
    'analysis_frequency': 180,  # 3 minutes between analysis
    'multi_pass_analysis': True,  # Enable multi-pass analysis
    'demo_signal_boost': True,  # Enable demo signal boosting
    'lower_threshold_passes': True,  # Enable lower threshold passes
}

# FIXED: Risk management - VERY RELAXED for more opportunities
MAX_PORTFOLIO_RISK = 0.35  # Increased from 0.25
MAX_POSITION_SIZE = 0.10   # Increased from 0.08
STOP_LOSS_PERCENTAGE = 0.04  # Increased from 0.03
DAILY_LOSS_LIMIT = 0.15    # Increased from 0.10
MAX_DAILY_TRADES = 25      # Increased from 20

# Web App Settings
WEB_HOST = '0.0.0.0'
WEB_PORT = 8000

# FIXED: Enhanced proposal generation settings
PROPOSAL_GENERATION = {
    'target_proposals_per_cycle': 8,  # Target at least 8 proposals
    'max_proposals_per_cycle': 15,    # Maximum 15 proposals
    'confidence_passes': [
        {'name': 'high_confidence', 'threshold': 0.20, 'max_signals': 5},
        {'name': 'medium_confidence', 'threshold': 0.12, 'max_signals': 5}, 
        {'name': 'opportunity_signals', 'threshold': 0.08, 'max_signals': 8},
        {'name': 'demo_enhanced', 'threshold': 0.05, 'max_signals': 10}
    ],
    'deduplication_enabled': True,
    'demo_signal_fallback': True,
    'volume_boost_enabled': True,
    'momentum_boost_enabled': True
}

# Auto-execution settings (KEEP FALSE FOR SAFETY)
AUTO_EXECUTE_TRADES = False
AUTO_EXECUTE_THRESHOLD = 0.85

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = 'coinquest.log'

# FIXED: Email base URL
EMAIL_BASE_URL = "http://localhost:8000"

def validate_config():
    """Validate configuration with enhanced proposal generation focus"""
    issues = []
    warnings = []
    
    # Check email setup
    if not EMAIL_ADDRESS:
        warnings.append("EMAIL_ADDRESS missing - email notifications disabled")
    if not EMAIL_PASSWORD:
        warnings.append("EMAIL_PASSWORD missing - email notifications disabled") 
    if not USER_EMAIL:
        warnings.append("USER_EMAIL missing - no one will receive notifications")
    
    # Check trading setup
    if not ALPACA_API_KEY:
        warnings.append("ALPACA_API_KEY missing - using simulation only")
    if not ALPACA_SECRET_KEY:
        warnings.append("ALPACA_SECRET_KEY missing - using simulation only")
    
    # Check thresholds - should be VERY low for more proposals
    if CONFIDENCE_THRESHOLD > 0.2:
        issues.append(f"CONFIDENCE_THRESHOLD too high ({CONFIDENCE_THRESHOLD}) - should be <= 0.2 for MORE proposals")
    
    if MIN_CONFIDENCE_THRESHOLD > 0.15:
        issues.append(f"MIN_CONFIDENCE_THRESHOLD too high ({MIN_CONFIDENCE_THRESHOLD}) - should be <= 0.15 for MORE proposals")
    
    if issues:
        print("❌ CRITICAL Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✅ Configuration validated for ENHANCED proposal generation")
    print(f"📊 Analysis settings: {ANALYSIS_SETTINGS}")
    print(f"📈 Confidence threshold: {CONFIDENCE_THRESHOLD} (VERY LOW for more signals)")
    print(f"🎯 Target proposals per cycle: {TARGET_PROPOSALS_PER_CYCLE}")
    print(f"📈 Max proposals per cycle: {MAX_SIGNALS_PER_CYCLE}")
    print(f"💰 Max position size: {MAX_POSITION_SIZE:.1%}")
    print(f"🎯 Symbols monitored: {len(ALL_CRYPTO_SYMBOLS)}")
    print(f"🔄 Multi-pass analysis: {ANALYSIS_SETTINGS['multi_pass_analysis']}")
    print(f"🚀 Demo signal boost: {ANALYSIS_SETTINGS['demo_signal_boost']}")
    
    return True

if __name__ == "__main__":
    validate_config()