# data_fetcher.py - Enhanced Data Provider (integrates your existing code)
import asyncio
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import config
from models import db_manager
import asyncio
import nest_asyncio

# Allow nested event loops (fixes the conflict)
nest_asyncio.apply()
# Import your existing crypto data provider
from crypto_data_provider import CryptoDataProvider

logger = logging.getLogger(__name__)

class EnhancedDataFetcher:
    """
    Enhanced data fetcher that combines your existing crypto provider
    with stock data fetching for the AI agent
    """
    
    def __init__(self):
        self.crypto_provider = CryptoDataProvider()  # Your existing crypto provider
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    async def get_crypto_data(self, symbol: str, period: int = 60) -> Optional[pd.DataFrame]:
        """Fetch crypto data using your existing crypto provider"""
        try:
            # Use your existing crypto provider's historical data method
            df = await self.crypto_provider.get_historical_data(symbol, interval='1d', limit=period)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return None
    
    async def get_realtime_price(self, symbol: str, asset_type: str = 'crypto') -> Optional[Dict]:
        """Get real-time crypto price data"""
        try:
            # Always use crypto provider since we're crypto-only now
            return await self.crypto_provider.get_realtime_data(symbol)
                
        except Exception as e:
            logger.error(f"Error getting realtime price for {symbol}: {e}")
            return None
    
    async def get_multiple_realtime_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get real-time prices for multiple crypto symbols
        """
        tasks = []
        for symbol in symbols:
            tasks.append(self.get_realtime_price(symbol, 'crypto'))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        price_data = {}
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception) and result:
                price_data[symbol] = result
        
        return price_data
    
    def get_crypto_candidates(self) -> List[str]:
        """Get expanded list of crypto symbols for analysis"""
        return [
            'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'AVAX', 
            'MATIC', 'LINK', 'UNI', 'LTC', 'BCH', 'ALGO', 'ATOM',
            'FTM', 'NEAR', 'ICP', 'APT', 'ARB', 'OP', 'DOGE', 'SHIB'
        ]
    
    async def close(self):
        """Close any open connections"""
        if hasattr(self.crypto_provider, 'close'):
            await self.crypto_provider.close()

# Global instance
data_fetcher = EnhancedDataFetcher()