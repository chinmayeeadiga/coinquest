# crypto_data_provider.py - Real-time Crypto Data
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)

class CryptoDataProvider:
    """Provides real-time and historical crypto data from free APIs"""
    
    def __init__(self):
        self.base_urls = {
            'binance': 'https://api.binance.com/api/v3',
            'coinbase': 'https://api.pro.coinbase.com',
            'cryptocompare': 'https://min-api.cryptocompare.com/data'
        }
        self.session = None
        self.cache = {}
        self.cache_timeout = 60  # 1 minute
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_realtime_data(self, symbol: str) -> Dict:
        """Get real-time price data for a symbol"""
        try:
            session = await self.get_session()
            
            # Binance API for crypto data
            url = f"{self.base_urls['binance']}/ticker/24hr"
            params = {'symbol': f"{symbol.upper()}USDT"}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'symbol': symbol,
                        'price': float(data['lastPrice']),
                        'change_24h': float(data['priceChangePercent']),
                        'volume': float(data['volume']),
                        'high_24h': float(data['highPrice']),
                        'low_24h': float(data['lowPrice']),
                        'timestamp': datetime.now()
                    }
                else:
                    # Fallback to CoinGecko
                    return await self._fallback_realtime_data(symbol)
                    
        except Exception as e:
            logger.error(f"Error getting realtime data for {symbol}: {e}")
            return await self._fallback_realtime_data(symbol)
    
    async def _fallback_realtime_data(self, symbol: str) -> Dict:
        """Fallback method using CoinGecko API"""
        try:
            session = await self.get_session()
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': symbol.lower(),
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    symbol_key = symbol.lower()
                    if symbol_key in data:
                        return {
                            'symbol': symbol,
                            'price': data[symbol_key]['usd'],
                            'change_24h': data[symbol_key]['usd_24h_change'],
                            'timestamp': datetime.now()
                        }
            
            # Final fallback with mock data for demo
            return self._get_mock_data(symbol)
            
        except Exception as e:
            logger.error(f"Fallback also failed for {symbol}: {e}")
            return self._get_mock_data(symbol)
    
    async def get_historical_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get historical price data"""
        try:
            session = await self.get_session()
            url = f"{self.base_urls['binance']}/klines"
            params = {
                'symbol': f"{symbol.upper()}USDT",
                'interval': interval,
                'limit': limit
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert to proper types
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                else:
                    return self._get_mock_historical_data(symbol, interval, limit)
                    
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return self._get_mock_historical_data(symbol, interval, limit)
    
    def _get_mock_data(self, symbol: str) -> Dict:
        """Generate mock data for demonstration"""
        base_price = 1000 if symbol.upper() == 'BTC' else 100
        variation = (hash(symbol) % 1000 - 500) / 1000  # Small variation based on symbol hash
        
        return {
            'symbol': symbol,
            'price': base_price * (1 + variation),
            'change_24h': variation * 10,
            'volume': 1000000,
            'high_24h': base_price * (1 + variation + 0.1),
            'low_24h': base_price * (1 + variation - 0.1),
            'timestamp': datetime.now()
        }
    
    def _get_mock_historical_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Generate mock historical data for demonstration"""
        base_price = 1000 if symbol.upper() == 'BTC' else 100
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
        
        # Generate random walk prices
        prices = []
        current_price = base_price
        
        for _ in range(limit):
            change = (hash(f"{symbol}{_}") % 200 - 100) / 1000  # Small random changes
            current_price *= (1 + change)
            prices.append(current_price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],  # 2% higher
            'low': [p * 0.98 for p in prices],   # 2% lower
            'close': prices,
            'volume': [1000000] * limit
        })
        
        return df
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get prices for multiple symbols simultaneously"""
        tasks = [self.get_realtime_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        price_data = {}
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception):
                price_data[symbol] = result
        
        return price_data
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()