# enhanced_trading_executor.py - Real-time Trading with Multiple APIs
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid
import json
import ccxt

exchange = ccxt.coinbase({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_API_SECRET',
    'password': 'YOUR_API_PASSPHRASE',
    'enableRateLimit': True,
})

print(exchange.fetch_balance())

# Import trading APIs
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

import config

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    current_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: float = 0.02
    reasoning: str = ""
    timestamp: datetime = None
    risk_score: float = 0.3
    volatility_forecast: float = 0.02
    model_predictions: Dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.model_predictions is None:
            self.model_predictions = {}

class RealTimeTradeExecutor:
    """Real-time trade execution across multiple APIs"""
    
    def __init__(self):
        self.alpaca_client = None
        self.crypto_exchanges = {}
        self.paper_trading = True
        
        # Initialize trading platforms
        self._initialize_alpaca()
        self._initialize_crypto_exchanges()
        
        # Trading limits and risk management
        self.max_position_size = 0.05  # 5% max per position
        self.max_portfolio_risk = 0.20  # 20% total portfolio risk
        self.min_order_value = 10.0    # $10 minimum
        
        logger.info(f"Trade executor initialized - Paper Trading: {self.paper_trading}")
    
    def _initialize_alpaca(self):
        """Initialize Alpaca paper trading"""
        if not ALPACA_AVAILABLE:
            logger.warning("Alpaca not available")
            return
        
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            logger.warning("Alpaca credentials missing")
            return
        
        try:
            self.alpaca_client = TradingClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                paper=True  # Always use paper trading for safety
            )
            
            # Test connection
            account = self.alpaca_client.get_account()
            logger.info(f"✅ Alpaca connected - Portfolio: ${float(account.portfolio_value):,.2f}")
            
        except Exception as e:
            logger.error(f"Alpaca initialization failed: {e}")
            self.alpaca_client = None
    
    def _initialize_crypto_exchanges(self):
        """Initialize crypto exchange connections for paper trading"""
        if not CCXT_AVAILABLE:
            logger.warning("CCXT not available - limited crypto support")
            return
        
        try:
            # Initialize paper trading exchanges
            self.crypto_exchanges = {
                'binance': ccxt.binance({
                    'sandbox': True,  # Paper trading mode
                    'apiKey': 'test',  # Sandbox keys
                    'secret': 'test',
                    'timeout': 30000,
                }),
                'coinbase':ccxt.coinbase
({
                    'sandbox': True,
                    'timeout': 30000,
                })
            }
            
            logger.info("✅ Crypto exchanges initialized (paper trading)")
            
        except Exception as e:
            logger.error(f"Crypto exchange initialization failed: {e}")
            self.crypto_exchanges = {}
    
    async def execute_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute trading signal with real-time order placement"""
        try:
            logger.info(f"🔄 Executing {signal.action} for {signal.symbol} (confidence: {signal.confidence:.1%})")
            
            # Validate signal
            validation_result = self._validate_signal(signal)
            if not validation_result['valid']:
                return {
                    'status': 'rejected',
                    'reason': validation_result['reason'],
                    'signal': signal.symbol
                }
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            
            # Route to appropriate exchange
            if signal.symbol in config.ALPACA_SUPPORTED_CRYPTOS and self.alpaca_client:
                return await self._execute_alpaca_order(signal, position_size)
            else:
                return await self._execute_crypto_order(signal, position_size)
                
        except Exception as e:
            logger.error(f"Signal execution error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'signal': signal.symbol
            }
    
    def _validate_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Validate trading signal before execution"""
        
        # Check confidence threshold
        if signal.confidence < config.CONFIDENCE_THRESHOLD:
            return {
                'valid': False,
                'reason': f'Confidence {signal.confidence:.1%} below threshold {config.CONFIDENCE_THRESHOLD:.1%}'
            }
        
        # Check action validity
        if signal.action not in ['BUY', 'SELL']:
            return {
                'valid': False,
                'reason': f'Invalid action: {signal.action}'
            }
        
        # Check price validity
        if signal.current_price <= 0:
            return {
                'valid': False,
                'reason': 'Invalid current price'
            }
        
        # Check position size
        if signal.position_size > self.max_position_size:
            return {
                'valid': False,
                'reason': f'Position size {signal.position_size:.1%} exceeds max {self.max_position_size:.1%}'
            }
        
        return {'valid': True}
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            # Get portfolio value
            portfolio_value = self._get_portfolio_value()
            
            # Base position size from signal
            base_amount = signal.position_size * portfolio_value
            
            # Adjust for confidence
            confidence_adjusted = base_amount * signal.confidence
            
            # Adjust for volatility (higher vol = smaller size)
            volatility_factor = max(0.5, 1.0 - signal.volatility_forecast)
            volatility_adjusted = confidence_adjusted * volatility_factor
            
            # Apply maximum position size limit
            max_amount = portfolio_value * self.max_position_size
            final_amount = min(volatility_adjusted, max_amount, config.TRADE_ALLOCATION_USD)
            
            return max(final_amount, self.min_order_value)
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return self.min_order_value
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            if self.alpaca_client:
                account = self.alpaca_client.get_account()
                return float(account.portfolio_value)
            return 100000.0  # Default for crypto-only
        except Exception:
            return 100000.0
    
    async def _execute_alpaca_order(self, signal: TradingSignal, amount_usd: float) -> Dict[str, Any]:
        """Execute order through Alpaca API"""
        try:
            symbol_map = {
                'BTC': 'BTCUSD', 'ETH': 'ETHUSD', 'DOGE': 'DOGEUSD',
                'LTC': 'LTCUSD', 'BCH': 'BCHUSD', 'AAVE': 'AAVEUSD',
                'UNI': 'UNIUSD', 'LINK': 'LINKUSD'
            }
            
            alpaca_symbol = symbol_map.get(signal.symbol)
            if not alpaca_symbol:
                return {
                    'status': 'error',
                    'message': f'Symbol {signal.symbol} not supported on Alpaca'
                }
            
            # Calculate quantity
            quantity = amount_usd / signal.current_price
            
            # Create order request
            side = OrderSide.BUY if signal.action == 'BUY' else OrderSide.SELL
            
            if signal.target_price and signal.stop_loss:
                # Bracket order with take profit and stop loss
                order_request = LimitOrderRequest(
                    symbol=alpaca_symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=signal.current_price,
                    order_class=OrderClass.BRACKET,
                    take_profit={'limit_price': signal.target_price},
                    stop_loss={'stop_price': signal.stop_loss}
                )
            else:
                # Simple market order
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            
            # Submit order
            order = self.alpaca_client.submit_order(order_request)
            
            logger.info(f"✅ Alpaca order submitted: {order.id}")
            
            return {
                'status': 'success',
                'order_id': order.id,
                'symbol': signal.symbol,
                'alpaca_symbol': alpaca_symbol,
                'side': signal.action,
                'quantity': quantity,
                'amount_usd': amount_usd,
                'price': signal.current_price,
                'platform': 'alpaca',
                'order_type': 'bracket' if signal.target_price else 'market',
                'message': f'Order executed on Alpaca: {quantity:.6f} {signal.symbol}',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Alpaca order execution failed: {e}")
            return {
                'status': 'error',
                'message': f'Alpaca execution failed: {str(e)}',
                'platform': 'alpaca'
            }
    
    async def _execute_crypto_order(self, signal: TradingSignal, amount_usd: float) -> Dict[str, Any]:
        """Execute order through crypto exchange API (paper trading)"""
        try:
            # For now, simulate the order since we're in paper trading mode
            quantity = amount_usd / signal.current_price
            
            # Simulate order execution delay
            await asyncio.sleep(0.5)
            
            # Generate simulated order ID
            order_id = f"CRYPTO_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"✅ Simulated crypto order: {quantity:.6f} {signal.symbol}")
            
            return {
                'status': 'success',
                'order_id': order_id,
                'symbol': signal.symbol,
                'side': signal.action,
                'quantity': quantity,
                'amount_usd': amount_usd,
                'price': signal.current_price,
                'platform': 'crypto_simulation',
                'order_type': 'market_simulation',
                'message': f'Simulated order: {quantity:.6f} {signal.symbol} @ ${signal.current_price:.6f}',
                'timestamp': datetime.now().isoformat(),
                'note': 'Paper trading simulation - no real funds involved'
            }
            
        except Exception as e:
            logger.error(f"Crypto order execution failed: {e}")
            return {
                'status': 'error',
                'message': f'Crypto execution failed: {str(e)}',
                'platform': 'crypto'
            }
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status"""
        try:
            portfolio_data = {
                'timestamp': datetime.now().isoformat(),
                'paper_trading': self.paper_trading,
                'platforms': []
            }
            
            # Alpaca portfolio
            if self.alpaca_client:
                try:
                    account = self.alpaca_client.get_account()
                    positions = self.alpaca_client.get_all_positions()
                    
                    alpaca_data = {
                        'platform': 'alpaca',
                        'status': 'connected',
                        'portfolio_value': float(account.portfolio_value),
                        'buying_power': float(account.buying_power),
                        'cash': float(account.cash),
                        'positions_count': len(positions),
                        'positions': [
                            {
                                'symbol': pos.symbol,
                                'quantity': float(pos.qty),
                                'market_value': float(pos.market_value) if pos.market_value else 0,
                                'unrealized_pl': float(pos.unrealized_pl) if pos.unrealized_pl else 0,
                                'side': pos.side
                            } for pos in positions
                        ]
                    }
                    portfolio_data['platforms'].append(alpaca_data)
                    portfolio_data['total_value'] = float(account.portfolio_value)
                    
                except Exception as e:
                    portfolio_data['platforms'].append({
                        'platform': 'alpaca',
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Crypto portfolio (simulated)
            portfolio_data['platforms'].append({
                'platform': 'crypto_simulation',
                'status': 'active',
                'note': 'Paper trading simulation',
                'estimated_value': 50000.0  # Simulated crypto portfolio value
            })
            
            if 'total_value' not in portfolio_data:
                portfolio_data['total_value'] = 100000.0
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Portfolio status error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get order history from all platforms"""
        try:
            all_orders = []
            
            # Alpaca orders
            if self.alpaca_client:
                try:
                    orders = self.alpaca_client.get_orders()
                    for order in orders[:limit]:
                        all_orders.append({
                            'order_id': order.id,
                            'symbol': order.symbol,
                            'side': order.side.value,
                            'quantity': float(order.qty),
                            'status': order.status.value,
                            'created_at': order.created_at.isoformat() if order.created_at else None,
                            'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                            'platform': 'alpaca'
                        })
                except Exception as e:
                    logger.error(f"Error getting Alpaca orders: {e}")
            
            return sorted(all_orders, key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Order history error: {e}")
            return []
    
    async def cancel_order(self, order_id: str, platform: str = None) -> Dict[str, Any]:
        """Cancel an open order"""
        try:
            if platform == 'alpaca' and self.alpaca_client:
                self.alpaca_client.cancel_order_by_id(order_id)
                return {
                    'status': 'success',
                    'message': f'Order {order_id} cancelled on Alpaca'
                }
            
            return {
                'status': 'error',
                'message': 'Order cancellation not supported for this platform'
            }
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_supported_symbols(self) -> Dict[str, Any]:
        """Get all supported trading symbols"""
        alpaca_cryptos = config.ALPACA_SUPPORTED_CRYPTOS if self.alpaca_client else []
        all_cryptos = config.ALL_CRYPTO_SYMBOLS
        
        return {
            'alpaca_supported': alpaca_cryptos,
            'all_cryptos': all_cryptos,
            'total_supported': len(all_cryptos),
            'real_trading_available': len(alpaca_cryptos),
            'simulation_available': len(all_cryptos),
            'paper_trading': self.paper_trading,
            'platforms_available': {
                'alpaca': self.alpaca_client is not None,
                'crypto_simulation': True
            }
        }

# Global instance
real_time_executor = RealTimeTradeExecutor()