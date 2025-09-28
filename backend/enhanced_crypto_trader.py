# Enhanced import at the top of the file
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
import asyncio
from dataclasses import dataclass
import logging
import logging
import config
# Configure logging once, at the start
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Add to the top after other imports
try:
    from realtime_crypto_data import realtime_crypto_provider
    REALTIME_DATA_AVAILABLE = True
    logger.info("Real-time crypto data provider loaded successfully")
except ImportError:
    REALTIME_DATA_AVAILABLE = False
    logger.warning("Real-time crypto data provider not available - using fallback prices")

# Alpaca crypto trading
try:
    import alpaca_trade_api as tradeapi
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    from alpaca.data.live import CryptoDataStream
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca Trade API not available. Using simulation mode only.")

# Enhanced crypto signal
from enhanced_crypto_analyzer import CryptoSignal

logger = logging.getLogger(__name__)

@dataclass
class TradeExecution:
    trade_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    order_type: str
    status: str
    timestamp: datetime
    alpaca_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    notes: str = ""

class EnhancedCryptoTrader:
    """
    Advanced crypto trader with Alpaca API integration, 
    sophisticated order management, and risk controls
    """
    
    def __init__(self):
        self.simulation_mode = not ALPACA_AVAILABLE
        self.alpaca_client = None
        self.data_stream = None
        self.max_position_size = 0.10  # 10% of portfolio max
        self.max_daily_trades = 20
        self.min_order_size = 10.0  # Minimum $10 orders
        
        # Portfolio tracking
        self.sim_portfolio = {
            'cash': 50000.0,  # $50k starting capital
            'positions': {},  # {symbol: {'quantity': float, 'avg_price': float}}
            'pending_orders': {},
            'daily_trades': 0,
            'daily_pnl': 0.0,
            'total_pnl': 0.0,
            'last_reset': datetime.now().date()
        }
        
        self.trade_history = []
        self.risk_limits = {
            'max_drawdown': -0.20,  # -20% max drawdown
            'daily_loss_limit': -0.05,  # -5% daily loss limit
            'position_concentration': 0.30  # Max 30% in any single asset
        }
        
        self._initialize_alpaca()
    
    def _initialize_alpaca(self):
        """Initialize Alpaca trading client"""
        if not ALPACA_AVAILABLE:
            logger.info("Alpaca not available. Using simulation mode.")
            return
        
        try:
            # Initialize Alpaca trading client for crypto
            self.alpaca_client = TradingClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                paper=True  # Use paper trading
            )
            
            # Test connection
            account = self.alpaca_client.get_account()
            logger.info(f"Connected to Alpaca Paper Trading. Portfolio Value: ${account.portfolio_value}")
            
            # Initialize data stream for real-time crypto data
            self.data_stream = CryptoDataStream(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY
            )
            
        except Exception as e:
            logger.warning(f"Failed to connect to Alpaca: {e}. Using simulation mode.")
            self.simulation_mode = True
    
    async def execute_signal(self, signal: CryptoSignal) -> Dict[str, Any]:
        """
        Execute a crypto trading signal with advanced order management
        """
        try:
            # Pre-execution checks
            risk_check = self._perform_risk_checks(signal)
            if not risk_check['approved']:
                return {
                    'status': 'rejected',
                    'reason': risk_check['reason'],
                    'signal': signal.symbol
                }
            
            # Reset daily counters if needed
            self._reset_daily_counters()
            
            # Calculate optimal order size
            order_size = self._calculate_order_size(signal)
            if order_size < self.min_order_size:
                return {
                    'status': 'rejected',
                    'reason': f'Order size ${order_size} below minimum ${self.min_order_size}',
                    'signal': signal.symbol
                }
            
            # Execute based on action
            if signal.action == 'BUY':
                result = await self._execute_buy_order(signal, order_size)
            elif signal.action == 'SELL':
                result = await self._execute_sell_order(signal, order_size)
            else:
                return {
                    'status': 'hold',
                    'message': f'HOLD signal for {signal.symbol} - no execution',
                    'signal': signal.symbol
                }
            
            # Update portfolio tracking
            if result['status'] == 'success':
                self._update_portfolio(result['execution'])
                self.sim_portfolio['daily_trades'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'signal': signal.symbol
            }
    
    def _perform_risk_checks(self, signal: CryptoSignal) -> Dict[str, Any]:
        """
        Comprehensive pre-trade risk checks
        """
        # Check daily trade limit
        if self.sim_portfolio['daily_trades'] >= self.max_daily_trades:
            return {
                'approved': False,
                'reason': f'Daily trade limit reached ({self.max_daily_trades})'
            }
        
        # Check daily loss limit
        if self.sim_portfolio['daily_pnl'] <= self.risk_limits['daily_loss_limit'] * self.sim_portfolio['cash']:
            return {
                'approved': False,
                'reason': 'Daily loss limit reached'
            }
        
        # Check maximum drawdown
        total_value = self._calculate_portfolio_value()
        if total_value <= self.sim_portfolio['cash'] * (1 + self.risk_limits['max_drawdown']):
            return {
                'approved': False,
                'reason': 'Maximum drawdown limit reached'
            }
        
        # Check position concentration
        if signal.action == 'BUY':
            current_position_value = self._get_position_value(signal.symbol)
            proposed_position_value = current_position_value + (signal.position_size * total_value)
            concentration = proposed_position_value / total_value
            
            if concentration > self.risk_limits['position_concentration']:
                return {
                    'approved': False,
                    'reason': f'Position concentration limit exceeded ({concentration:.1%})'
                }
        
        # Check minimum confidence threshold
        if signal.confidence < 0.3:
            return {
                'approved': False,
                'reason': f'Signal confidence too low ({signal.confidence:.1%})'
            }
        
        return {'approved': True, 'reason': 'All risk checks passed'}
    
    async def _execute_buy_order(self, signal: CryptoSignal, order_size: float) -> Dict[str, Any]:
        """
        Execute a buy order for crypto
        """
        try:
            if self.simulation_mode:
                return await self._simulate_buy_order(signal, order_size)
            
            # Prepare Alpaca crypto buy order
            symbol = signal.symbol.upper()
            if not symbol.endswith('USD'):
                symbol += 'USD'  # Alpaca crypto pairs end with USD
            
            # Calculate quantity (in crypto units, not USD)
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return {
                    'status': 'error',
                    'message': f'Could not get current price for {symbol}'
                }
            
            quantity = order_size / current_price
            
            # Create market order (for immediate execution)
            market_order = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            
            # Submit order to Alpaca
            order = self.alpaca_client.submit_order(order_data=market_order)
            
            # Create execution record
            execution = TradeExecution(
                trade_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side='buy',
                quantity=quantity,
                price=current_price,
                order_type='market',
                status='submitted',
                timestamp=datetime.now(),
                alpaca_order_id=str(order.id),
                notes=f'ML confidence: {signal.confidence:.1%}'
            )
            
            # Wait for fill (simplified - in production use webhooks)
            await asyncio.sleep(2)
            filled_order = self.alpaca_client.get_order_by_id(order.id)
            
            if filled_order.status == 'filled':
                execution.status = 'filled'
                execution.fill_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else current_price
                execution.commission = 0.0  # Alpaca crypto has no commissions
            
            return {
                'status': 'success',
                'message': f'Buy order executed: {quantity:.6f} {signal.symbol} at ${current_price:.2f}',
                'execution': execution,
                'alpaca_order': filled_order
            }
            
        except Exception as e:
            logger.error(f"Alpaca buy order error for {signal.symbol}: {e}")
            return await self._simulate_buy_order(signal, order_size)
    
    async def _execute_sell_order(self, signal: CryptoSignal, order_size: float) -> Dict[str, Any]:
        """
        Execute a sell order for crypto
        """
        try:
            # Check if we have position to sell
            if signal.symbol not in self.sim_portfolio['positions']:
                return {
                    'status': 'error',
                    'message': f'No position to sell for {signal.symbol}'
                }
            
            position = self.sim_portfolio['positions'][signal.symbol]
            available_quantity = position['quantity']
            
            if self.simulation_mode:
                return await self._simulate_sell_order(signal, min(order_size, available_quantity * position['avg_price']))
            
            # Alpaca crypto sell order
            symbol = signal.symbol.upper()
            if not symbol.endswith('USD'):
                symbol += 'USD'
            
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return {
                    'status': 'error',
                    'message': f'Could not get current price for {symbol}'
                }
            
            # Calculate quantity to sell
            sell_quantity = min(available_quantity, order_size / current_price)
            
            # Create market sell order
            market_order = MarketOrderRequest(
                symbol=symbol,
                qty=sell_quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            
            # Submit order
            order = self.alpaca_client.submit_order(order_data=market_order)
            
            # Create execution record
            execution = TradeExecution(
                trade_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side='sell',
                quantity=sell_quantity,
                price=current_price,
                order_type='market',
                status='submitted',
                timestamp=datetime.now(),
                alpaca_order_id=str(order.id),
                notes=f'ML confidence: {signal.confidence:.1%}'
            )
            
            # Wait for fill
            await asyncio.sleep(2)
            filled_order = self.alpaca_client.get_order_by_id(order.id)
            
            if filled_order.status == 'filled':
                execution.status = 'filled'
                execution.fill_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else current_price
                execution.commission = 0.0
            
            return {
                'status': 'success',
                'message': f'Sell order executed: {sell_quantity:.6f} {signal.symbol} at ${current_price:.2f}',
                'execution': execution,
                'alpaca_order': filled_order
            }
            
        except Exception as e:
            logger.error(f"Alpaca sell order error for {signal.symbol}: {e}")
            return await self._simulate_sell_order(signal, order_size)
    
    async def _simulate_buy_order(self, signal: CryptoSignal, order_size: float) -> Dict[str, Any]:
        """
        Simulate a crypto buy order with REAL prices
        """
        try:
            # Import real-time data provider
            from realtime_crypto_data import realtime_crypto_provider
            
            # Get REAL current price
            real_price = await realtime_crypto_provider.get_real_price(signal.symbol)
            
            if real_price is None or real_price <= 0:
                return {
                    'status': 'error',
                    'message': f'Could not get real price for {signal.symbol}'
                }
            
            current_price = real_price
            quantity = order_size / current_price
            
            logger.info(f"Real-time {signal.symbol} price: ${current_price:,.2f}, buying ${order_size:.2f} = {quantity:.6f} {signal.symbol}")
            
            # Check if we have enough cash
            if order_size > self.sim_portfolio['cash']:
                return {
                    'status': 'error',
                    'message': f'Insufficient cash: need ${order_size:.2f}, have ${self.sim_portfolio["cash"]:.2f}'
                }
            
            # Execute simulated trade with real price
            self.sim_portfolio['cash'] -= order_size
            
            if signal.symbol in self.sim_portfolio['positions']:
                # Average down/up existing position
                existing = self.sim_portfolio['positions'][signal.symbol]
                total_quantity = existing['quantity'] + quantity
                total_cost = (existing['quantity'] * existing['avg_price']) + order_size
                avg_price = total_cost / total_quantity
                
                self.sim_portfolio['positions'][signal.symbol] = {
                    'quantity': total_quantity,
                    'avg_price': avg_price
                }
            else:
                # New position
                self.sim_portfolio['positions'][signal.symbol] = {
                    'quantity': quantity,
                    'avg_price': current_price
                }
            
            # Create execution record
            execution = TradeExecution(
                trade_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side='buy',
                quantity=quantity,
                price=current_price,
                order_type='market_sim',
                status='filled',
                timestamp=datetime.now(),
                notes=f'Simulated trade with REAL price - ML confidence: {signal.confidence:.1%}'
            )
            
            return {
                'status': 'success',
                'message': f'Real-time buy: {quantity:.6f} {signal.symbol} at ${current_price:,.2f} (${order_size:.2f} total)',
                'execution': execution,
                'real_price_used': True
            }
            
        except Exception as e:
            logger.error(f"Simulation buy error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _simulate_sell_order(self, signal: CryptoSignal, order_size: float) -> Dict[str, Any]:
        """
        Simulate a crypto sell order with REAL prices
        """
        try:
            if signal.symbol not in self.sim_portfolio['positions']:
                return {
                    'status': 'error',
                    'message': f'No position to sell for {signal.symbol}'
                }
            
            # Import real-time data provider
            from realtime_crypto_data import realtime_crypto_provider
            
            # Get REAL current price
            real_price = await realtime_crypto_provider.get_real_price(signal.symbol)
            
            if real_price is None or real_price <= 0:
                return {
                    'status': 'error',
                    'message': f'Could not get real price for {signal.symbol}'
                }
            
            position = self.sim_portfolio['positions'][signal.symbol]
            current_price = real_price
            
            logger.info(f"Real-time {signal.symbol} price: ${current_price:,.2f}, position avg: ${position['avg_price']:,.2f}")
            
            # Calculate quantity to sell
            max_sell_value = position['quantity'] * current_price
            sell_value = min(order_size, max_sell_value)
            sell_quantity = sell_value / current_price
            
            # Execute simulated sell with real price
            self.sim_portfolio['cash'] += sell_value
            
            # Update position
            remaining_quantity = position['quantity'] - sell_quantity
            if remaining_quantity > 0.000001:  # Keep position if significant amount remains
                self.sim_portfolio['positions'][signal.symbol]['quantity'] = remaining_quantity
            else:
                del self.sim_portfolio['positions'][signal.symbol]
            
            # Calculate P&L using real prices
            pnl = (current_price - position['avg_price']) * sell_quantity
            pnl_percentage = ((current_price - position['avg_price']) / position['avg_price']) * 100 if position['avg_price'] > 0 else 0
            
            self.sim_portfolio['daily_pnl'] += pnl
            self.sim_portfolio['total_pnl'] += pnl
            
            # Create execution record
            execution = TradeExecution(
                trade_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side='sell',
                quantity=sell_quantity,
                price=current_price,
                order_type='market_sim',
                status='filled',
                timestamp=datetime.now(),
                notes=f'Simulated trade with REAL price - P&L: ${pnl:.2f} ({pnl_percentage:+.1f}%) - ML confidence: {signal.confidence:.1%}'
            )
            
            return {
                'status': 'success',
                'message': f'Real-time sell: {sell_quantity:.6f} {signal.symbol} at ${current_price:,.2f} (P&L: ${pnl:+.2f}, {pnl_percentage:+.1f}%)',
                'execution': execution,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'real_price_used': True
            }
            
        except Exception as e:
            logger.error(f"Simulation sell error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }['positions'][signal.symbol]
            current_price = signal.price_target or position['avg_price']
            
            # Calculate quantity to sell
            max_sell_value = position['quantity'] * current_price
            sell_value = min(order_size, max_sell_value)
            sell_quantity = sell_value / current_price
            
            # Execute simulated sell
            self.sim_portfolio['cash'] += sell_value
            
            # Update position
            remaining_quantity = position['quantity'] - sell_quantity
            if remaining_quantity > 0.000001:  # Keep position if significant amount remains
                self.sim_portfolio['positions'][signal.symbol]['quantity'] = remaining_quantity
            else:
                del self.sim_portfolio['positions'][signal.symbol]
            
            # Calculate P&L
            pnl = (current_price - position['avg_price']) * sell_quantity
            self.sim_portfolio['daily_pnl'] += pnl
            self.sim_portfolio['total_pnl'] += pnl
            
            # Create execution record
            execution = TradeExecution(
                trade_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side='sell',
                quantity=sell_quantity,
                price=current_price,
                order_type='market_sim',
                status='filled',
                timestamp=datetime.now(),
                notes=f'Simulated trade - P&L: ${pnl:.2f} - ML confidence: {signal.confidence:.1%}'
            )
            
            return {
                'status': 'success',
                'message': f'Simulated sell: {sell_quantity:.6f} {signal.symbol} at ${current_price:.2f} (P&L: ${pnl:.2f})',
                'execution': execution,
                'pnl': pnl
            }
            
        except Exception as e:
            logger.error(f"Simulation sell error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current crypto price from real-time APIs
        """
        try:
            # Import the real-time data provider
            from realtime_crypto_data import realtime_crypto_provider
            
            # Get REAL current price
            price = await realtime_crypto_provider.get_real_price(symbol)
            
            if price and price > 0:
                logger.info(f"Real-time price for {symbol}: ${price:,.2f}")
                return price
            else:
                logger.warning(f"Could not get real price for {symbol}, using fallback")
                # Fallback prices based on typical ranges (better than hardcoded 50000)
                fallback_prices = {
                    'BTC': 43000.0, 'ETH': 2300.0, 'BNB': 310.0, 'SOL': 98.0,
                    'ADA': 0.48, 'XRP': 0.52, 'DOT': 7.2, 'AVAX': 36.0,
                    'MATIC': 0.85, 'LINK': 14.5, 'UNI': 6.8, 'LTC': 73.0,
                    'ALGO': 0.18, 'ATOM': 10.2, 'DOGE': 0.088, 'SHIB': 0.000024
                }
                return fallback_prices.get(symbol, 1000.0)  # Default fallback
            
        except Exception as e:
            logger.error(f"Error getting real price for {symbol}: {e}")
            # Emergency fallback
            return {'BTC': 43000, 'ETH': 2300, 'SOL': 98}.get(symbol, 100.0)
    
    def _calculate_order_size(self, signal: CryptoSignal) -> float:
        """
        Calculate optimal order size based on signal and portfolio
        """
        try:
            portfolio_value = self._calculate_portfolio_value()
            
            # Base size on signal's position_size recommendation
            base_size = signal.position_size * portfolio_value
            
            # Adjust for confidence
            confidence_adjustment = signal.confidence
            adjusted_size = base_size * confidence_adjustment
            
            # Apply maximum position size limit
            max_size = portfolio_value * self.max_position_size
            final_size = min(adjusted_size, max_size)
            
            # Ensure minimum order size
            return max(final_size, self.min_order_size)
            
        except Exception as e:
            logger.error(f"Order size calculation error: {e}")
            return self.min_order_size
    
    def _calculate_portfolio_value(self) -> float:
        """
        Calculate total portfolio value using REAL current prices
        """
        try:
            total_value = self.sim_portfolio['cash']
            
            if not self.sim_portfolio['positions']:
                return total_value
            
            # Get real-time prices for all positions
            position_symbols = list(self.sim_portfolio['positions'].keys())
            
            # Use asyncio to get real prices
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._get_portfolio_prices(position_symbols))
                        real_prices = future.result(timeout=10)
                else:
                    # Safe to run async
                    real_prices = asyncio.run(self._get_portfolio_prices(position_symbols))
            except:
                # Fallback: use average prices
                real_prices = {}
                for symbol, position in self.sim_portfolio['positions'].items():
                    real_prices[symbol] = position['avg_price']
            
            # Calculate position values with real prices
            for symbol, position in self.sim_portfolio['positions'].items():
                if symbol in real_prices:
                    current_price = real_prices[symbol]
                    position_value = position['quantity'] * current_price
                    total_value += position_value
                    
                    # Update unrealized P&L
                    unrealized_pnl = (current_price - position['avg_price']) * position['quantity']
                    logger.debug(f"{symbol}: {position['quantity']:.6f} @ ${current_price:.2f} = ${position_value:.2f} (P&L: ${unrealized_pnl:+.2f})")
                else:
                    # Fallback to average price
                    position_value = position['quantity'] * position['avg_price']
                    total_value += position_value
            
            return total_value
            
        except Exception as e:
            logger.error(f"Portfolio value calculation error: {e}")
            # Emergency fallback: use cash + estimated position values
            estimated_value = self.sim_portfolio['cash']
            for position in self.sim_portfolio['positions'].values():
                estimated_value += position['quantity'] * position['avg_price']
            return estimated_value
    
    async def _get_portfolio_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get real-time prices for portfolio symbols
        """
        try:
            from realtime_crypto_data import realtime_crypto_provider
            return await realtime_crypto_provider.get_multiple_prices(symbols)
        except Exception as e:
            logger.error(f"Error getting portfolio prices: {e}")
            return {}
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio status with REAL prices
        """
        try:
            total_value = self._calculate_portfolio_value()
            positions_value = total_value - self.sim_portfolio['cash']
            
            # Calculate metrics
            total_return = (total_value - 50000) / 50000 if total_value > 0 else 0
            cash_ratio = self.sim_portfolio['cash'] / total_value if total_value > 0 else 1
            
            # Get position breakdown with real prices
            position_breakdown = []
            try:
                # Try to get real-time prices for positions
                position_symbols = list(self.sim_portfolio['positions'].keys())
                if position_symbols:
                    # Quick sync call for immediate status
                    try:
                        from realtime_crypto_data import realtime_crypto_provider
                        # Use cached prices if available
                        real_prices = {}
                        for symbol in position_symbols:
                            cached_key = f"price_{symbol}"
                            if cached_key in realtime_crypto_provider.cache:
                                cached_price, timestamp = realtime_crypto_provider.cache[cached_key]
                                if (datetime.now() - timestamp).seconds < 60:  # Use 1-minute cached prices
                                    real_prices[symbol] = cached_price
                    except:
                        real_prices = {}
                
                for symbol, position in self.sim_portfolio['positions'].items():
                    current_price = real_prices.get(symbol, position['avg_price'])
                    position_value = position['quantity'] * current_price
                    allocation = position_value / total_value if total_value > 0 else 0
                    unrealized_pnl = (current_price - position['avg_price']) * position['quantity']
                    unrealized_pnl_pct = (current_price - position['avg_price']) / position['avg_price'] * 100 if position['avg_price'] > 0 else 0
                    
                    position_breakdown.append({
                        'symbol': symbol,
                        'quantity': position['quantity'],
                        'avg_price': position['avg_price'],
                        'current_price': current_price,
                        'market_value': position_value,
                        'allocation': allocation,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': unrealized_pnl_pct,
                        'using_real_price': symbol in real_prices
                    })
            except Exception as e:
                logger.error(f"Error calculating position breakdown: {e}")
                # Fallback to basic calculation
                for symbol, position in self.sim_portfolio['positions'].items():
                    position_value = position['quantity'] * position['avg_price']
                    allocation = position_value / total_value if total_value > 0 else 0
                    
                    position_breakdown.append({
                        'symbol': symbol,
                        'quantity': position['quantity'],
                        'avg_price': position['avg_price'],
                        'current_price': position['avg_price'],  # Fallback
                        'market_value': position_value,
                        'allocation': allocation,
                        'unrealized_pnl': 0,
                        'unrealized_pnl_pct': 0,
                        'using_real_price': False
                    })
            
            # Recent trades summary
            recent_trades = [
                {
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'timestamp': trade.timestamp.isoformat(),
                    'status': trade.status,
                    'notes': trade.notes
                }
                for trade in self.trade_history[-10:]  # Last 10 trades
            ]
            
            return {
                'total_value': round(total_value, 2),
                'cash': round(self.sim_portfolio['cash'], 2),
                'positions_value': round(positions_value, 2),
                'total_return': round(total_return, 4),
                'total_return_pct': round(total_return * 100, 2),
                'daily_pnl': round(self.sim_portfolio['daily_pnl'], 2),
                'total_pnl': round(self.sim_portfolio['total_pnl'], 2),
                'daily_trades': self.sim_portfolio['daily_trades'],
                'cash_ratio': round(cash_ratio, 4),
                'positions_count': len(self.sim_portfolio['positions']),
                'position_breakdown': position_breakdown,
                'recent_trades': recent_trades,
                'risk_metrics': {
                    'max_daily_trades': self.max_daily_trades,
                    'max_position_size': self.max_position_size,
                    'daily_loss_limit': self.risk_limits['daily_loss_limit'],
                    'max_drawdown_limit': self.risk_limits['max_drawdown']
                },
                'trading_mode': 'simulation_with_real_prices' if self.simulation_mode else 'alpaca_paper',
                'real_time_pricing': True,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio status error: {e}")
            return {
                'error': str(e),
                'total_value': 0,
                'trading_mode': 'simulation' if self.simulation_mode else 'alpaca_paper',
                'real_time_pricing': False
            }
    
    def _get_position_value(self, symbol: str) -> float:
        """
        Get current value of a position
        """
        if symbol not in self.sim_portfolio['positions']:
            return 0.0
        
        position = self.sim_portfolio['positions'][symbol]
        return position['quantity'] * position['avg_price']
    
    def _update_portfolio(self, execution: TradeExecution):
        """
        Update portfolio tracking after trade execution
        """
        try:
            self.trade_history.append(execution)
            
            # Update daily P&L if it's a sell order
            if execution.side == 'sell' and execution.symbol in self.sim_portfolio['positions']:
                position = self.sim_portfolio['positions'][execution.symbol]
                pnl = (execution.price - position['avg_price']) * execution.quantity
                self.sim_portfolio['daily_pnl'] += pnl
                self.sim_portfolio['total_pnl'] += pnl
            
        except Exception as e:
            logger.error(f"Portfolio update error: {e}")
    
    def _reset_daily_counters(self):
        """
        Reset daily counters if it's a new trading day
        """
        try:
            today = datetime.now().date()
            if today != self.sim_portfolio['last_reset']:
                self.sim_portfolio['daily_trades'] = 0
                self.sim_portfolio['daily_pnl'] = 0.0
                self.sim_portfolio['last_reset'] = today
                logger.info(f"Daily counters reset for {today}")
                
        except Exception as e:
            logger.error(f"Daily reset error: {e}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio status
        """
        try:
            total_value = self._calculate_portfolio_value()
            positions_value = total_value - self.sim_portfolio['cash']
            
            # Calculate metrics
            total_return = (total_value - 50000) / 50000 if total_value > 0 else 0
            cash_ratio = self.sim_portfolio['cash'] / total_value if total_value > 0 else 1
            
            # Get position breakdown
            position_breakdown = []
            for symbol, position in self.sim_portfolio['positions'].items():
                position_value = position['quantity'] * position['avg_price']
                allocation = position_value / total_value if total_value > 0 else 0
                
                position_breakdown.append({
                    'symbol': symbol,
                    'quantity': position['quantity'],
                    'avg_price': position['avg_price'],
                    'market_value': position_value,
                    'allocation': allocation,
                    'unrealized_pnl': 0  # Would need current prices to calculate
                })
            
            # Recent trades summary
            recent_trades = [
                {
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'timestamp': trade.timestamp.isoformat(),
                    'status': trade.status
                }
                for trade in self.trade_history[-10:]  # Last 10 trades
            ]
            
            return {
                'total_value': total_value,
                'cash': self.sim_portfolio['cash'],
                'positions_value': positions_value,
                'total_return': total_return,
                'daily_pnl': self.sim_portfolio['daily_pnl'],
                'total_pnl': self.sim_portfolio['total_pnl'],
                'daily_trades': self.sim_portfolio['daily_trades'],
                'cash_ratio': cash_ratio,
                'positions_count': len(self.sim_portfolio['positions']),
                'position_breakdown': position_breakdown,
                'recent_trades': recent_trades,
                'risk_metrics': {
                    'max_daily_trades': self.max_daily_trades,
                    'max_position_size': self.max_position_size,
                    'daily_loss_limit': self.risk_limits['daily_loss_limit'],
                    'max_drawdown_limit': self.risk_limits['max_drawdown']
                },
                'trading_mode': 'simulation' if self.simulation_mode else 'alpaca_paper',
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio status error: {e}")
            return {
                'error': str(e),
                'total_value': 0,
                'trading_mode': 'simulation' if self.simulation_mode else 'alpaca_paper'
            }
    
    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get formatted trade history
        """
        try:
            recent_trades = self.trade_history[-limit:] if limit else self.trade_history
            
            formatted_trades = []
            for trade in recent_trades:
                formatted_trades.append({
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'order_type': trade.order_type,
                    'status': trade.status,
                    'timestamp': trade.timestamp.isoformat(),
                    'alpaca_order_id': trade.alpaca_order_id,
                    'fill_price': trade.fill_price,
                    'commission': trade.commission,
                    'slippage': trade.slippage,
                    'notes': trade.notes
                })
            
            return formatted_trades
            
        except Exception as e:
            logger.error(f"Trade history error: {e}")
            return []
    
    def close_position(self, symbol: str, percentage: float = 1.0) -> Dict[str, Any]:
        """
        Close a position (full or partial)
        """
        try:
            if symbol not in self.sim_portfolio['positions']:
                return {
                    'status': 'error',
                    'message': f'No position found for {symbol}'
                }
            
            position = self.sim_portfolio['positions'][symbol]
            close_quantity = position['quantity'] * percentage
            
            # Create a mock sell signal for position closing
            from enhanced_crypto_analyzer import CryptoSignal
            close_signal = CryptoSignal(
                symbol=symbol,
                action='SELL',
                confidence=0.8,
                technical_score=0.5,
                ml_score=0.5,
                risk_score=0.3,
                volatility_forecast=0.02,
                price_target=position['avg_price'],  # Use avg price as target
                stop_loss=None,
                position_size=percentage,
                reasoning=f'Position close requested - {percentage:.0%}',
                model_predictions={},
                timestamp=datetime.now()
            )
            
            # Execute the close
            return asyncio.run(self._execute_sell_order(close_signal, close_quantity * position['avg_price']))
            
        except Exception as e:
            logger.error(f"Position close error for {symbol}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def update_risk_limits(self, new_limits: Dict[str, float]) -> Dict[str, Any]:
        """
        Update risk management limits
        """
        try:
            for key, value in new_limits.items():
                if key in self.risk_limits:
                    old_value = self.risk_limits[key]
                    self.risk_limits[key] = value
                    logger.info(f"Updated risk limit {key}: {old_value} -> {value}")
            
            return {
                'status': 'success',
                'updated_limits': self.risk_limits,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Risk limits update error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics
        """
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0
                }
            
            # Basic metrics
            total_trades = len(self.trade_history)
            sell_trades = [t for t in self.trade_history if t.side == 'sell']
            
            # Calculate wins/losses (simplified)
            wins = sum(
                1
    for t in sell_trades
    if 'P&L:' in t.notes and float(t.notes.split('P&L: ')[1].split(' ')[0]) > 0
)

            win_rate = wins / len(sell_trades) if sell_trades else 0
            
            # Portfolio performance
            initial_value = 50000
            current_value = self._calculate_portfolio_value()
            total_return = (current_value - initial_value) / initial_value
            
            return {
                'total_trades': total_trades,
                'buy_trades': len([t for t in self.trade_history if t.side == 'buy']),
                'sell_trades': len(sell_trades),
                'win_rate': win_rate,
                'total_return': total_return,
                'total_pnl': self.sim_portfolio['total_pnl'],
                'daily_pnl': self.sim_portfolio['daily_pnl'],
                'initial_value': initial_value,
                'current_value': current_value,
                'max_drawdown': -0.05,  # Placeholder - would need historical values
                'sharpe_ratio': 0.8,    # Placeholder - would need return series
                'trading_days': (datetime.now().date() - self.sim_portfolio['last_reset']).days + 1,
                'avg_trades_per_day': total_trades / max(1, (datetime.now().date() - datetime(2024, 1, 1).date()).days),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return {
                'error': str(e),
                'total_trades': len(self.trade_history)
            }

# Global instance
enhanced_crypto_trader = EnhancedCryptoTrader()