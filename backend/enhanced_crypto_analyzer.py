# enhanced_crypto_analyzer.py - Advanced ML-based Crypto Analysis Engine (FIXED)
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML and statistical models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from scipy import stats

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM models disabled.")

# GARCH models
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("ARCH package not available. GARCH models disabled.")

# Technical analysis - use fallback if pandas_ta fails
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas_ta not available. Using basic technical indicators.")

logger = logging.getLogger(__name__)

@dataclass
class CryptoSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    technical_score: float
    ml_score: float
    risk_score: float
    volatility_forecast: float
    price_target: Optional[float]
    stop_loss: Optional[float]
    position_size: float
    reasoning: str
    model_predictions: Dict[str, float]
    timestamp: datetime

class BasicTechnicalAnalyzer:
    """Fallback technical analysis when pandas_ta is not available"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD indicator"""
        ema_fast = BasicTechnicalAnalyzer.ema(data, fast)
        ema_slow = BasicTechnicalAnalyzer.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = BasicTechnicalAnalyzer.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = BasicTechnicalAnalyzer.sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

class TechnicalAnalyzer:
    """Technical analysis wrapper that handles both pandas_ta and fallback"""
    
    def __init__(self):
        self.basic_ta = BasicTechnicalAnalyzer()
    
    def analyze(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Perform technical analysis and return score and reasoning
        """
        try:
            if len(df) < 50:
                return 0.5, "Insufficient data for technical analysis"
            
            signals = []
            reasoning_parts = []
            
            # Moving averages
            sma_20 = self.basic_ta.sma(df['close'], 20)
            sma_50 = self.basic_ta.sma(df['close'], 50)
            current_price = df['close'].iloc[-1]
            
            if current_price > sma_20.iloc[-1]:
                signals.append(0.6)
                reasoning_parts.append("Above 20-day SMA")
            else:
                signals.append(0.4)
                reasoning_parts.append("Below 20-day SMA")
            
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                signals.append(0.6)
                reasoning_parts.append("20-day > 50-day SMA")
            else:
                signals.append(0.4)
                reasoning_parts.append("20-day < 50-day SMA")
            
            # RSI
            rsi = self.basic_ta.rsi(df['close'])
            current_rsi = rsi.iloc[-1]
            
            if current_rsi < 30:
                signals.append(0.7)
                reasoning_parts.append("RSI oversold")
            elif current_rsi > 70:
                signals.append(0.3)
                reasoning_parts.append("RSI overbought")
            else:
                signals.append(0.5)
                reasoning_parts.append("RSI neutral")
            
            # MACD
            macd_data = self.basic_ta.macd(df['close'])
            if macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1]:
                signals.append(0.6)
                reasoning_parts.append("MACD bullish")
            else:
                signals.append(0.4)
                reasoning_parts.append("MACD bearish")
            
            # Volume trend (if available)
            if 'volume' in df.columns:
                vol_sma = df['volume'].rolling(20).mean()
                if df['volume'].iloc[-1] > vol_sma.iloc[-1]:
                    signals.append(0.6)
                    reasoning_parts.append("High volume")
                else:
                    signals.append(0.5)
                    reasoning_parts.append("Normal volume")
            
            # Calculate overall score
            tech_score = np.mean(signals)
            reasoning = " | ".join(reasoning_parts)
            
            return tech_score, reasoning
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return 0.5, f"Technical analysis error: {str(e)}"

class EnhancedCryptoAnalyzer:
    """
    Advanced crypto analyzer with multiple ML models, GARCH volatility modeling,
    and comprehensive risk assessment
    """
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.models = {}
        self.scalers = {}
        self.model_weights = {
            'technical': 0.3,
            'lstm': 0.25,
            'ensemble': 0.25,
            'garch': 0.2
        }
        self.min_data_points = 100
        self.setup_models()
        
    def setup_models(self):
        """Initialize ML models"""
        try:
            # Ensemble classifier for direction prediction
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            self.models['ensemble'] = VotingClassifier(
                estimators=[('rf', rf), ('xgb', xgb_model)],
                voting='soft'
            )
            
            # Scalers for different model inputs
            self.scalers['features'] = StandardScaler()
            self.scalers['price'] = MinMaxScaler()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
    
    async def analyze_crypto(self, df: pd.DataFrame, symbol: str) -> CryptoSignal:
        """
        Comprehensive crypto analysis using multiple ML approaches
        """
        if df is None or df.empty or len(df) < self.min_data_points:
            return self._create_fallback_signal(symbol, "Insufficient data for analysis")
        
        try:
            # 1. Technical Analysis
            tech_score, tech_reasoning = self.technical_analyzer.analyze(df)
            
            # 2. Feature Engineering
            features_df = self._engineer_features(df)
            
            # 3. ML Predictions
            ml_predictions = await self._get_ml_predictions(features_df, df)
            
            # 4. GARCH Volatility Forecasting
            vol_forecast = self._forecast_volatility(df)
            
            # 5. Risk Assessment
            risk_metrics = self._calculate_risk_metrics(df, vol_forecast)
            
            # 6. Signal Generation
            signal = self._generate_signal(
                symbol=symbol,
                df=df,
                tech_score=tech_score,
                tech_reasoning=tech_reasoning,
                ml_predictions=ml_predictions,
                vol_forecast=vol_forecast,
                risk_metrics=risk_metrics
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return self._create_fallback_signal(symbol, f"Analysis error: {str(e)}")
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for crypto data
        """
        features = df.copy()
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_momentum'] = df['close'].pct_change(5)
        features['price_acceleration'] = features['returns'].diff()
        
        # Volatility features
        features['realized_vol'] = features['returns'].rolling(20).std()
        features['vol_of_vol'] = features['realized_vol'].rolling(10).std()
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            features['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            features['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
        
        # Technical indicators using our basic implementation
        if len(df) >= 30:
            try:
                basic_ta = BasicTechnicalAnalyzer()
                
                # Trend indicators
                features["sma_20"] = basic_ta.sma(df["close"], 20)
                features["ema_12"] = basic_ta.ema(df["close"], 12)
                features["ema_26"] = basic_ta.ema(df["close"], 26)

                # Momentum indicators
                features["rsi"] = basic_ta.rsi(df["close"], 14)

                macd = basic_ta.macd(df["close"], fast=12, slow=26, signal=9)
                features["macd"] = macd["macd"]
                features["macd_signal"] = macd["signal"]
                features["macd_hist"] = macd["histogram"]

                # Volatility indicators
                bbands = basic_ta.bollinger_bands(df["close"], 20)
                features["bb_upper"] = bbands["upper"]
                features["bb_middle"] = bbands["middle"]
                features["bb_lower"] = bbands["lower"]

                features["atr"] = basic_ta.atr(df["high"], df["low"], df["close"], 14)

            except Exception as e:
                logger.warning(f"Technical indicators failed: {e}")
        
        # Regime detection features
        features['bull_bear_regime'] = self._detect_market_regime(df)
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'vol_lag_{lag}'] = features['realized_vol'].shift(lag)
        
        # Time-based features
        features['hour'] = pd.to_datetime(df.index).hour if hasattr(df.index, 'hour') else 0
        features['day_of_week'] = pd.to_datetime(df.index).dayofweek if hasattr(df.index, 'dayofweek') else 0
        
        return features.fillna(method='ffill').fillna(0)
    
    async def _get_ml_predictions(self, features_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict[str, float]:
        """
        Get predictions from multiple ML models
        """
        predictions = {}
        
        try:
            # Prepare features for ML models
            feature_columns = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            X = features_df[feature_columns].fillna(0)
            
            # Create target variable (next period return direction)
            y = (features_df['returns'].shift(-1) > 0).astype(int)
            
            # Remove last row (no target)
            X = X[:-1]
            y = y[:-1]
            
            if len(X) < 50:  # Minimum data for training
                return {'ensemble': 0.5, 'lstm': 0.5}
            
            # Ensemble model prediction
            predictions['ensemble'] = await self._predict_ensemble(X, y)
            
            # LSTM prediction (if TensorFlow available)
            if TF_AVAILABLE:
                predictions['lstm'] = await self._predict_lstm(price_df)
            else:
                predictions['lstm'] = 0.5
            
            # Additional statistical models
            predictions['mean_reversion'] = self._predict_mean_reversion(features_df)
            predictions['momentum'] = self._predict_momentum(features_df)
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            predictions = {'ensemble': 0.5, 'lstm': 0.5, 'mean_reversion': 0.5, 'momentum': 0.5}
        
        return predictions
    
    async def _predict_ensemble(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Ensemble model prediction
        """
        try:
            if len(X) < 100:
                return 0.5
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scalers['features'].fit_transform(X_train)
            X_test_scaled = self.scalers['features'].transform(X_test)
            
            # Train ensemble
            self.models['ensemble'].fit(X_train_scaled, y_train)
            
            # Predict on latest data
            latest_features = X.iloc[-1:].fillna(0)
            latest_scaled = self.scalers['features'].transform(latest_features)
            prediction_proba = self.models['ensemble'].predict_proba(latest_scaled)[0]
            
            return prediction_proba[1] if len(prediction_proba) > 1 else 0.5
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return 0.5
    
    async def _predict_lstm(self, df: pd.DataFrame, sequence_length: int = 60) -> float:
        """
        LSTM neural network prediction
        """
        try:
            if len(df) < sequence_length + 50:
                return 0.5
            
            # Prepare data for LSTM
            prices = df['close'].values.reshape(-1, 1)
            scaled_prices = self.scalers['price'].fit_transform(prices)
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(scaled_prices) - 1):
                X.append(scaled_prices[i-sequence_length:i, 0])
                y.append(1 if scaled_prices[i+1, 0] > scaled_prices[i, 0] else 0)
            
            X, y = np.array(X), np.array(y)
            
            if len(X) < 50:
                return 0.5
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            # Train model
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            
            # Predict on latest sequence
            latest_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
            prediction = model.predict(latest_sequence, verbose=0)[0, 0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return 0.5
    
    def _predict_mean_reversion(self, features_df: pd.DataFrame) -> float:
        """
        Mean reversion prediction using statistical tests
        """
        try:
            returns = features_df['returns'].dropna()
            if len(returns) < 30:
                return 0.5
            
            # Simple mean reversion logic
            current_return = returns.iloc[-1]
            recent_mean = returns.tail(20).mean()
            
            if current_return < recent_mean:
                return 0.6  # Expect reversion up
            else:
                return 0.4  # Expect reversion down
                
        except Exception as e:
            logger.error(f"Mean reversion prediction error: {e}")
            return 0.5
    
    def _predict_momentum(self, features_df: pd.DataFrame) -> float:
        """
        Momentum-based prediction
        """
        try:
            if 'price_momentum' not in features_df.columns:
                return 0.5
            
            momentum = features_df['price_momentum'].tail(5).mean()
            volatility = features_df['realized_vol'].tail(10).mean()
            
            # Adjust momentum by volatility (higher vol = less reliable momentum)
            adjusted_momentum = momentum / (1 + volatility)
            
            # Convert to probability
            momentum_prob = 0.5 + np.tanh(adjusted_momentum * 10) * 0.3
            
            return max(0.1, min(0.9, momentum_prob))
            
        except Exception as e:
            logger.error(f"Momentum prediction error: {e}")
            return 0.5
    
    def _forecast_volatility(self, df: pd.DataFrame) -> float:
        """
        GARCH volatility forecasting
        """
        if not ARCH_AVAILABLE:
            # Fallback to simple rolling volatility
            returns = df['close'].pct_change().dropna()
            return returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0.02
        
        try:
            returns = df['close'].pct_change().dropna() * 100  # Convert to percentage
            
            if len(returns) < 50:
                return returns.std() / 100 if len(returns) > 0 else 0.02
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(returns, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            
            # Forecast next period volatility
            forecast = garch_fit.forecast(horizon=1)
            vol_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100
            
            return vol_forecast
            
        except Exception as e:
            logger.error(f"GARCH forecasting error: {e}")
            # Fallback to exponentially weighted moving average
            returns = df['close'].pct_change().dropna()
            ewm_vol = returns.ewm(span=30).std().iloc[-1] if len(returns) > 0 else 0.02
            return ewm_vol
    
    def _calculate_risk_metrics(self, df: pd.DataFrame, vol_forecast: float) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics
        """
        try:
            returns = df['close'].pct_change().dropna()
            
            metrics = {
                'volatility_forecast': vol_forecast,
                'historical_vol': returns.std() if len(returns) > 0 else 0.02,
                'var_95': np.percentile(returns, 5) if len(returns) > 20 else -0.05,
                'var_99': np.percentile(returns, 1) if len(returns) > 20 else -0.08,
                'max_drawdown': self._calculate_max_drawdown(df),
                'sharpe_estimate': self._estimate_sharpe(returns),
            }
            
            # Conditional VaR (Expected Shortfall)
            if len(returns) > 20:
                var_95 = metrics['var_95']
                tail_returns = returns[returns <= var_95]
                metrics['cvar_95'] = tail_returns.mean() if len(tail_returns) > 0 else var_95
            else:
                metrics['cvar_95'] = -0.07
            
            return metrics
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return {
                'volatility_forecast': vol_forecast,
                'historical_vol': 0.02,
                'var_95': -0.05,
                'cvar_95': -0.07,
                'max_drawdown': -0.10,
                'sharpe_estimate': 0.0
            }
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        try:
            prices = df['close']
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except:
            return -0.10
    
    def _estimate_sharpe(self, returns: pd.Series) -> float:
        """Estimate Sharpe ratio"""
        try:
            if len(returns) < 30:
                return 0.0
            
            excess_returns = returns - 0.02/252  # Assume 2% risk-free rate
            return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0
        except:
            return 0.0
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Simple market regime detection (bull/bear/sideways)
        """
        try:
            returns = df['close'].pct_change()
            sma_short = df['close'].rolling(20).mean()
            sma_long = df['close'].rolling(50).mean()
            
            # Simple regime classification
            regime = pd.Series(0, index=df.index)  # 0 = sideways
            regime[sma_short > sma_long] = 1  # 1 = bull
            regime[sma_short < sma_long] = -1  # -1 = bear
            
            return regime
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return pd.Series(0, index=df.index)
    
    def _generate_signal(self, symbol: str, df: pd.DataFrame, tech_score: float,
                        tech_reasoning: str, ml_predictions: Dict[str, float],
                        vol_forecast: float, risk_metrics: Dict[str, float]) -> CryptoSignal:
        """
        Generate comprehensive trading signal
        """
        try:
            # Combine all model predictions
            ensemble_score = ml_predictions.get('ensemble', 0.5)
            lstm_score = ml_predictions.get('lstm', 0.5)
            mean_reversion_score = ml_predictions.get('mean_reversion', 0.5)
            momentum_score = ml_predictions.get('momentum', 0.5)
            
            # Weight the ML predictions
            ml_score = (
                ensemble_score * 0.4 +
                lstm_score * 0.3 +
                momentum_score * 0.2 +
                mean_reversion_score * 0.1
            )
            
            # Overall score combining technical and ML
            overall_score = (
                tech_score * self.model_weights['technical'] +
                ml_score * (self.model_weights['lstm'] + self.model_weights['ensemble']) +
                (1 - vol_forecast * 5) * self.model_weights['garch']  # Lower vol = higher score
            )
            
            # Risk-adjusted confidence
            base_confidence = abs(overall_score - 0.5) * 2
            risk_adjustment = 1 - min(risk_metrics['volatility_forecast'] * 10, 0.5)
            confidence = base_confidence * risk_adjustment
            
            # Determine action
            if confidence < 0.3:
                action = "HOLD"
            elif overall_score >= 0.6:
                action = "BUY"
            elif overall_score <= 0.4:
                action = "SELL"
            else:
                action = "HOLD"
            
            # Calculate price targets and position sizing
            current_price = float(df['close'].iloc[-1])
            atr = self._calculate_atr(df)
            
            if action == "BUY":
                price_target = current_price * (1 + atr * 2 * confidence)
                stop_loss = current_price * (1 - atr * 1.5)
            elif action == "SELL":
                price_target = current_price * (1 - atr * 2 * confidence)
                stop_loss = current_price * (1 + atr * 1.5)
            else:
                price_target = None
                stop_loss = None
            
            # Position sizing using Kelly Criterion approximation
            position_size = self._calculate_position_size(
                confidence, risk_metrics['var_95'], vol_forecast
            )
            
            # Generate comprehensive reasoning
            reasoning = self._generate_reasoning(
                tech_reasoning, ml_predictions, vol_forecast, 
                risk_metrics, confidence, action
            )
            
            return CryptoSignal(
                symbol=symbol,
                action=action,
                confidence=round(confidence, 3),
                technical_score=round(tech_score, 3),
                ml_score=round(ml_score, 3),
                risk_score=round(vol_forecast, 3),
                volatility_forecast=round(vol_forecast, 4),
                price_target=round(price_target, 2) if price_target else None,
                stop_loss=round(stop_loss, 2) if stop_loss else None,
                position_size=round(position_size, 4),
                reasoning=reasoning,
                model_predictions={
                    'technical': round(tech_score, 3),
                    'ensemble_ml': round(ensemble_score, 3),
                    'lstm': round(lstm_score, 3),
                    'momentum': round(momentum_score, 3),
                    'mean_reversion': round(mean_reversion_score, 3),
                    'overall': round(overall_score, 3)
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return self._create_fallback_signal(symbol, f"Signal generation error: {str(e)}")
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr / df['close'].iloc[-1]  # Normalize by price
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.02  # Default 2%
    
    def _calculate_position_size(self, confidence: float, var_95: float, vol_forecast: float) -> float:
        """
        Calculate position size using modified Kelly Criterion
        """
        try:
            # Kelly fraction approximation
            win_prob = 0.5 + (confidence - 0.5) * 0.5
            avg_win = abs(var_95) * 1.5  # Expected win
            avg_loss = abs(var_95)  # Expected loss
            
            if avg_loss > 0:
                kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.02
            
            # Adjust for volatility
            vol_adjustment = 1 / (1 + vol_forecast * 10)
            adjusted_size = kelly_fraction * vol_adjustment
            
            # Conservative cap
            return min(adjusted_size, 0.05)  # Max 5% of portfolio
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 0.02  # Default 2%
    
    def _generate_reasoning(self, tech_reasoning: str, ml_predictions: Dict[str, float],
                          vol_forecast: float, risk_metrics: Dict[str, float],
                          confidence: float, action: str) -> str:
        """
        Generate comprehensive reasoning for the signal
        """
        try:
            reasoning_parts = [
                f"Technical Analysis: {tech_reasoning}",
                f"ML Ensemble: {ml_predictions.get('ensemble', 0.5):.3f}",
                f"LSTM Neural Net: {ml_predictions.get('lstm', 0.5):.3f}",
                f"Momentum Model: {ml_predictions.get('momentum', 0.5):.3f}",
                f"Mean Reversion: {ml_predictions.get('mean_reversion', 0.5):.3f}",
                f"GARCH Vol Forecast: {vol_forecast:.4f}",
                f"95% VaR: {risk_metrics.get('var_95', -0.05):.3f}",
                f"Max Drawdown: {risk_metrics.get('max_drawdown', -0.10):.3f}",
                f"Sharpe Estimate: {risk_metrics.get('sharpe_estimate', 0.0):.2f}",
                f"Final Confidence: {confidence:.3f}"
            ]
            
            recommendation = f"Recommendation: {action}"
            if action != 'HOLD':
                recommendation += f" with {confidence:.1%} confidence"
            
            reasoning_parts.append(recommendation)
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Reasoning generation error: {e}")
            return f"Analysis completed with {confidence:.1%} confidence -> {action}"
    
    def _create_fallback_signal(self, symbol: str, reason: str) -> CryptoSignal:
        """
        Create a fallback signal when analysis fails
        """
        return CryptoSignal(
            symbol=symbol,
            action="HOLD",
            confidence=0.3,
            technical_score=0.5,
            ml_score=0.5,
            risk_score=0.5,
            volatility_forecast=0.02,
            price_target=None,
            stop_loss=None,
            position_size=0.01,
            reasoning=reason,
            model_predictions={
                'technical': 0.5,
                'ensemble_ml': 0.5,
                'lstm': 0.5,
                'momentum': 0.5,
                'mean_reversion': 0.5,
                'overall': 0.5
            },
            timestamp=datetime.now()
        )
    
    def get_model_diagnostics(self) -> Dict[str, any]:
        """
        Get diagnostic information about the models
        """
        return {
            'tensorflow_available': TF_AVAILABLE,
            'arch_garch_available': ARCH_AVAILABLE,
            'pandas_ta_available': PANDAS_TA_AVAILABLE,
            'model_weights': self.model_weights,
            'min_data_points': self.min_data_points,
            'models_initialized': list(self.models.keys()),
            'scalers_initialized': list(self.scalers.keys())
        }

# Global instance
enhanced_crypto_analyzer = EnhancedCryptoAnalyzer()