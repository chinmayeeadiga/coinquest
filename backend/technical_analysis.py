# technical_analysis.py - Technical Analysis Engine
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Technical analysis using various indicators"""
    
    def __init__(self):
        self.indicators = {
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger_bands': self._calculate_bollinger_bands,
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'stochastic': self._calculate_stochastic
        }
    
    def analyze(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Comprehensive technical analysis"""
        if df.empty or len(df) < 20:
            return 0.5, "Insufficient data for technical analysis"
        
        try:
            scores = []
            reasoning_parts = []
            
            # Calculate all indicators
            rsi_score, rsi_reason = self._analyze_rsi(df)
            macd_score, macd_reason = self._analyze_macd(df)
            bb_score, bb_reason = self._analyze_bollinger_bands(df)
            trend_score, trend_reason = self._analyze_trend(df)
            volume_score, volume_reason = self._analyze_volume(df)
            
            scores.extend([rsi_score, macd_score, bb_score, trend_score, volume_score])
            reasoning_parts.extend([rsi_reason, macd_reason, bb_reason, trend_reason, volume_reason])
            
            # Calculate weighted average score
            weights = [0.2, 0.2, 0.2, 0.25, 0.15]  # RSI, MACD, BB, Trend, Volume
            final_score = sum(score * weight for score, weight in zip(scores, weights))
            
            # Generate final reasoning
            final_reasoning = " | ".join([r for r in reasoning_parts if r])
            
            return final_score, final_reasoning
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return 0.5, f"Technical analysis incomplete: {str(e)}"
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _analyze_rsi(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Analyze RSI indicator"""
        rsi = self._calculate_rsi(df)
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < 30:
            return 0.8, "RSI indicates oversold conditions (bullish)"
        elif current_rsi > 70:
            return 0.2, "RSI indicates overbought conditions (bearish)"
        elif 30 <= current_rsi <= 50:
            return 0.6, "RSI in lower neutral range (slightly bullish)"
        elif 50 < current_rsi <= 70:
            return 0.4, "RSI in upper neutral range (slightly bearish)"
        else:
            return 0.5, "RSI in neutral range"
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return {'macd': macd, 'signal': signal, 'histogram': histogram}
    
    def _analyze_macd(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Analyze MACD indicator"""
        macd_data = self._calculate_macd(df)
        current_macd = macd_data['macd'].iloc[-1]
        current_signal = macd_data['signal'].iloc[-1]
        current_histogram = macd_data['histogram'].iloc[-1]
        prev_histogram = macd_data['histogram'].iloc[-2] if len(macd_data['histogram']) > 1 else 0
        
        if current_macd > current_signal and current_histogram > prev_histogram:
            return 0.8, "MACD bullish crossover with increasing momentum"
        elif current_macd < current_signal and current_histogram < prev_histogram:
            return 0.2, "MACD bearish crossover with decreasing momentum"
        elif current_macd > current_signal:
            return 0.6, "MACD above signal line (bullish)"
        elif current_macd < current_signal:
            return 0.4, "MACD below signal line (bearish)"
        else:
            return 0.5, "MACD neutral"
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        return {'sma': sma, 'upper_band': upper_band, 'lower_band': lower_band}
    
    def _analyze_bollinger_bands(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Analyze Bollinger Bands"""
        bb_data = self._calculate_bollinger_bands(df)
        current_price = df['close'].iloc[-1]
        current_lower = bb_data['lower_band'].iloc[-1]
        current_upper = bb_data['upper_band'].iloc[-1]
        current_sma = bb_data['sma'].iloc[-1]
        
        band_width = (current_upper - current_lower) / current_sma
        
        if current_price <= current_lower:
            return 0.8, "Price at or below lower Bollinger Band (bullish)"
        elif current_price >= current_upper:
            return 0.2, "Price at or above upper Bollinger Band (bearish)"
        elif band_width > 0.1:  # High volatility
            if current_price > current_sma:
                return 0.6, "High volatility, price above SMA (bullish)"
            else:
                return 0.4, "High volatility, price below SMA (bearish)"
        else:
            return 0.5, "Price within normal Bollinger Band range"
    
    def _analyze_trend(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Analyze price trend using multiple moving averages"""
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        ema_12 = df['close'].ewm(span=12).mean()
        
        current_price = df['close'].iloc[-1]
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        current_ema_12 = ema_12.iloc[-1]
        
        # Check alignment of moving averages
        bullish_alignment = current_price > current_ema_12 > current_sma_20 > current_sma_50
        bearish_alignment = current_price < current_ema_12 < current_sma_20 < current_sma_50
        
        if bullish_alignment:
            return 0.9, "Strong bullish trend with aligned moving averages"
        elif bearish_alignment:
            return 0.1, "Strong bearish trend with aligned moving averages"
        elif current_price > current_sma_20 and current_sma_20 > current_sma_50:
            return 0.7, "Moderate bullish trend"
        elif current_price < current_sma_20 and current_sma_20 < current_sma_50:
            return 0.3, "Moderate bearish trend"
        else:
            return 0.5, "No clear trend"
    
    def _analyze_volume(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Analyze volume trends"""
        if 'volume' not in df.columns:
            return 0.5, "Volume data not available"
        
        volume_sma = df['volume'].rolling(window=20).mean()
        current_volume = df['volume'].iloc[-1]
        avg_volume = volume_sma.iloc[-1]
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] if len(df) > 1 else 0
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 1.5 and price_change > 0:
            return 0.8, "High volume with price increase (bullish)"
        elif volume_ratio > 1.5 and price_change < 0:
            return 0.2, "High volume with price decrease (bearish)"
        elif volume_ratio > 1.2 and price_change > 0:
            return 0.6, "Above average volume with price increase"
        elif volume_ratio < 0.8:
            return 0.4, "Low volume indicates weak momentum"
        else:
            return 0.5, "Normal volume conditions"
    
    def _calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return df['close'].rolling(window=period).mean()
    
    def _calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=period).mean()
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator"""
        low_14 = df['low'].rolling(window=period).min()
        high_14 = df['high'].rolling(window=period).max()
        stoch = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        return stoch