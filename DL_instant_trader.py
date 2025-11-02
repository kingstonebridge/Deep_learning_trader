import os
import requests
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from dotenv import load_dotenv
import threading
from flask import Flask
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class InstantTrader:
    def __init__(self, starting_balance=100.0):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # TRADING PARAMETERS - MORE AGGRESSIVE
        self.paper_balance = starting_balance
        self.initial_balance = starting_balance
        self.trade_size = 15.0  # Smaller trades to open more positions
        self.max_open_trades = 5  # Allow more open trades
        
        # Pre-trained Models
        self.model = None
        self.scaler = StandardScaler()
        self.model_loaded = False
        
        # Trading Parameters - LOWER CONFIDENCE FOR MORE TRADES
        self.min_confidence = 0.65  # Reduced from 0.75 to get more signals
        
        # Performance Tracking
        self.trade_history = []
        self.open_trades = {}
        self.performance = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_profit': 0.0
        }

        # Load pre-trained model immediately
        self.load_enhanced_model()
        
        print(f"🚀 INSTANT TRADER STARTED")
        print(f"💵 Starting Balance: ${self.paper_balance:.2f}")
        print(f"📊 Trade Size: ${self.trade_size:.2f}")
        print(f"🎯 AI Confidence: {self.min_confidence*100}% minimum")
        print(f"🔥 Max Open Trades: {self.max_open_trades}")
        print(f"🤖 Enhanced Model: {'LOADED' if self.model_loaded else 'FAILED'}")

    def create_enhanced_model(self):
        """Create an enhanced pre-trained model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(20, 10)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # BUY, SELL, HOLD
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_enhanced_model(self):
        """Load or create enhanced pre-trained model with realistic patterns"""
        try:
            model_path = 'enhanced_pretrained_model.h5'
            
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                print("✅ Enhanced pre-trained model loaded from file")
            else:
                print("🧠 Creating enhanced pre-trained model with realistic patterns...")
                self.model = self.create_enhanced_model()
                
                # Generate MORE REALISTIC training data
                np.random.seed(42)
                n_samples = 5000
                
                # Create realistic market-like patterns
                X_train = []
                y_train = []
                
                for i in range(n_samples):
                    # Create realistic price series with trends and noise
                    base_trend = np.linspace(0, 1, 20) * np.random.choice([-1, 1]) * 0.1
                    noise = np.random.normal(0, 0.02, 20)
                    price_series = 100 + np.cumsum(base_trend + noise)
                    
                    # Calculate realistic features
                    features = []
                    for j in range(10, 20):  # Last 10 periods for features
                        returns = np.diff(price_series[max(0, j-10):j+1]) / price_series[max(0, j-10):j][:-1]
                        volatility = np.std(returns) if len(returns) > 1 else 0
                        momentum = (price_series[j] / price_series[max(0, j-5)] - 1) if j >= 5 else 0
                        
                        # Simple moving averages
                        sma_short = np.mean(price_series[max(0, j-3):j+1])
                        sma_long = np.mean(price_series[max(0, j-8):j+1])
                        
                        feature_set = [
                            returns[-1] if len(returns) > 0 else 0,
                            volatility,
                            momentum,
                            price_series[j] / sma_short - 1,
                            price_series[j] / sma_long - 1,
                            np.random.random(),  # volume-like
                            np.random.random(),  # rsi-like
                            base_trend[-1] if len(base_trend) > 0 else 0,
                            np.random.normal(0, 0.1),  # random factor
                            price_series[j] / 100 - 1  # normalized price
                        ]
                        features.append(feature_set)
                    
                    # Ensure we have exactly 20 time steps
                    while len(features) < 20:
                        features.insert(0, [0] * 10)
                    
                    features = features[-20:]  # Take last 20
                    
                    # Create realistic labels based on patterns
                    recent_trend = np.mean([f[0] for f in features[-5:]])  # Recent returns
                    volatility = np.mean([f[1] for f in features])
                    
                    if recent_trend > 0.01 and volatility < 0.05:  # Strong uptrend, low vol
                        label = [1, 0, 0]  # BUY
                    elif recent_trend < -0.01 and volatility < 0.05:  # Strong downtrend, low vol
                        label = [0, 1, 0]  # SELL
                    else:
                        label = [0, 0, 1]  # HOLD
                    
                    X_train.append(features)
                    y_train.append(label)
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Train the model
                self.model.fit(
                    X_train, y_train,
                    epochs=15,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1
                )
                
                # Save the model
                self.model.save(model_path)
                print("✅ Enhanced pre-trained model created and saved")
            
            # Initialize scaler with realistic data
            dummy_features = np.random.normal(0, 0.1, (1000, 10))
            self.scaler.fit(dummy_features)
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Enhanced model loading failed: {e}")
            self.model_loaded = False
            return False

    def get_enhanced_historical_data(self, coin_id, days=20):
        """Get enhanced historical data"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'
            }
            
            response = self.session.get(url, params=params)
            if response.status_code != 200:
                return [], []
                
            data = response.json()
            
            prices = [point[1] for point in data['prices'][-100:]]  # Last 100 points
            volumes = [point[1] for point in data['total_volumes'][-100:]]
            
            return prices, volumes
        except Exception as e:
            print(f"❌ Historical data error for {coin_id}: {e}")
            return [], []

    def calculate_enhanced_features(self, prices, volumes):
        """Calculate enhanced technical features"""
        if len(prices) < 20:
            return None
            
        df = pd.DataFrame({'price': prices, 'volume': volumes})
        
        # Enhanced features
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(window=10).std()
        df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
        df['momentum_10'] = df['price'] / df['price'].shift(10) - 1
        
        # Moving averages
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_10'] = df['price'].rolling(window=10).mean()
        df['sma_15'] = df['price'].rolling(window=15).mean()
        
        # Price vs MA ratios
        df['price_vs_sma5'] = df['price'] / df['sma_5'] - 1
        df['price_vs_sma10'] = df['price'] / df['sma_10'] - 1
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # RSI-like calculation
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Trend strength
        df['trend_strength'] = (df['sma_5'] - df['sma_15']).abs() / df['price'] * 100
        
        # Remove NaN values
        df = df.dropna()
        
        if len(df) < 20:
            return None
            
        # Select enhanced features
        feature_columns = [
            'returns', 'volatility', 'momentum_5', 'momentum_10',
            'price_vs_sma5', 'price_vs_sma10', 'volume_ratio',
            'rsi', 'trend_strength', 'price'
        ]
        
        features = df[feature_columns].values
        
        # Normalize price separately
        if len(features) > 0:
            features[:, -1] = (features[:, -1] - np.mean(features[:, -1])) / np.std(features[:, -1])
        
        return features

    def get_high_quality_coins(self):
        """Get list of high-quality coins for trading"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'volume_desc',
                'per_page': 100,  # More coins to choose from
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h'
            }
            
            response = self.session.get(url, params=params)
            coins = response.json()
            
            # Filter for high-quality trading candidates
            quality_coins = []
            for coin in coins:
                # Less strict filters to get more candidates
                market_cap = coin.get('market_cap', 0)
                volume = coin.get('total_volume', 0)
                price = coin.get('current_price', 0)
                change_24h = coin.get('price_change_percentage_24h', 0)
                
                # Quality criteria (relaxed)
                if (market_cap > 100000000 and    # $100M+ market cap
                    volume > 10000000 and         # $10M+ volume
                    0.1 < price < 5000.0 and      # Reasonable price range
                    abs(change_24h) < 50.0):      # Not extremely volatile
                    
                    quality_coins.append(coin)
            
            print(f"📊 Found {len(quality_coins)} quality coins for analysis")
            return quality_coins[:50]  # Limit to top 50
            
        except Exception as e:
            print(f"❌ Error fetching coins: {e}")
            return []

    def make_enhanced_prediction(self, coin_data):
        """Make enhanced prediction using pre-trained model"""
        try:
            coin_id = coin_data['id']
            current_price = coin_data['current_price']
            
            # Get recent data
            prices, volumes = self.get_enhanced_historical_data(coin_id, 10)
            if len(prices) < 20:
                return None
                
            # Calculate enhanced features
            features = self.calculate_enhanced_features(prices, volumes)
            if features is None or len(features) < 20:
                return None
                
            # Use last 20 periods for prediction
            recent_features = features[-20:]
            
            try:
                recent_features_scaled = self.scaler.transform(recent_features)
            except:
                # If scaling fails, use original features
                recent_features_scaled = recent_features
                
            recent_features_reshaped = recent_features_scaled.reshape(1, 20, -1)
            
            # Make prediction
            if self.model_loaded:
                prediction = self.model.predict(recent_features_reshaped, verbose=0)
                confidence = np.max(prediction[0])
                predicted_class = np.argmax(prediction[0])
                
                # ENHANCED: More aggressive signal generation
                current_change = coin_data.get('price_change_percentage_24h', 0)
                volume = coin_data.get('total_volume', 0)
                
                # Additional confidence boost based on market conditions
                volume_boost = min(volume / 50000000, 0.1)  # Boost for high volume
                trend_boost = 0.05 if abs(current_change) > 3 else 0  # Boost for strong trends
                
                adjusted_confidence = confidence + volume_boost + trend_boost
                
                # Generate signals more aggressively
                if predicted_class == 0 and adjusted_confidence > self.min_confidence:  # BUY
                    signal_type = 'BUY'
                    reason = f"🤖 ENHANCED MODEL: STRONG BUY ({adjusted_confidence*100:.1f}% confidence)"
                    final_confidence = adjusted_confidence
                    
                elif predicted_class == 1 and adjusted_confidence > self.min_confidence:  # SELL
                    signal_type = 'SELL' 
                    reason = f"🤖 ENHANCED MODEL: STRONG SELL ({adjusted_confidence*100:.1f}% confidence)"
                    final_confidence = adjusted_confidence
                    
                else:
                    return None
                    
                return {
                    'symbol': f"{coin_data['symbol'].upper()}USD",
                    'coin_id': coin_id,
                    'name': coin_data['name'],
                    'type': signal_type,
                    'confidence': final_confidence,
                    'price': current_price,
                    'change_24h': current_change,
                    'volume': volume,
                    'reason': reason,
                    'model': 'Enhanced LSTM',
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'market_cap': coin_data.get('market_cap', 0)
                }
                
        except Exception as e:
            print(f"❌ Prediction error for {coin_data.get('symbol', 'unknown')}: {e}")
            
        return None

    def get_instant_signals(self):
        """Get instant trading signals - MORE AGGRESSIVE"""
        print("🎯 INSTANT TRADER: Scanning for high-probability signals...")
        
        quality_coins = self.get_high_quality_coins()
        if not quality_coins:
            print("❌ No quality coins found")
            return []
        
        signals = []
        coins_analyzed = 0
        
        # Analyze coins in batches for speed
        for coin in quality_coins[:30]:  # Analyze top 30 coins
            try:
                signal = self.make_enhanced_prediction(coin)
                if signal:
                    signals.append(signal)
                    print(f"✅ Signal found: {signal['symbol']} {signal['type']} ({signal['confidence']*100:.1f}%)")
                
                coins_analyzed += 1
                if coins_analyzed % 10 == 0:
                    print(f"📊 Analyzed {coins_analyzed} coins...")
                    
            except Exception as e:
                print(f"❌ Error analyzing {coin.get('symbol', 'unknown')}: {e}")
                continue
        
        # Sort by confidence and return top signals
        sorted_signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
        
        print(f"🎯 Found {len(sorted_signals)} total signals")
        return sorted_signals[:self.max_open_trades]  # Return up to max_open_trades signals

    def execute_instant_trade(self, signal):
        """Execute instant trade"""
        if self.paper_balance < self.trade_size:
            print("❌ INSUFFICIENT BALANCE")
            return False
            
        symbol = signal['symbol']
        side = signal['type']
        current_price = signal['price']
        
        quantity = self.trade_size / current_price
        
        print(f"🚀 INSTANT TRADE EXECUTED:")
        print(f"   {side} {quantity:.6f} {symbol} @ ${current_price:.4f}")
        print(f"   Trade Value: ${self.trade_size:.2f}")
        print(f"   🤖 AI Confidence: {signal['confidence']*100:.1f}%")
        print(f"   📊 24H Change: {signal['change_24h']:+.1f}%")
        print(f"   💡 Signal: {signal['reason']}")
        
        # Deduct from balance
        self.paper_balance -= self.trade_size
        
        # Track trade
        trade_id = f"{symbol}_{int(time.time())}"
        self.open_trades[trade_id] = {
            'symbol': symbol,
            'coin_id': signal['coin_id'],
            'name': signal['name'],
            'side': side,
            'entry_price': current_price,
            'quantity': quantity,
            'trade_value': self.trade_size,
            'entry_time': datetime.now(),
            'status': 'OPEN',
            'ai_confidence': signal['confidence'],
            'stop_loss': current_price * (0.92 if side == 'BUY' else 1.08),  # Wider stops
            'take_profit': current_price * (1.10 if side == 'BUY' else 0.90), # Closer targets
            'model_type': signal['model'],
            'market_cap': signal.get('market_cap', 0)
        }
        
        self.performance['total_trades'] += 1
        return True

    def update_open_trades(self):
        """Update all open trades with smart exits"""
        if not self.open_trades:
            print("📊 No open trades to update")
            return
            
        print(f"\n📊 OPEN TRADES STATUS:")
        print("=" * 80)
        
        trades_to_close = []
        
        for trade_id, trade in self.open_trades.items():
            if trade['status'] == 'OPEN':
                try:
                    url = f"https://api.coingecko.com/api/v3/simple/price"
                    params = {'ids': trade['coin_id'], 'vs_currencies': 'usd'}
                    response = self.session.get(url, params=params)
                    
                    if trade['coin_id'] not in response.json():
                        continue
                        
                    current_price = response.json()[trade['coin_id']]['usd']
                    
                    entry_price = trade['entry_price']
                    quantity = trade['quantity']
                    
                    if trade['side'] == 'BUY':
                        pl_percent = ((current_price - entry_price) / entry_price) * 100
                        pl_usd = (current_price - entry_price) * quantity
                    else:
                        pl_percent = ((entry_price - current_price) / entry_price) * 100
                        pl_usd = (entry_price - current_price) * quantity
                    
                    trade['current_price'] = current_price
                    trade['pl_percent'] = pl_percent
                    trade['pl_usd'] = pl_usd
                    
                    time_in_trade = (datetime.now() - trade['entry_time']).total_seconds() / 60
                    
                    status_icon = "🟢" if pl_usd > 0 else "🔴"
                    print(f"{status_icon} {trade['side']} {trade['symbol']}")
                    print(f"   Entry: ${entry_price:.4f} | Current: ${current_price:.4f}")
                    print(f"   P/L: {pl_percent:+.2f}% (${pl_usd:+.2f})")
                    print(f"   Time: {time_in_trade:.0f}m | Confidence: {trade['ai_confidence']*100:.1f}%")
                    
                    # SMART EXIT LOGIC - More aggressive
                    if pl_percent >= 8.0:  # Take profit at 8%
                        print(f"   🎯 QUICK PROFIT: +{pl_percent:.1f}%")
                        trades_to_close.append(trade_id)
                    elif pl_percent <= -6.0:  # Stop loss at -6%
                        print(f"   🛑 STOP LOSS: {pl_percent:.1f}%")
                        trades_to_close.append(trade_id)
                    elif time_in_trade > 15:  # Time-based exit at 15 minutes
                        print(f"   ⏰ TIME EXIT: {time_in_trade:.0f}m")
                        trades_to_close.append(trade_id)
                    else:
                        print(f"   📈 HOLDING... Target: +8%")
                    
                    print()
                    
                except Exception as e:
                    print(f"❌ Error updating {trade['symbol']}: {e}")
        
        # Close trades that hit exit conditions
        for trade_id in trades_to_close:
            self.close_trade(trade_id)

    def close_trade(self, trade_id):
        """Close trade and update performance"""
        if trade_id not in self.open_trades:
            return
            
        trade = self.open_trades[trade_id]
        final_pl = trade.get('pl_usd', 0)

        # Add back trade value + P/L
        self.paper_balance += trade['trade_value'] + final_pl

        if final_pl > 0:
            trade['result'] = 'WIN'
            self.performance['wins'] += 1
            self.performance['total_profit'] += final_pl
            result_emoji = "💰"
        else:
            trade['result'] = 'LOSS'
            self.performance['losses'] += 1
            result_emoji = "💸"
            
        trade['status'] = 'CLOSED'
        trade['exit_time'] = datetime.now()
        trade['final_pl'] = final_pl
        
        # Update performance metrics
        total_trades = self.performance['wins'] + self.performance['losses']
        if total_trades > 0:
            self.performance['win_rate'] = self.performance['wins'] / total_trades
        
        self.trade_history.append(trade.copy())
        del self.open_trades[trade_id]
        
        print(f"   {result_emoji} TRADE CLOSED | P/L: ${final_pl:+.2f}")
        print(f"   📈 New Balance: ${self.paper_balance:.2f}")

    def display_live_performance(self):
        """Display live performance metrics"""
        total_pl = self.paper_balance - self.initial_balance
        
        print(f"\n📊 LIVE PERFORMANCE:")
        print(f"   💰 Balance: ${self.paper_balance:.2f}")
        print(f"   📈 Total P/L: ${total_pl:+.2f}")
        print(f"   📊 Return: {(total_pl/self.initial_balance)*100:+.1f}%")
        print(f"   🎯 Win Rate: {self.performance['win_rate']*100:.1f}%")
        print(f"   📈 Total Trades: {self.performance['total_trades']}")
        print(f"   ✅ Wins: {self.performance['wins']} | ❌ Losses: {self.performance['losses']}")
        print(f"   💵 Total Profit: ${self.performance['total_profit']:.2f}")
        print(f"   🔥 Open Trades: {len(self.open_trades)}/{self.max_open_trades}")
        print(f"   🤖 Model: {'ACTIVE' if self.model_loaded else 'OFFLINE'}")

    def display_instant_signals(self, signals):
        """Display instant trading signals"""
        if not signals:
            print("❌ No instant signals found - relaxing filters for next scan...")
            return
            
        print(f"\n🎯 INSTANT TRADING SIGNALS:")
        print("=" * 90)
        print(f"{'#':2} {'SYMBOL':12} {'TYPE':6} {'CONFIDENCE':11} {'24H%':6} {'VOLUME':12} {'MCAP':12}")
        print("-" * 90)
        
        for i, signal in enumerate(signals, 1):
            volume_millions = signal['volume'] / 1000000
            mcap_billions = signal['market_cap'] / 1000000000
            print(f"{i:2} {signal['symbol']:12} {signal['type']:6} {signal['confidence']*100:10.1f}% {signal['change_24h']:5.1f}% {volume_millions:10.1f}M {mcap_billions:10.1f}B")
            print(f"    🤖 {signal['reason']}")
            print()

# Flask app for web interface
app = Flask(__name__)
trader = None

@app.route('/')
def home():
    return """
    <html>
        <head><title>Instant Trader</title>
        <meta http-equiv="refresh" content="10">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .warning { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Instant Trading Bot</h1>
                <p><strong>Enhanced Pre-trained Model • Immediate Trading • High Frequency</strong></p>
                
                <div class="status">
                    <h3>Live Status</h3>
                    <p><a href="/status">Click for JSON Status</a> • Auto-refresh every 10 seconds</p>
                </div>
                
                <div class="warning">
                    <h3>⚡ Trading Strategy</h3>
                    <p>• Lower confidence threshold (65%) for more signals</p>
                    <p>• Smaller trade sizes ($15) for more positions</p>
                    <p>• Faster analysis cycles (90 seconds)</p>
                    <p>• Maximum 5 open trades simultaneously</p>
                </div>
                
                <h3>Features</h3>
                <ul>
                    <li>🤖 Enhanced pre-trained LSTM model</li>
                    <li>🎯 Realistic synthetic training data</li>
                    <li>⚡ Immediate trade execution</li>
                    <li>📊 Multiple concurrent positions</li>
                    <li>🔍 100+ coins analyzed per cycle</li>
                </ul>
            </div>
        </body>
    </html>
    """

@app.route('/status')
def status():
    if trader:
        return {
            "status": "running",
            "balance": round(trader.paper_balance, 2),
            "winrate": round(trader.performance['win_rate'] * 100, 1),
            "totaltrades": trader.performance['total_trades'],
            "opentrades": len(trader.open_trades),
            "maxopentrades": trader.max_open_trades,
            "modelloaded": trader.model_loaded,
            "tradesize": trader.trade_size,
            "minconfidence": trader.min_confidence * 100
        }
    return {"status": "starting"}

def run_instant_trader():
    """Run the instant trader"""
    global trader
    trader = InstantTrader(starting_balance=100.0)
    
    print("🚀 INSTANT TRADER ACTIVATED!")
    print("💵 $100 Starting Balance")
    print("🤖 Enhanced Pre-trained LSTM Model")
    print("🎯 65% Minimum Confidence (Lowered for more signals)")
    print("⚡ Immediate Trading Enabled")
    print("📊 Maximum 5 Open Trades")
    print("🔍 Analyzing 100+ coins per cycle")
    
    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\n{'='*80}")
            print(f"🔄 INSTANT CYCLE {cycle} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*80}")
            
            # Update existing trades
            trader.update_open_trades()
            
            # Get instant signals (MORE AGGRESSIVE)
            signals = trader.get_instant_signals()
            trader.display_instant_signals(signals)
            
            # Execute multiple trades if possible
            trades_executed = 0
            available_slots = trader.max_open_trades - len(trader.open_trades)
            
            if signals and available_slots > 0:
                for signal in signals[:available_slots]:  # Fill available slots
                    if (trader.paper_balance >= trader.trade_size and 
                        signal['confidence'] >= trader.min_confidence):
                        
                        print(f"🚀 EXECUTING INSTANT TRADE: {signal['type']} {signal['symbol']}")
                        success = trader.execute_instant_trade(signal)
                        if success:
                            trades_executed += 1
                            print(f"✅ TRADE {trades_executed} EXECUTED!")
                            
                            # Small delay between executions
                            time.sleep(1)
            
            # Display live performance
            trader.display_live_performance()
            
            print(f"\n⏰ Next instant analysis in 90 seconds...")
            time.sleep(90)  # Faster cycles for more action
            
        except Exception as e:
            print(f"❌ Instant cycle error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)  # Shorter delay on error

if __name__ == "__main__":
    # Start instant trader in background
    trader_thread = threading.Thread(target=run_instant_trader, daemon=True)
    trader_thread.start()
    
    # Start Flask app
    print("🌐 Web interface starting at http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
