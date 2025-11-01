import os
import requests
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from dotenv import load_dotenv
import threading
from flask import Flask
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class DeepLearningTrader:
    def __init__(self, starting_balance=100.0):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # REAL TRADING PARAMETERS
        self.paper_balance = starting_balance
        self.initial_balance = starting_balance
        self.trade_size = 20.0  # $20 per trade
        
        # Deep Learning Models
        self.price_predictor = None
        self.trend_classifier = None
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Trading Parameters
        self.min_confidence = 0.72  # 72% minimum confidence
        self.risk_reward = 2.0  # 1:2 risk:reward
        
        # Performance Tracking
        self.trade_history = []
        self.open_trades = {}
        self.performance = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'sharpe_ratio': 0.0
        }

        print(f"🧠 DEEP LEARNING TRADING BOT")
        print(f"💵 Starting Balance: ${self.paper_balance:.2f}")
        print(f"📊 Trade Size: ${self.trade_size:.2f}")
        print(f"🎯 AI Confidence: {self.min_confidence*100}% minimum")
        print(f"🚀 Real Deep Learning Models Active")

    def create_lstm_model(self, input_shape):
        """Create advanced LSTM model for price prediction"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def create_cnn_lstm_model(self, input_shape):
        """Create CNN-LSTM hybrid model for trend classification"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            LSTM(50, return_sequences=True),
            Dropout(0.3),
            LSTM(25),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def get_historical_data(self, coin_id, days=60):
        """Get extensive historical data for deep learning"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            prices = [point[1] for point in data['prices']]
            volumes = [point[1] for point in data['total_volumes']]
            
            return prices, volumes
        except Exception as e:
            print(f"❌ Historical data error: {e}")
            return [], []

    def calculate_technical_features(self, prices, volumes):
        """Calculate advanced technical indicators for deep learning"""
        if len(prices) < 50:
            return None
            
        df = pd.DataFrame({'price': prices, 'volume': volumes})
        
        # Price-based features
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['momentum'] = df['price'] / df['price'].shift(5) - 1
        
        # Moving averages
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price position
        df['price_vs_sma'] = df['price'] / df['sma_20'] - 1
        df['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Remove NaN values
        df = df.dropna()
        
        # Select features for model
        feature_columns = [
            'returns', 'volatility', 'momentum', 'rsi', 
            'volume_ratio', 'price_vs_sma', 'sma_cross',
            'macd', 'macd_signal', 'macd_histogram'
        ]
        
        return df[feature_columns].values

    def prepare_deep_learning_data(self, features, lookback=30):
        """Prepare data for deep learning models"""
        X, y = [], []
        
        for i in range(lookback, len(features)-1):
            X.append(features[i-lookback:i])
            # Create labels based on future price movement
            future_return = features[i+1, 0] 
            if future_return > 0.005:  # Strong uptrend
                y.append([1, 0, 0])  # BUY
            elif future_return < -0.003:  # Strong downtrend
                y.append([0, 1, 0])  # SELL
            else:
                y.append([0, 0, 1])  # HOLD
        
        return np.array(X), np.array(y)

    def train_deep_learning_models(self, coin_id):
        """Train deep learning models on historical data"""
        print(f"🧠 Training Deep Learning Models for {coin_id}...")
        
        try:
            prices, volumes = self.get_historical_data(coin_id, 90)
            if len(prices) < 100:
                print("❌ Not enough data for training.")
                return False
                
            # Calculate features
            features = self.calculate_technical_features(prices, volumes)
            if features is None:
                print("❌ Features calculation failed.")
                return False
                
            # Prepare data
            X, y = self.prepare_deep_learning_data(features)
            
            if len(X) < 50:
                print("❌ Not enough feature sequences for training.")
                return False
                
            # Split data
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Scale features
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            self.scaler.fit(X_train_reshaped)
            X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
            X_test_scaled = self.scaler.transform(X_test_reshaped).reshape(X_test.shape)
            
            # Create and train LSTM model
            self.price_predictor = self.create_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # Train model
            history = self.price_predictor.fit(
                X_train_scaled, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test_scaled, y_test),
                verbose=0,
                shuffle=False
            )
            
            # Evaluate model
            train_accuracy = history.history['accuracy'][-1]
            val_accuracy = history.history['val_accuracy'][-1]
            
            print(f"✅ Model Trained - Train Accuracy: {train_accuracy:.3f}, Val Accuracy: {val_accuracy:.3f}")
            self.model_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False

    def deep_learning_prediction(self, coin_data):
        """Make prediction using deep learning model"""
        try:
            coin_id = coin_data['id']
            current_price = coin_data['current_price']
            
            # Get recent data for prediction
            prices, volumes = self.get_historical_data(coin_id, 30)
            if len(prices) < 50:
                return None
                
            # Calculate features
            features = self.calculate_technical_features(prices, volumes)
            if features is None:
                return None
                
            # Use last sequence for prediction
            lookback = 30
            if len(features) < lookback:
                return None
                
            recent_features = features[-lookback:]
            recent_features_scaled = self.scaler.transform(recent_features)
            recent_features_reshaped = recent_features_scaled.reshape(1, lookback, -1)
            
            # Make prediction
            if self.price_predictor:
                prediction = self.price_predictor.predict(recent_features_reshaped, verbose=0)
                confidence = np.max(prediction[0])
                predicted_class = np.argmax(prediction[0])
                
                # Interpret prediction
                if predicted_class == 0 and confidence > self.min_confidence:  # BUY
                    signal_type = 'BUY'
                    reason = f"🧠 DL Prediction: BUY ({confidence*100:.1f}% confidence)"
                elif predicted_class == 1 and confidence > self.min_confidence:  # SELL
                    signal_type = 'SELL' 
                    reason = f"🧠 DL Prediction: SELL ({confidence*100:.1f}% confidence)"
                else:
                    return None
                    
                return {
                    'symbol': f"{coin_data['symbol'].upper()}USD",
                    'coin_id': coin_id,
                    'name': coin_data['name'],
                    'type': signal_type,
                    'confidence': confidence,
                    'price': current_price,
                    'change_24h': coin_data.get('price_change_percentage_24h', 0),
                    'volume': coin_data.get('total_volume', 0),
                    'reason': reason,
                    'model': 'LSTM',
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                
        except Exception as e:
            print(f"❌ DL prediction error: {e}")
            
        return None

    def get_deep_learning_signals(self):
        """Get signals from deep learning models"""
        print("🧠 Deep Learning AI Scanning Markets...")
        
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'volume_desc', 
                'per_page': 50,
                'page': 1,
                'sparkline': False
            }
            
            response = self.session.get(url, params=params)
            coins = response.json()
            
            signals = []
            for coin in coins:
                # Filter quality coins
                if (coin.get('market_cap', 0) > 1000000000 and
                    coin.get('total_volume', 0) > 50000000 and
                    5.0 < coin.get('current_price', 0) < 1000.0):
                    
                    # Train model if not trained
                    if not self.model_trained:
                        self.train_deep_learning_models(coin['id'])
                    
                    # Get deep learning prediction
                    signal = self.deep_learning_prediction(coin)
                    if signal and signal['confidence'] >= self.min_confidence:
                        signals.append(signal)
            
            return sorted(signals, key=lambda x: x['confidence'], reverse=True)[:3]
            
        except Exception as e:
            print(f"❌ DL signal error: {e}")
            return []

    def execute_dl_trade(self, signal):
        """Execute deep learning recommended trade"""
        if self.paper_balance < self.trade_size:
            print("❌ INSUFFICIENT BALANCE")
            return False
            
        symbol = signal['symbol']
        side = signal['type']
        current_price = signal['price']
        
        quantity = self.trade_size / current_price
        
        print(f"🧠 DEEP LEARNING TRADE EXECUTED:")
        print(f"   {side} {quantity:.6f} {symbol} @ ${current_price:.4f}")
        print(f"   Trade Value: ${self.trade_size:.2f}")
        print(f"   🤖 AI Confidence: {signal['confidence']*100:.1f}%")
        print(f"   🧠 Model: {signal['model']}")
        print(f"   💡 Signal: {signal['reason']}")
        
        # Deduct from balance
        self.paper_balance -= self.trade_size
        
        # Track trade
        trade_id = f"{symbol}{int(time.time())}"
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
            'stop_loss': current_price * (0.96 if side == 'BUY' else 1.04),
            'take_profit': current_price * (1.15 if side == 'BUY' else 0.85),
            'model_type': signal['model']
        }
        
        self.performance['total_trades'] += 1
        return True

    def update_dl_trades(self):
        """Update deep learning trades with smart exits"""
        if not self.open_trades:
            return
            
        print(f"\n📊 DEEP LEARNING TRADES:")
        print("-" * 70)
        
        for trade_id, trade in list(self.open_trades.items()):
            if trade['status'] == 'OPEN':
                try:
                    url = f"https://api.coingecko.com/api/v3/simple/price"
                    params = {'ids': trade['coin_id'], 'vs_currencies': 'usd'}
                    response = self.session.get(url, params=params)
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
                    print(f"   🤖 AI Confidence: {trade['ai_confidence']*100:.1f}%")
                    print(f"   Entry: ${entry_price:.4f} | Current: ${current_price:.4f}")
                    print(f"   P/L: {pl_percent:+.2f}% (${pl_usd:+.2f})")
                    
                    # Smart AI exits
                    if pl_percent >= 15.0:
                        print(f"   🎯 AI PROFIT: +{pl_percent:.1f}%")
                        self.close_dl_trade(trade_id)
                    elif pl_percent <= -4.0:
                        print(f"   🛑 AI STOP LOSS: {pl_percent:.1f}%")
                        self.close_dl_trade(trade_id)
                    elif time_in_trade > 20:
                        print(f"   ⏰ AI TIME EXIT: {time_in_trade:.0f}m")
                        self.close_dl_trade(trade_id)
                    else:
                        print(f"   🤖 AI HOLDING: {time_in_trade:.0f}m")
                    
                    print()
                    
                except Exception as e:
                    print(f"❌ Error updating {trade['symbol']}: {e}")

    def close_dl_trade(self, trade_id):
        """Close deep learning trade"""
        trade = self.open_trades[trade_id]
        final_pl = trade.get('pl_usd', 0)

        self.paper_balance += trade['trade_value'] + final_pl

        if final_pl > 0:
            trade['result'] = 'WIN'
            self.performance['wins'] += 1
            self.performance['total_profit'] += final_pl
        else:
            trade['result'] = 'LOSS'
            self.performance['losses'] += 1
            
        trade['status'] = 'CLOSED'
        trade['exit_time'] = datetime.now()
        trade['final_pl'] = final_pl
        
        # Update performance metrics
        total_trades = self.performance['wins'] + self.performance['losses']
        if total_trades > 0:
            self.performance['win_rate'] = self.performance['wins'] / total_trades
        
        self.trade_history.append(trade.copy())
        del self.open_trades[trade_id]
        
        result_icon = "💰 AI WIN" if final_pl > 0 else "💸 AI LOSS"
        print(f"   {result_icon} | P/L: ${final_pl:+.2f}")
        print(f"   📈 New Balance: ${self.paper_balance:.2f}")

    def display_dl_performance(self):
        """Display deep learning performance"""
        total_pl = self.paper_balance - self.initial_balance
        
        print(f"\n🧠 DEEP LEARNING PERFORMANCE:")
        print(f"   💰 Balance: ${self.paper_balance:.2f}")
        print(f"   📈 Total P/L: ${total_pl:+.2f}")
        print(f"   📊 Return: {(total_pl/self.initial_balance)*100:+.1f}%")
        print(f"   🤖 Win Rate: {self.performance['win_rate']*100:.1f}%")
        print(f"   📊 Total Trades: {self.performance['total_trades']}")
        print(f"   ✅ Wins: {self.performance['wins']} | ❌ Losses: {self.performance['losses']}")
        print(f"   💵 Total Profit: ${self.performance['total_profit']:.2f}")
        print(f"   🔥 Open Trades: {len(self.open_trades)}")
        print(f"   🧠 Model Status: {'TRAINED' if self.model_trained else 'TRAINING'}")

    def display_dl_signals(self, signals):
        """Display deep learning signals"""
        if not signals:
            print("❌ No high-confidence DL signals found")
            return
            
        print(f"\n🎯 DEEP LEARNING SIGNALS:")
        print("=" * 80)
        print(f"{'#':2} {'SYMBOL':12} {'TYPE':6} {'CONFIDENCE':10} {'MODEL':8} {'24H%':6}")
        print("-" * 80)
        
        for i, signal in enumerate(signals, 1):
            print(f"{i:2} {signal['symbol']:12} {signal['type']:6} {signal['confidence']*100:9.1f}% {signal['model']:8} {signal['change_24h']:5.1f}%")
            print(f"    🧠 {signal['reason']}")
            print(f"    💎 {signal['name']}")
            print()

# Flask app for Replit
app = Flask(__name__)
trader = None

@app.route('/')
def home():
    return """
    <html>
        <head><title>Deep Learning Trader</title></head>
        <body>
            <h1>🧠 Deep Learning Trading Bot</h1>
            <p>🤖 Real LSTM Neural Networks</p>
            <p>💵 $100 Demo Balance</p>
            <p>🎯 72%+ AI Confidence</p>
            <p><a href="/status">Live Status</a></p>
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
            "modeltrained": trader.model_trained
        }
    return {"status": "starting"}

def run_dl_trader():
    """Run the deep learning trader"""
    global trader
    trader = DeepLearningTrader(starting_balance=100.0)
    
    print("🚀 DEEP LEARNING TRADER STARTED")
    print("💵 $100 Demo Balance")
    print("🧠 Real LSTM Neural Networks")
    print("🎯 72% Minimum Confidence")
    print("🌐 Deployed on Replit")
    
    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\n{'='*70}")
            print(f"🔄 DL CYCLE {cycle} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*70}")
            
            # Update trades
            trader.update_dl_trades()
            
            # Get DL signals
            signals = trader.get_deep_learning_signals()
            trader.display_dl_signals(signals)
            
            # Execute trades
            if (signals and 
                trader.paper_balance >= trader.trade_size and 
                len(trader.open_trades) < 2):
                
                best_signal = signals[0]
                if best_signal['confidence'] >= trader.min_confidence:
                    print(f"🤖 EXECUTING DL TRADE: {best_signal['type']} {best_signal['symbol']}")
                    success = trader.execute_dl_trade(best_signal)
                    if success:
                        print("✅ DL TRADE EXECUTED!")
            
            # Display performance
            trader.display_dl_performance()
            
            print(f"\n⏰ Next DL analysis in 3 minutes...")
            time.sleep(180)
            
        except Exception as e:
            print(f"❌ DL cycle error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    # Start DL trader in background
    dl_thread = threading.Thread(target=run_dl_trader, daemon=True)
    dl_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)
