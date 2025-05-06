import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_data(file_path):
    """Load and prepare the data"""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def identify_pollution_hotspots(df):
    """Identify pollution hotspots using DBSCAN clustering"""
    try:
        X = df[['x_coord', 'y_coord', 'pm25']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        clusters = dbscan.fit_predict(X_scaled)
        df['cluster'] = clusters
        return df
    except Exception as e:
        print(f"Error in clustering: {e}")
        return df

def calculate_pollution_risk(df):
    """Calculate pollution risk index for each cluster"""
    try:
        risk_by_cluster = {}
        for cluster in df['cluster'].unique():
            if cluster != -1:  # Exclude noise points
                cluster_data = df[df['cluster'] == cluster]
                avg_pm25 = cluster_data['pm25'].mean()
                exceed_threshold = (cluster_data['pm25'] > 50).mean()
                risk_index = (avg_pm25 * 0.7) + (exceed_threshold * 0.3)
                risk_by_cluster[cluster] = risk_index
        return risk_by_cluster
    except Exception as e:
        print(f"Error calculating risk: {e}")
        return {}

def calculate_weather_index(df):
    """Calculate adverse weather index"""
    try:
        temp_norm = (df['temperature'] - df['temperature'].min()) / (df['temperature'].max() - df['temperature'].min())
        humidity_norm = df['humidity'] / 100
        wind_norm = 1 - (df['wind_speed'] / df['wind_speed'].max())
        
        weather_index = (0.3 * temp_norm + 0.4 * humidity_norm + 0.3 * wind_norm)
        return weather_index
    except Exception as e:
        print(f"Error calculating weather index: {e}")
        return None

def calculate_pollution_trend(df):
    """Calculate pollution trend"""
    try:
        df = df.sort_values('timestamp')
        df['pollution_trend'] = df.groupby(['x_coord', 'y_coord'])['pm25'].transform(
            lambda x: np.polyfit(range(len(x[-24:])), x[-24:], 1)[0] if len(x) >= 24 else np.nan
        )
        return df
    except Exception as e:
        print(f"Error calculating trend: {e}")
        return df

def prepare_lstm_data(df, lookback=24, forecast=6):
    """Prepare data for LSTM model"""
    try:
        data = df['pm25'].values
        X, y = [], []
        for i in range(len(data) - lookback - forecast + 1):
            X.append(data[i:(i + lookback)])
            y.append(data[(i + lookback):(i + lookback + forecast)])
        return np.array(X), np.array(y)
    except Exception as e:
        print(f"Error preparing LSTM data: {e}")
        return None, None

def build_lstm_model(lookback):
    """Build and compile LSTM model"""
    try:
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1)),
            Dense(6)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        return None

def main():
    # Load data
    file_path = r'E:\HOC\Lập trình Python\baitapvenha\bai_thuc_hanh_5_5\bai2\Data_Number_4.csv'
    df = load_data(file_path)
    if df is None:
        return
    
    # Process data
    df = identify_pollution_hotspots(df)
    risk_indices = calculate_pollution_risk(df)
    df['weather_index'] = calculate_weather_index(df)
    df = calculate_pollution_trend(df)
    
    # Prepare LSTM data
    X, y = prepare_lstm_data(df)
    if X is None or y is None:
        return
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    model = build_lstm_model(24)
    if model is None:
        return
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_test.flatten(), predictions.flatten()))
    print(f'RMSE: {rmse}')
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[0], label='Actual')
    plt.plot(predictions[0], label='Predicted')
    plt.title('PM2.5 Prediction vs Actual')
    plt.xlabel('Hours ahead')
    plt.ylabel('PM2.5 concentration')
    plt.legend()
    plt.savefig('pm25_prediction.png')  # Save the plot instead of showing it
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    main()
