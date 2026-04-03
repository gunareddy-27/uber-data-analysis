import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from sklearn.ensemble import IsolationForest
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

ANOMALY_MODEL_PATH = 'models/anomaly_model.pkl'
FORECAST_MODEL_PATH = 'models/forecast_model.pkl'

MODEL_PATH = 'models/trip_duration_model.pkl'
SCALER_PATH = 'models/trip_duration_scaler.pkl'
DEMAND_MODEL_PATH = 'models/demand_model.pkl'
CATEGORY_MODEL_PATH = 'models/category_model.pkl'
PURPOSE_MODEL_PATH = 'models/purpose_model.pkl'
CATEGORY_ENCODER_PATH = 'models/category_encoder.pkl'
PURPOSE_ENCODER_PATH = 'models/purpose_encoder.pkl'
LOCATION_MODEL_PATH = 'models/location_model.pkl'
STOP_ENCODER_PATH = 'models/stop_encoder.pkl'
LOC_CAT_ENCODER_PATH = 'models/loc_cat_encoder.pkl'
LOC_PURP_ENCODER_PATH = 'models/loc_purp_encoder.pkl'

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # Parse dates
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')
    df['END_DATE'] = pd.to_datetime(df['END_DATE'], errors='coerce')

    # Drop rows with missing values in critical columns
    # Note: Dataset does not have LAT/LON, so we use MILES as a proxy for distance/location info
    df.dropna(subset=['START_DATE', 'END_DATE', 'MILES'], inplace=True)

    # Feature Engineering
    df['Hour'] = df['START_DATE'].dt.hour
    df['Weekday'] = df['START_DATE'].dt.dayofweek
    df['Month'] = df['START_DATE'].dt.month
    df['Trip_Duration'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 60

    # Remove outliers (e.g., more than 5 hours)
    df = df[df['Trip_Duration'] < 300]

    return df

def train_trip_duration_model(df):
    print("Training Trip Duration Model...")
    # Use MILES instead of LAT/LON
    df_model = df[['MILES', 'Hour', 'Weekday', 'Trip_Duration']].dropna()
    
    X = df_model[['MILES', 'Hour', 'Weekday']]
    y = df_model['Trip_Duration']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict and Evaluate
    y_pred = model.predict(X_test_scaled)
    print(f"Trip Duration Model - MAE: {mean_absolute_error(y_test, y_pred):.2f} minutes")
    print(f"Trip Duration Model - R²: {r2_score(y_test, y_pred):.2f}")

    # Save models
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    return model, scaler

def train_demand_model(df):
    print("Training Demand Prediction Model...")
    
    # 1. Regression Model (Total hourly/daily demand)
    df_demand = df.groupby(['Hour', 'Weekday']).size().reset_index(name='Trip_Count')
    X = df_demand[['Hour', 'Weekday']]
    y = df_demand['Trip_Count']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 2. Location Intelligence: Find most frequent START for each (Hour, Weekday)
    # This acts as our "Revenue Optimizer" lookup table
    loc_demand = df.groupby(['Hour', 'Weekday', 'START']).size().reset_index(name='Frequency')
    
    # For each Hour/Weekday, find the START with max Frequency
    best_locs = loc_demand.sort_values(['Hour', 'Weekday', 'Frequency'], ascending=[True, True, False])
    best_locs = best_locs.drop_duplicates(subset=['Hour', 'Weekday'], keep='first')
    
    # Save both
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open(DEMAND_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    with open('models/best_locations_lookup.pkl', 'wb') as f:
        pickle.dump(best_locs, f)
        
    return model, best_locs

def train_classification_models(df):
    print("Training Classification Models (Category & Purpose)...")
    
    # Prepare Data
    # Fill missing values only for training context here or drop
    df_clf = df.dropna(subset=['CATEGORY', 'PURPOSE', 'MILES'])
    
    X = df_clf[['MILES', 'Hour', 'Weekday']]
    
    # 1. Category Model
    le_cat = LabelEncoder()
    y_cat = le_cat.fit_transform(df_clf['CATEGORY'])
    
    model_cat = RandomForestClassifier(n_estimators=100, random_state=42)
    model_cat.fit(X, y_cat)
    print(f"Category Model Accuracy: {model_cat.score(X, y_cat):.2f}")

    # 2. Purpose Model
    le_purp = LabelEncoder()
    y_purp = le_purp.fit_transform(df_clf['PURPOSE'])
    
    model_purp = RandomForestClassifier(n_estimators=100, random_state=42)
    model_purp.fit(X, y_purp)
    print(f"Purpose Model Accuracy: {model_purp.score(X, y_purp):.2f}")
    
    # Save
    with open(CATEGORY_MODEL_PATH, 'wb') as f:
        pickle.dump(model_cat, f)
    with open(PURPOSE_MODEL_PATH, 'wb') as f:
        pickle.dump(model_purp, f)
    with open(CATEGORY_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_cat, f)
    with open(PURPOSE_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_purp, f)
        
    return model_cat, model_purp, le_cat, le_purp

def train_location_model(df):
    print("Training Location Prediction Model (Stop Location) with extended features...")
    
    # Prepare Data
    # We need STOP, MILES, PURPOSE, CATEGORY for this model (Start removed)
    df_loc = df.dropna(subset=['STOP', 'Hour', 'Weekday', 'MILES', 'PURPOSE', 'CATEGORY'])
    
    # Encode Stop Locations (Target)
    le_stop = LabelEncoder()
    y = le_stop.fit_transform(df_loc['STOP'].astype(str))

    # Encode Category & Purpose (Features)
    le_cat = LabelEncoder()
    X_cat = le_cat.fit_transform(df_loc['CATEGORY'].astype(str))
    
    le_purp = LabelEncoder()
    X_purp = le_purp.fit_transform(df_loc['PURPOSE'].astype(str))
    
    # Features: Hour, Weekday, Miles, Encoded Category, Encoded Purpose
    X = pd.DataFrame({
        'Hour': df_loc['Hour'],
        'Weekday': df_loc['Weekday'],
        'MILES': df_loc['MILES'],
        'CATEGORY_ENCODED': X_cat,
        'PURPOSE_ENCODED': X_purp
    })
    
    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print(f"Location Model Accuracy: {model.score(X, y):.2f}")
    
    # Save
    with open(LOCATION_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(STOP_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_stop, f)
    with open(LOC_CAT_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_cat, f)
    with open(LOC_PURP_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_purp, f)
        
    return model, le_stop, le_cat, le_purp

def train_anomaly_detection(df):
    print("Training Anomaly Detection Model (Isolation Forest)...")
    X = df[['MILES', 'Trip_Duration']].dropna()
    model = IsolationForest(contamination=0.08, random_state=42)
    model.fit(X)
    with open(ANOMALY_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model

def train_time_series_forecast(df):
    print("Training Time-Series Forecast Model (ARIMA)...")
    if ARIMA is None:
        print("⚠ Statsmodels (ARIMA) not installed. Skipping.")
        return None
    
    df['Date'] = df['START_DATE'].dt.date
    series = df.groupby('Date').size()
    
    try:
        model = ARIMA(series.values, order=(5, 1, 0))
        model_fit = model.fit()
        with open(FORECAST_MODEL_PATH, 'wb') as f:
            pickle.dump(model_fit, f)
        return model_fit
    except Exception as e:
        print(f"⚠ ARIMA Training failed: {e}")
        return None

def load_models():
    models = {}
    
    # Helper to load safely
    def load_pickle(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    models['duration_model'] = load_pickle(MODEL_PATH)
    models['duration_scaler'] = load_pickle(SCALER_PATH)
    print("Loaded Duration Model.")

    models['location_model'] = load_pickle(LOCATION_MODEL_PATH)
    models['stop_encoder'] = load_pickle(STOP_ENCODER_PATH)
    models['loc_cat_encoder'] = load_pickle(LOC_CAT_ENCODER_PATH)
    models['loc_purp_encoder'] = load_pickle(LOC_PURP_ENCODER_PATH)
    print("Loaded Location Model.")

    models['anomaly_model'] = load_pickle(ANOMALY_MODEL_PATH)
    models['forecast_model'] = load_pickle(FORECAST_MODEL_PATH)
    print("Loaded Anomaly and Forecast Models.")

    models['demand_model'] = load_pickle(DEMAND_MODEL_PATH)
    models['best_locations'] = load_pickle('models/best_locations_lookup.pkl')
    print("Loaded Demand and Revenue Optimizer Models.")

    models['category_model'] = load_pickle(CATEGORY_MODEL_PATH)
    models['purpose_model'] = load_pickle(PURPOSE_MODEL_PATH)
    models['category_encoder'] = load_pickle(CATEGORY_ENCODER_PATH)
    models['purpose_encoder'] = load_pickle(PURPOSE_ENCODER_PATH)
    print("Loaded Classification Models.")
        
    return models

if __name__ == "__main__":
    filepath = 'datasets/UberDataset.csv'
    if os.path.exists(filepath):
        df = load_and_prepare_data(filepath)
        train_trip_duration_model(df)
        train_demand_model(df)
        train_classification_models(df)
        train_location_model(df)
        train_anomaly_detection(df)
        train_time_series_forecast(df)
    else:
        print("Dataset not found!")
