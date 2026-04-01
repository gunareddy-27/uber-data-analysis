import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import pickle
import os
from sklearn.ensemble import IsolationForest
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

class UberMLModel:
    """
    A unified Machine Learning module for the Uber Analytics Pro application.
    Encapsulates training, prediction, and serialization logic for all 5 core models.
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Paths for serialization
        self.paths = {
            'duration_model': os.path.join(model_dir, 'trip_duration_model.pkl'),
            'duration_scaler': os.path.join(model_dir, 'trip_duration_scaler.pkl'),
            'demand_model': os.path.join(model_dir, 'demand_model.pkl'),
            'best_locs': os.path.join(model_dir, 'best_locations_lookup.pkl'),
            'category_model': os.path.join(model_dir, 'category_model.pkl'),
            'purpose_model': os.path.join(model_dir, 'purpose_model.pkl'),
            'cat_encoder': os.path.join(model_dir, 'category_encoder.pkl'),
            'purp_encoder': os.path.join(model_dir, 'purpose_encoder.pkl'),
            'location_model': os.path.join(model_dir, 'location_model.pkl'),
            'stop_encoder': os.path.join(model_dir, 'stop_encoder.pkl'),
            'loc_cat_encoder': os.path.join(model_dir, 'loc_cat_encoder.pkl'),
            'loc_purp_encoder': os.path.join(model_dir, 'loc_purp_encoder.pkl'),
            'anomaly_model': os.path.join(model_dir, 'anomaly_model.pkl'),
            'forecast_model': os.path.join(model_dir, 'forecast_model.pkl')
        }

    def prepare_data(self, df):
        """Prepare raw Uber dataset for ML tasks."""
        df = df.copy()
        df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')
        df['END_DATE'] = pd.to_datetime(df['END_DATE'], errors='coerce')
        df = df.dropna(subset=['START_DATE', 'END_DATE', 'MILES'])
        
        # Features
        df['Hour'] = df['START_DATE'].dt.hour
        df['Weekday'] = df['START_DATE'].dt.dayofweek
        df['Trip_Duration'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 60
        
        # Outlier removal
        df = df[df['Trip_Duration'] < 300]
        return df

    def train_all(self, csv_path):
        """Train all models in the pipeline."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
            
        df_raw = pd.read_csv(csv_path)
        df = self.prepare_data(df_raw)
        
        print(f"--- Starting Global Model Training on {len(df)} records ---")
        self._train_duration(df)
        self._train_demand(df)
        self._train_classification(df)
        self._train_location(df)
        self._train_anomaly_detection(df)
        self._train_time_series_forecast(df)
        print("--- Global Training Complete ---")

    def _train_duration(self, df):
        X = df[['MILES', 'Hour', 'Weekday']]
        y = df['Trip_Duration']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        self._save('duration_model', model)
        self._save('duration_scaler', scaler)
        print("✓ Trip Duration Model Trained.")

    def _train_demand(self, df):
        df_demand = df.groupby(['Hour', 'Weekday']).size().reset_index(name='Trip_Count')
        X = df_demand[['Hour', 'Weekday']]
        y = df_demand['Trip_Count']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Revenue Hotspots lookup
        loc_demand = df.groupby(['Hour', 'Weekday', 'START']).size().reset_index(name='Frequency')
        best_locs = loc_demand.sort_values(['Hour', 'Weekday', 'Frequency'], ascending=[True, True, False])
        best_locs = best_locs.drop_duplicates(subset=['Hour', 'Weekday'], keep='first')
        
        self._save('demand_model', model)
        self._save('best_locs', best_locs)
        print("✓ Demand & Hotspot Models Trained.")

    def _train_classification(self, df):
        df_clf = df.dropna(subset=['CATEGORY', 'PURPOSE'])
        X = df_clf[['MILES', 'Hour', 'Weekday']]
        
        # Category
        le_cat = LabelEncoder()
        y_cat = le_cat.fit_transform(df_clf['CATEGORY'])
        model_cat = RandomForestClassifier(n_estimators=100, random_state=42)
        model_cat.fit(X, y_cat)
        
        # Purpose
        le_purp = LabelEncoder()
        y_purp = le_purp.fit_transform(df_clf['PURPOSE'])
        model_purp = RandomForestClassifier(n_estimators=100, random_state=42)
        model_purp.fit(X, y_purp)
        
        self._save('category_model', model_cat)
        self._save('purpose_model', model_purp)
        self._save('cat_encoder', le_cat)
        self._save('purp_encoder', le_purp)
        print("✓ Classification Models Trained.")

    def _train_location(self, df):
        df_loc = df.dropna(subset=['STOP', 'PURPOSE', 'CATEGORY'])
        
        le_stop = LabelEncoder()
        y = le_stop.fit_transform(df_loc['STOP'].astype(str))
        
        le_cat = LabelEncoder()
        X_cat = le_cat.fit_transform(df_loc['CATEGORY'].astype(str))
        
        le_purp = LabelEncoder()
        X_purp = le_purp.fit_transform(df_loc['PURPOSE'].astype(str))
        
        X = pd.DataFrame({
            'Hour': df_loc['Hour'],
            'Weekday': df_loc['Weekday'],
            'MILES': df_loc['MILES'],
            'CATEGORY_ENCODED': X_cat,
            'PURPOSE_ENCODED': X_purp
        })
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        self._save('location_model', model)
        self._save('stop_encoder', le_stop)
        self._save('loc_cat_encoder', le_cat)
        self._save('loc_purp_encoder', le_purp)
        print("✓ Smart Destination Model Trained.")

    def _train_anomaly_detection(self, df):
        """Train Isolation Forest to detect outlier rides (fraud/errors)."""
        X = df[['MILES', 'Trip_Duration']]
        model = IsolationForest(contamination=0.08, random_state=42)
        model.fit(X)
        self._save('anomaly_model', model)
        print("✓ Anomaly Detection Model (Isolation Forest) Trained.")

    def _train_time_series_forecast(self, df):
        """Train ARIMA model for next 24 hours demand forecasting."""
        if ARIMA is None:
            print("⚠ Statsmodels (ARIMA) not installed. Skipping forecast training.")
            return

        df['Date'] = df['START_DATE'].dt.date
        series = df.groupby('Date').size()
        
        try:
            # ARIMA(5,1,0) for daily trend tracking
            model = ARIMA(series.values, order=(5, 1, 0))
            model_fit = model.fit()
            self._save('forecast_model', model_fit)
            print("✓ Time-Series Forecast Model (ARIMA) Trained.")
        except Exception as e:
            print(f"⚠ ARIMA Training failed: {e}")

    def _save(self, key, obj):
        with open(self.paths[key], 'wb') as f:
            pickle.dump(obj, f)

    def load_all(self):
        """Load all models into a dictionary for use in Flask."""
        models = {}
        for key, path in self.paths.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models[key] = pickle.load(f)
            else:
                models[key] = None
        return models

if __name__ == "__main__":
    # Test script for standalone training
    trainer = UberMLModel()
    dataset = 'datasets/UberDataset.csv'
    if os.path.exists(dataset):
        trainer.train_all(dataset)
    else:
        print("Dataset not found. Please ensure UberDataset.csv is in the datasets folder.")
