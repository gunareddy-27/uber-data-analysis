import pandas as pd
import numpy as np
import time
import random
from datetime import datetime

class UberSystemIntelligence:
    """
    Advanced simulation engine for Dynamic Pricing and Real-Time Streaming.
    """
    
    def __init__(self, dataset_path='datasets/UberDataset.csv'):
        self.dataset_path = dataset_path
        
    def simulate_pricing(self, price_increase_pct):
        """
        Simulates the impact of price changes on demand and revenue.
        Uses a standard price elasticity of demand model (Elasticity = -1.5).
        """
        # Load sample data
        df = pd.read_csv(self.dataset_path)
        base_trips = len(df)
        base_revenue = df['MILES'].sum() * 2.15 # Average $2.15/mile
        
        elasticity = -1.2 # Demand drops by 1.2% for every 1% price increase
        
        # Calculate impact
        demand_change_pct = price_increase_pct * elasticity
        new_trips = base_trips * (1 + (demand_change_pct / 100))
        
        # New revenue = new_trips * average_miles * (new_price_per_mile)
        avg_miles = df['MILES'].mean()
        new_price_per_mile = 2.15 * (1 + (price_increase_pct / 100))
        new_revenue = new_trips * avg_miles * new_price_per_mile
        
        revenue_impact = new_revenue - base_revenue
        
        return {
            'base_trips': int(base_trips),
            'new_trips': int(new_trips),
            'demand_change_pct': round(demand_change_pct, 2),
            'base_revenue': round(base_revenue, 2),
            'new_revenue': round(new_revenue, 2),
            'revenue_impact': round(revenue_impact, 2),
            'is_positive': revenue_impact > 0
        }

    def generate_live_stream(self, count=10):
        """
        Simulates a Kafka/Spark real-time ride stream.
        Returns a list of mock ride events.
        """
        df = pd.read_csv(self.dataset_path)
        locations = df['START'].dropna().unique().tolist()
        categories = df['CATEGORY'].dropna().unique().tolist()
        purposes = df['PURPOSE'].dropna().unique().tolist()
        
        stream_data = []
        for _ in range(count):
            event = {
                'id': f"ride_{random.randint(10000, 99999)}",
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'location': random.choice(locations),
                'miles': round(random.uniform(1.5, 30.0), 2),
                'category': random.choice(categories),
                'purpose': random.choice(purposes),
                'status': random.choice(['REQUESTED', 'IN_PROGRESS', 'COMPLETED'])
            }
            stream_data.append(event)
            
        return stream_data

    def detect_anomalies(self, df, model):
        """Uses Isolation Forest to flag anomalous rides."""
        if model is None:
            return []
            
        df_prep = df.copy()
        df_prep['START_DATE'] = pd.to_datetime(df_prep['START_DATE'], errors='coerce')
        df_prep['END_DATE'] = pd.to_datetime(df_prep['END_DATE'], errors='coerce')
        df_prep['Trip_Duration'] = (df_prep['END_DATE'] - df_prep['START_DATE']).dt.total_seconds() / 60
        df_prep = df_prep.dropna(subset=['MILES', 'Trip_Duration'])
        
        X = df_prep[['MILES', 'Trip_Duration']]
        preds = model.predict(X)
        
        # -1 indicates an anomaly
        anomalies = df_prep[preds == -1]
        return anomalies.head(20).to_dict(orient='records')
