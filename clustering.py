import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def cluster_rides(df, n_clusters=5):
    """
    Perform clustering on Uber ride data based on time and location.
    
    Args:
        df (pd.DataFrame): Uber ride data
        n_clusters (int): Number of clusters to create
        
    Returns:
        pd.DataFrame: Data with cluster labels added
    """
    if df is None or df.empty:
        return df
    
    # Prepare features for clustering
    try:
        # Convert datetime and extract features
        df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')
        df['Hour'] = df['START_DATE'].dt.hour
        df['DayOfWeek'] = df['START_DATE'].dt.dayofweek  # Monday=0, Sunday=6
        
        # Use pickup location and time features
        features = df[['LAT', 'LON', 'Hour', 'DayOfWeek']].dropna()
        
        if len(features) == 0:
            return df
            
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to original DataFrame
        df.loc[features.index, 'Cluster'] = clusters
        
    except Exception as e:
        print(f"Error during clustering: {e}")
    
    return df