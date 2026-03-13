import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template
from analysis import analyze_data
from visualization import plot_analysis

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Data cleaning and preprocessing
def clean_data(df):
    df.dropna(inplace=True)
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')
    df["Month"] = df["START_DATE"].dt.month_name()
    return df

# Flask app
app = Flask(__name__)

df = load_data('C:\Users\anush\OneDrive\Desktop\uber\datasets\UberDataset.csv')
df = clean_data(df)

hourly_trips, daily_trips, weekday_trips = analyze_data(df)

monthly_trips = df.groupby("Month").size().reindex([
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
], fill_value=0) if "Month" in df.columns and df["Month"].notna().sum() > 0 else None

trip_durations = None
if "END_DATE" in df.columns:
    df["END_DATE"] = pd.to_datetime(df["END_DATE"], errors='coerce')
    df["Trip_Duration"] = (df["END_DATE"] - df["START_DATE"]).dt.total_seconds() / 60
    trip_durations = df["Trip_Duration"].dropna()
    trip_durations = trip_durations[trip_durations < 300]

purpose_counts = df['PURPOSE'].value_counts() if 'PURPOSE' in df.columns else None

plot_analysis(hourly_trips, daily_trips, weekday_trips, monthly_trips, trip_durations, purpose_counts)

@app.route('/')
def home():
    return render_template('index.html', 
                           hourly_trips_img='hourly_trips.png',
                           daily_trips_img='daily_trips.png',
                           weekday_trips_img='weekday_trips.png',
                           monthly_trips_img='monthly_trips.png',
                           hourly_heatmap_img='hourly_heatmap.png',
                           trip_purpose_distribution_img='trip_purpose_distribution.png',
                           trip_duration_img='trip_duration.png')

if __name__ == '__main__':
    app.run(debug=True)
