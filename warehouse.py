"""
Ride-Sharing Data Warehouse Analytics Module
Computes KPIs: Revenue by City, Surge Pricing Trends, Peak Hour Analysis, Demand Heatmaps
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


# --- Fare Estimation ---
def estimate_fare(miles, hour):
    """
    Simulate realistic Uber fare based on distance and time-of-day surge.
    Formula: $2.50 base + $1.75/mile + surge multiplier
    """
    base_fare = 2.50
    per_mile = 1.75
    surge = get_surge_multiplier(hour)
    return round((base_fare + per_mile * miles) * surge, 2)


def get_surge_multiplier(hour):
    """Return a surge pricing multiplier based on hour of day."""
    # Peak hours: 7-9 AM, 5-8 PM (commute), Late night 11 PM-2 AM
    surge_map = {
        0: 1.5, 1: 1.6, 2: 1.4, 3: 1.1, 4: 1.0, 5: 1.0,
        6: 1.1, 7: 1.6, 8: 1.8, 9: 1.5, 10: 1.1, 11: 1.0,
        12: 1.2, 13: 1.1, 14: 1.0, 15: 1.1, 16: 1.3, 17: 1.7,
        18: 1.9, 19: 1.8, 20: 1.5, 21: 1.3, 22: 1.4, 23: 1.5
    }
    return surge_map.get(hour, 1.0)


# --- Core Analytics Functions ---

def prepare_warehouse_df(df):
    """Add warehouse-specific computed columns to the dataframe."""
    wdf = df.copy()
    wdf['START_DATE'] = pd.to_datetime(wdf['START_DATE'], errors='coerce')
    wdf['END_DATE'] = pd.to_datetime(wdf['END_DATE'], errors='coerce')
    wdf['Hour'] = wdf['START_DATE'].dt.hour
    wdf['Weekday'] = wdf['START_DATE'].dt.dayofweek
    wdf['DayName'] = wdf['START_DATE'].dt.day_name()
    wdf['Month'] = wdf['START_DATE'].dt.month_name()
    wdf['Trip_Duration'] = (wdf['END_DATE'] - wdf['START_DATE']).dt.total_seconds() / 60

    # Compute estimated fare and surge
    wdf['MILES'] = pd.to_numeric(wdf['MILES'], errors='coerce')
    wdf = wdf.dropna(subset=['MILES', 'Hour'])
    wdf['Surge_Multiplier'] = wdf['Hour'].apply(get_surge_multiplier)
    wdf['Estimated_Fare'] = wdf.apply(lambda r: estimate_fare(r['MILES'], r['Hour']), axis=1)

    return wdf


def compute_warehouse_kpis(wdf):
    """Compute top-level KPI metrics."""
    total_revenue = round(wdf['Estimated_Fare'].sum(), 2)
    total_trips = len(wdf)
    avg_fare = round(wdf['Estimated_Fare'].mean(), 2) if total_trips > 0 else 0

    # Top revenue city
    city_rev = wdf.groupby('START')['Estimated_Fare'].sum()
    top_city = city_rev.idxmax() if not city_rev.empty else 'N/A'
    top_city_rev = round(city_rev.max(), 2) if not city_rev.empty else 0

    # Peak revenue hour
    hour_rev = wdf.groupby('Hour')['Estimated_Fare'].sum()
    peak_hour = int(hour_rev.idxmax()) if not hour_rev.empty else 0
    peak_hour_rev = round(hour_rev.max(), 2) if not hour_rev.empty else 0

    # Avg surge
    avg_surge = round(wdf['Surge_Multiplier'].mean(), 2)

    return {
        'total_revenue': total_revenue,
        'total_trips': total_trips,
        'avg_fare': avg_fare,
        'top_city': top_city,
        'top_city_rev': top_city_rev,
        'peak_hour': peak_hour,
        'peak_hour_rev': peak_hour_rev,
        'avg_surge': avg_surge
    }


def revenue_by_city(wdf):
    """Revenue grouped by START location (city)."""
    city_data = wdf.groupby('START').agg(
        Total_Revenue=('Estimated_Fare', 'sum'),
        Trip_Count=('Estimated_Fare', 'count'),
        Avg_Fare=('Estimated_Fare', 'mean'),
        Avg_Miles=('MILES', 'mean')
    ).reset_index().sort_values('Total_Revenue', ascending=False)
    city_data = city_data.round(2)
    return city_data


def surge_pricing_trends(wdf):
    """Surge pricing by hour with actual demand vs. surge factor."""
    hourly = wdf.groupby('Hour').agg(
        Avg_Surge=('Surge_Multiplier', 'mean'),
        Trip_Count=('Estimated_Fare', 'count'),
        Avg_Fare=('Estimated_Fare', 'mean'),
        Total_Revenue=('Estimated_Fare', 'sum')
    ).reset_index().round(2)
    return hourly


def peak_hour_analysis(wdf):
    """Detailed hour-by-hour analysis with peak classification."""
    hourly = wdf.groupby('Hour').agg(
        Trip_Count=('Estimated_Fare', 'count'),
        Total_Revenue=('Estimated_Fare', 'sum'),
        Avg_Fare=('Estimated_Fare', 'mean'),
        Avg_Miles=('MILES', 'mean'),
        Avg_Surge=('Surge_Multiplier', 'mean')
    ).reset_index().round(2)

    # Classify peak/off-peak (top 33% by trips = Peak)
    threshold = hourly['Trip_Count'].quantile(0.67)
    hourly['Period'] = hourly['Trip_Count'].apply(
        lambda x: 'Peak' if x >= threshold else 'Off-Peak'
    )
    return hourly


def demand_heatmap_data(wdf):
    """Hour × Weekday demand matrix."""
    days_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    hdf = wdf.copy()
    hdf['DayLabel'] = hdf['Weekday'].map(days_map)
    matrix = hdf.pivot_table(index='DayLabel', columns='Hour',
                             values='Estimated_Fare', aggfunc='count').fillna(0)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    matrix = matrix.reindex(days_order)
    return matrix


# --- Chart Generation ---

def generate_warehouse_charts(df):
    """Generate all warehouse Plotly HTML charts and save to static/."""
    if not os.path.exists('static'):
        os.makedirs('static')

    wdf = prepare_warehouse_df(df)
    if wdf.empty:
        print("⚠ Warehouse: No valid data to generate charts.")
        return {}

    kpis = compute_warehouse_kpis(wdf)

    # 1. Revenue by City (Top 15)
    try:
        city_data = revenue_by_city(wdf).head(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=city_data['START'],
            y=city_data['Total_Revenue'],
            marker=dict(
                color=city_data['Total_Revenue'],
                colorscale='Tealgrn',
                showscale=True,
                colorbar=dict(title='Revenue ($)')
            ),
            text=city_data['Total_Revenue'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<br>Trips: %{customdata[0]}<br>Avg Fare: $%{customdata[1]:.2f}<extra></extra>',
            customdata=city_data[['Trip_Count', 'Avg_Fare']].values
        ))
        fig.update_layout(
            title='Revenue by City (Top 15)',
            xaxis_title='City / Location',
            yaxis_title='Estimated Revenue ($)',
            template='plotly_white',
            xaxis_tickangle=-45,
            margin=dict(b=120)
        )
        fig.write_html('static/warehouse_revenue_by_city.html')
    except Exception as e:
        print(f"Warehouse chart error (revenue_by_city): {e}")

    # 2. Surge Pricing Trends
    try:
        surge_data = surge_pricing_trends(wdf)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=surge_data['Hour'], y=surge_data['Avg_Surge'],
                mode='lines+markers', name='Surge Multiplier',
                line=dict(color='#EF553B', width=3),
                marker=dict(size=8),
                hovertemplate='Hour: %{x}:00<br>Surge: %{y:.2f}x<extra></extra>'
            ), secondary_y=False
        )
        fig.add_trace(
            go.Bar(
                x=surge_data['Hour'], y=surge_data['Trip_Count'],
                name='Trip Volume', opacity=0.4,
                marker_color='#636EFA',
                hovertemplate='Hour: %{x}:00<br>Trips: %{y}<extra></extra>'
            ), secondary_y=True
        )
        fig.update_layout(
            title='Surge Pricing Trends vs. Demand Volume',
            xaxis_title='Hour of Day',
            template='plotly_white',
            legend=dict(x=0.01, y=0.99),
            hovermode='x unified'
        )
        fig.update_yaxes(title_text='Surge Multiplier', secondary_y=False)
        fig.update_yaxes(title_text='Number of Trips', secondary_y=True)
        fig.write_html('static/warehouse_surge_trends.html')
    except Exception as e:
        print(f"Warehouse chart error (surge_trends): {e}")

    # 3. Peak Hour Analysis
    try:
        peak_data = peak_hour_analysis(wdf)
        colors = ['#EF553B' if p == 'Peak' else '#636EFA' for p in peak_data['Period']]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=peak_data['Hour'], y=peak_data['Trip_Count'],
                name='Trip Count', marker_color=colors,
                hovertemplate='Hour: %{x}:00<br>Trips: %{y}<br>%{customdata}<extra></extra>',
                customdata=peak_data['Period']
            ), secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=peak_data['Hour'], y=peak_data['Avg_Fare'],
                mode='lines+markers', name='Avg Fare ($)',
                line=dict(color='#00CC96', width=3),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='Hour: %{x}:00<br>Avg Fare: $%{y:.2f}<extra></extra>'
            ), secondary_y=True
        )
        fig.update_layout(
            title='Peak Hour Analysis — Trip Volume vs Avg Fare',
            xaxis_title='Hour of Day',
            template='plotly_white',
            legend=dict(x=0.01, y=0.99),
            hovermode='x unified'
        )
        fig.update_yaxes(title_text='Trip Count', secondary_y=False)
        fig.update_yaxes(title_text='Average Fare ($)', secondary_y=True)
        fig.write_html('static/warehouse_peak_hours.html')
    except Exception as e:
        print(f"Warehouse chart error (peak_hours): {e}")

    # 4. Ride Demand Heatmap (Hour × Weekday)
    try:
        matrix = demand_heatmap_data(wdf)
        fig = px.imshow(
            matrix,
            labels=dict(x='Hour of Day', y='Day of Week', color='Ride Count'),
            x=[f'{h}:00' for h in matrix.columns],
            y=matrix.index,
            color_continuous_scale='YlOrRd',
            title='Ride Demand Heatmap (Hour × Day of Week)',
            aspect='auto'
        )
        fig.update_layout(template='plotly_white', margin=dict(l=100))
        fig.write_html('static/warehouse_demand_heatmap.html')
    except Exception as e:
        print(f"Warehouse chart error (demand_heatmap): {e}")

    # 5. Revenue Distribution by Hour (Treemap-style sunburst)
    try:
        hourly_rev = wdf.groupby(['DayName', 'Hour']).agg(
            Revenue=('Estimated_Fare', 'sum'),
            Trips=('Estimated_Fare', 'count')
        ).reset_index()
        hourly_rev['HourLabel'] = hourly_rev['Hour'].apply(lambda h: f'{h}:00')
        hourly_rev = hourly_rev.round(2)

        fig = px.sunburst(
            hourly_rev,
            path=['DayName', 'HourLabel'],
            values='Revenue',
            color='Revenue',
            color_continuous_scale='Viridis',
            title='Revenue Breakdown: Day → Hour',
            hover_data={'Trips': True, 'Revenue': ':.2f'}
        )
        fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
        fig.write_html('static/warehouse_revenue_sunburst.html')
    except Exception as e:
        print(f"Warehouse chart error (sunburst): {e}")

    return kpis
