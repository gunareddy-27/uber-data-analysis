import plotly.graph_objects as go
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import numpy as np
import plotly.express as px

def plot_analysis(hourly_trips, daily_trips, weekday_trips, monthly_trips, 
                 trip_durations, purpose_counts, df):
    """Generate and save interactive Plotly charts with clustering analysis."""
    if not os.path.exists('static'):
        os.makedirs('static')

    # 1. Basic Plots (unchanged from your original)
    # Hourly Trips Plot
    if hourly_trips is not None and not hourly_trips.empty:
        fig = px.bar(hourly_trips, x=hourly_trips.index, y=hourly_trips.values,
                     labels={'x': "Hour of Day", 'y': "Number of Trips"},
                     title="Hourly Uber Trips", text=hourly_trips.values)
        fig.update_traces(textposition='outside', hoverinfo="x+y")
        fig.write_html("static/hourly_trips.html")

    # Daily Trips Plot
    if daily_trips is not None and not daily_trips.empty:
        fig = px.line(daily_trips, x=daily_trips.index, y=daily_trips.values,
                      labels={'x': "Date", 'y': "Number of Trips"},
                      title="Daily Uber Trips")
        fig.write_html("static/daily_trips.html")

    # Weekday Trips Plot
    if weekday_trips is not None and not weekday_trips.empty:
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig = px.bar(weekday_trips, x=weekday_trips.index, y=weekday_trips.values,
                     labels={'x': "Day of Week", 'y': "Number of Trips"},
                     title="Weekday Uber Trips", category_orders={"x": weekday_names})
        fig.write_html("static/weekday_trips.html")

    # Monthly Trips Plot
    if monthly_trips is not None and not monthly_trips.empty:
        fig = px.line(monthly_trips, x=monthly_trips.index, y=monthly_trips.values,
                      labels={'x': "Month", 'y': "Number of Trips"},
                      title="Monthly Uber Trips")
        fig.write_html("static/monthly_trips.html")

    # Trip Durations Plot
    if trip_durations is not None and not trip_durations.empty:
        fig = px.histogram(trip_durations, x=trip_durations.values,
                           labels={'x': "Trip Duration (minutes)", 'y': "Count"},
                           title="Trip Duration Distribution")
        fig.write_html("static/trip_durations.html")

    # Trip Purposes Plot
    if purpose_counts is not None and not purpose_counts.empty:
        fig = px.pie(purpose_counts, values=purpose_counts.values, names=purpose_counts.index,
                      title="Trip Purposes Distribution")
        fig.write_html("static/trip_purposes.html")

    # 2. Enhanced Clustering Visualizations
    if df is not None and not df.empty:
        # A. Temporal Clustering (hour + weekday)
        try:
            if 'Hour' in df.columns and 'Weekday' in df.columns:
                # Convert Weekday to numeric for clustering if it's strings
                temp_df = df[['Hour', 'Weekday']].dropna().copy()
                if temp_df['Weekday'].dtype == object:
                    l_enc = LabelEncoder()
                    temp_df['Weekday_Num'] = l_enc.fit_transform(temp_df['Weekday'])
                    temporal_data = temp_df[['Hour', 'Weekday_Num']]
                else:
                    temporal_data = temp_df[['Hour', 'Weekday']]

                if len(temporal_data) > 0:
                    scaler = StandardScaler()
                    temp_scaled = scaler.fit_transform(temporal_data)
                    
                    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                    df['temp_cluster'] = kmeans.fit_predict(temp_scaled)
                    
                    fig = px.scatter(df, x='Hour', y='Weekday', color='temp_cluster',
                                   title="Temporal Clustering (Hour vs Weekday)",
                                   labels={'Hour': 'Hour of Day', 'Weekday': 'Day of Week'})
                    fig.write_html("static/temporal_clusters.html")
                    
                    # Additional temporal cluster analysis
                    cluster_counts = df.groupby(['temp_cluster', 'Weekday']).size().reset_index(name='count')
                    fig = px.bar(cluster_counts, x='Weekday', y='count', color='temp_cluster',
                                title="Trip Distribution by Temporal Cluster",
                                labels={'Weekday': 'Day of Week', 'count': 'Number of Trips'})
                    fig.write_html("static/temporal_cluster_distribution.html")
        except Exception as e:
            print(f"Temporal clustering failed: {str(e)}")

        # B. Trip Purpose Clustering by Duration
        try:
            if 'PURPOSE' in df.columns and 'Trip_Duration' in df.columns:
                # Create a purpose-duration dataframe
                purpose_duration = df[['PURPOSE', 'Trip_Duration']].dropna()
                
                # Get average duration by purpose
                avg_duration = purpose_duration.groupby('PURPOSE')['Trip_Duration'].mean().reset_index()
                
                # Cluster purposes by average duration
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                avg_duration['purpose_cluster'] = kmeans.fit_predict(
                    avg_duration[['Trip_Duration']])
                
                fig = px.bar(avg_duration, x='PURPOSE', y='Trip_Duration', 
                            color='purpose_cluster',
                            title="Trip Purpose Clustering by Average Duration",
                            labels={'Trip_Duration': 'Average Duration (minutes)'})
                fig.write_html("static/purpose_duration_clusters.html")
        except Exception as e:
            print(f"Purpose clustering failed: {str(e)}")

        # C. Time-Duration Clustering
        try:
            if 'Hour' in df.columns and 'Trip_Duration' in df.columns:
                time_duration = df[['Hour', 'Trip_Duration']].dropna()
                
                # Scale features
                scaler = StandardScaler()
                td_scaled = scaler.fit_transform(time_duration)
                
                # Cluster
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                time_duration['td_cluster'] = kmeans.fit_predict(td_scaled)
                
                # Visualize
                fig = px.scatter(time_duration, x='Hour', y='Trip_Duration', 
                               color='td_cluster',
                               title="Time-Duration Clusters",
                               labels={'Hour': 'Hour of Day', 
                                      'Trip_Duration': 'Trip Duration (minutes)'})
                fig.write_html("static/time_duration_clusters.html")
                
                # Add box plot visualization
                fig = px.box(time_duration, x='td_cluster', y='Trip_Duration',
                            title="Trip Duration Distribution by Time-Duration Cluster")
                fig.write_html("static/time_duration_boxplot.html")
        except Exception as e:
            print(f"Time-duration clustering failed: {str(e)}")

        # D. Enhanced K-Means on Trip Durations
        if trip_durations is not None and not trip_durations.empty:
            try:
                trip_data = trip_durations.dropna().values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(trip_data)

                df_clustered = pd.DataFrame({
                    'Trip Duration': trip_data.flatten(), 
                    'Cluster': clusters,
                    'Count': 1  # For aggregation
                })
                
                # Cluster statistics
                cluster_stats = df_clustered.groupby('Cluster')['Trip Duration'].agg(
                    ['mean', 'median', 'count', 'std']).reset_index()
                cluster_stats.columns = ['Cluster', 'Mean Duration', 'Median Duration', 
                                       'Trip Count', 'Duration Std Dev']
                
                # Save cluster statistics to CSV
                cluster_stats.to_csv("static/duration_cluster_stats.csv", index=False)
                
                # Improved visualization with box plots
                fig = px.box(df_clustered, x='Cluster', y='Trip Duration',
                            title="Trip Duration Clusters Distribution",
                            points="all",  # Show all points
                            hover_data=['Trip Duration'])
                fig.write_html("static/kmeans_trip_duration.html")
                
                # Add histogram by cluster
                fig = px.histogram(df_clustered, x='Trip Duration', color='Cluster',
                                  title="Trip Duration Distribution by Cluster",
                                  barmode='overlay', marginal='rug',
                                  hover_data=['Trip Duration'])
                fig.write_html("static/kmeans_trip_duration_hist.html")
                
                # Add violin plot
                fig = px.violin(df_clustered, y='Trip Duration', x='Cluster',
                               box=True, points="all",
                               title="Trip Duration Distribution by Cluster (Violin Plot)")
                fig.write_html("static/kmeans_trip_duration_violin.html")
            except Exception as e:
                print(f"Duration clustering failed: {str(e)}")

        # E. Combined Temporal-Purpose Analysis
        try:
            if 'PURPOSE' in df.columns and 'Hour' in df.columns:
                # Create pivot table of trips by Hour and PURPOSE
                purpose_hour = df.groupby(['Hour', 'PURPOSE']).size().unstack().fillna(0)
                
                # Normalize and cluster purposes by hourly pattern
                scaler = StandardScaler()
                ph_scaled = scaler.fit_transform(purpose_hour.T)
                
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                purpose_clusters = kmeans.fit_predict(ph_scaled)
                
                # Create dataframe for visualization
                ph_df = purpose_hour.T.reset_index()
                ph_df['purpose_cluster'] = purpose_clusters
                
                # Melt for plotting
                ph_melt = ph_df.melt(id_vars=['PURPOSE', 'purpose_cluster'], 
                                    var_name='Hour', value_name='count')
                
                # Plot clustered purposes
                fig = px.line(ph_melt, x='Hour', y='count', color='PURPOSE',
                             facet_col='purpose_cluster', facet_col_wrap=1,
                             title="Trip Purpose Patterns by Hour (Clustered)",
                             height=800)
                fig.write_html("static/purpose_hour_clusters.html")
        except Exception as e:
            print(f"Purpose-hour clustering failed: {str(e)}")

        # F. Route Network Graph (Simplified for Plotly)
        try:
            if 'START' in df.columns and 'STOP' in df.columns:
                routes = df.groupby(['START', 'STOP']).size().reset_index(name='trips')
                routes = routes.sort_values('trips', ascending=False).head(50) # Top 50 routes
                
                fig = px.parallel_categories(routes, dimensions=['START', 'STOP'], 
                                          color="trips", color_continuous_scale=px.colors.sequential.Inferno,
                                          title="Interactive Route Flow (Top 50 Routes)")
                fig.write_html("static/route_network.html")
        except Exception as e:
            print(f"Route network failed: {str(e)}")

        # G. Anomaly Detection Visualizer
        try:
            if 'MILES' in df.columns and 'Trip_Duration' in df.columns:
                data = df[['MILES', 'Trip_Duration']].dropna()
                if len(data) > 0:
                    iso_forest = IsolationForest(contamination=0.05, random_state=42)
                    df['is_anomaly'] = iso_forest.fit_predict(data)
                    df['anomaly_label'] = df['is_anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
                    
                    fig = px.scatter(df, x='MILES', y='Trip_Duration', color='anomaly_label',
                                   symbol='anomaly_label',
                                   title="Trip Anomaly Detection (Isolation Forest)",
                                   labels={'Trip_Duration': 'Duration (min)', 'MILES': 'Distance (mi)'},
                                   color_discrete_map={'Normal': '#636EFA', 'Anomaly': '#EF553B'},
                                   hover_data=['START', 'STOP', 'PURPOSE'])
                    fig.write_html("static/anomalies.html")
        except Exception as e:
            print(f"Anomaly detection failed: {str(e)}")

        # H. Calendar Heatmap (Daily Intensity)
        try:
            if 'START_DATE' in df.columns:
                cal_df = df.copy()
                cal_df['date_only'] = cal_df['START_DATE'].dt.date
                daily_counts = cal_df.groupby('date_only').size().reset_index(name='trips')
                daily_counts['date_only'] = pd.to_datetime(daily_counts['date_only'])
                
                # Use week of year and day name
                daily_counts['week'] = daily_counts['date_only'].dt.strftime('%W').astype(int)
                daily_counts['day'] = daily_counts['date_only'].dt.day_name()
                
                # Pivot for heatmap
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = daily_counts.pivot_table(index='day', columns='week', values='trips', aggfunc='sum')
                heatmap_data = heatmap_data.reindex(days_order).fillna(0)
                
                fig = px.imshow(heatmap_data, 
                               labels=dict(x="Week of Year", y="Day of Week", color="Trips"),
                               color_continuous_scale='Greens',
                               title="Yearly Trip Activity Heatmap")
                fig.write_html("static/calendar_heatmap.html")
        except Exception as e:
            print(f"Calendar heatmap failed: {str(e)}")

        # I. Demand Forecast Heatmap (Hour vs Weekday)
        try:
            if 'Hour' in df.columns and 'Weekday' in df.columns:
                # Map Weekday number to name for better visualization
                days_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                           4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                h_df = df.copy()
                h_df['day_name'] = h_df['Weekday'].map(days_map)
                
                demand_matrix = h_df.pivot_table(index='day_name', columns='Hour', 
                                              values='START', aggfunc='count').fillna(0)
                
                # Sort days correctly
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                demand_matrix = demand_matrix.reindex(days_order)
                
                fig = px.imshow(demand_matrix,
                               labels=dict(x="Hour of Day", y="Day of Week", color="Demand Density"),
                               x=demand_matrix.columns,
                               y=demand_matrix.index,
                               color_continuous_scale='YlOrRd',
                               title="Revenue Optimizer: Demand Forecast Heatmap")
                fig.write_html("static/demand_forecast_heatmap.html")
        except Exception as e:
            print(f"Demand forecast heatmap failed: {str(e)}")