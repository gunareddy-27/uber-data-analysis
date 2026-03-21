"""
Data Analyst AI Agent — Auto Insights Generator
Upload any CSV → Auto-clean → Generate charts → Plain-English insights → Trend predictions
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import os
import json
import uuid
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# 1. DATA CLEANING
# ──────────────────────────────────────────────

def clean_data(df):
    """Auto-clean a CSV dataframe. Returns cleaned df + cleaning report."""
    report = []
    original_shape = df.shape

    # Drop fully empty rows/cols
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    dropped_rows = original_shape[0] - df.shape[0]
    dropped_cols = original_shape[1] - df.shape[1]
    if dropped_rows > 0:
        report.append(f"Removed {dropped_rows} completely empty rows.")
    if dropped_cols > 0:
        report.append(f"Removed {dropped_cols} completely empty columns.")

    # Drop duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        report.append(f"Removed {dup_count} duplicate rows.")

    # Auto-detect and convert date columns AND Automated Feature Engineering
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                if parsed.notna().sum() > len(df) * 0.5:
                    df[col] = parsed
                    report.append(f"Converted '{col}' to datetime.")
                    
                    # Automated Feature Engineering (XAI / Time series preparation)
                    df[f'{col}_Year'] = df[col].dt.year
                    df[f'{col}_Month'] = df[col].dt.month
                    df[f'{col}_DayOfWeek'] = df[col].dt.dayofweek
                    df[f'{col}_IsWeekend'] = df[f'{col}_DayOfWeek'].isin([5, 6]).astype(int)
                    report.append(f"Auto-engineered time-series features (Year, Month, DayOfWeek, IsWeekend) from '{col}'.")
            except Exception:
                pass

    # Auto-convert numeric columns stored as strings
    for col in df.select_dtypes(include=['object']).columns:
        try:
            numeric = pd.to_numeric(df[col].str.replace(',', '').str.strip(), errors='coerce')
            if numeric.notna().sum() > len(df) * 0.5:
                df[col] = numeric
                report.append(f"Converted '{col}' to numeric.")
        except Exception:
            pass

    # Fill missing numeric with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = df[col].isna().sum()
        if 0 < missing < len(df) * 0.5:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report.append(f"Filled {missing} missing values in '{col}' with median ({median_val:.2f}).")

    # Fill missing categorical with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        missing = df[col].isna().sum()
        if 0 < missing < len(df) * 0.5:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            report.append(f"Filled {missing} missing values in '{col}' with mode ('{mode_val}').")

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    if not report:
        report.append("Data was already clean — no changes needed.")

    return df, report


# ──────────────────────────────────────────────
# 2. AUTO INSIGHTS (Plain English)
# ──────────────────────────────────────────────

def generate_insights(df):
    """Generate plain-English insights from the dataset."""
    insights = []

    # Dataset overview
    rows, cols = df.shape
    insights.append({
        'icon': 'fa-table',
        'category': 'Overview',
        'color': 'indigo',
        'text': f"This dataset contains **{rows:,} records** across **{cols} columns**. "
                f"Column types: {len(df.select_dtypes(include=[np.number]).columns)} numeric, "
                f"{len(df.select_dtypes(include=['object']).columns)} text, "
                f"{len(df.select_dtypes(include=['datetime64']).columns)} date."
    })

    # Missing data assessment
    total_missing = df.isna().sum().sum()
    missing_pct = (total_missing / (rows * cols)) * 100
    if total_missing > 0:
        worst_col = df.isna().sum().idxmax()
        worst_pct = (df[worst_col].isna().sum() / rows) * 100
        insights.append({
            'icon': 'fa-exclamation-triangle',
            'category': 'Data Quality',
            'color': 'amber',
            'text': f"There are **{total_missing:,} missing values** ({missing_pct:.1f}% of all cells). "
                    f"The column with the most gaps is **'{worst_col}'** ({worst_pct:.1f}% missing)."
        })
    else:
        insights.append({
            'icon': 'fa-check-circle',
            'category': 'Data Quality',
            'color': 'green',
            'text': "The dataset has **no missing values** — excellent data quality!"
        })

    # Numeric column insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        for col in numeric_cols[:5]:  # Top 5 numeric cols
            series = df[col].dropna()
            if len(series) > 0:
                mean_val = series.mean()
                std_val = series.std()
                min_val = series.min()
                max_val = series.max()
                skew = series.skew()

                skew_desc = "roughly symmetric"
                if skew > 1:
                    skew_desc = "heavily right-skewed (long tail of high values)"
                elif skew > 0.5:
                    skew_desc = "moderately right-skewed"
                elif skew < -1:
                    skew_desc = "heavily left-skewed"
                elif skew < -0.5:
                    skew_desc = "moderately left-skewed"

                insights.append({
                    'icon': 'fa-chart-bar',
                    'category': f'Column: {col}',
                    'color': 'blue',
                    'text': f"**{col}** ranges from **{min_val:,.2f}** to **{max_val:,.2f}** "
                            f"(mean: {mean_val:,.2f}, std: {std_val:,.2f}). "
                            f"The distribution is {skew_desc}."
                })

        # Correlations
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Find strongest correlation (excluding self)
            np.fill_diagonal(corr_matrix.values, 0)
            max_corr_idx = np.unravel_index(np.abs(corr_matrix.values).argmax(), corr_matrix.shape)
            col1 = corr_matrix.columns[max_corr_idx[0]]
            col2 = corr_matrix.columns[max_corr_idx[1]]
            corr_val = corr_matrix.iloc[max_corr_idx[0], max_corr_idx[1]]

            strength = "weak"
            if abs(corr_val) > 0.7:
                strength = "strong"
            elif abs(corr_val) > 0.4:
                strength = "moderate"

            direction = "positive" if corr_val > 0 else "negative"

            insights.append({
                'icon': 'fa-link',
                'category': 'Correlations',
                'color': 'purple',
                'text': f"The strongest correlation is between **'{col1}'** and **'{col2}'** "
                        f"(r = {corr_val:.3f}), a {strength} {direction} relationship. "
                        f"{'When one goes up, the other tends to go up too.' if corr_val > 0 else 'When one goes up, the other tends to go down.'}"
            })

    # Categorical column insights
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols[:3]:  # Top 3
        n_unique = df[col].nunique()
        top_val = df[col].value_counts().index[0] if n_unique > 0 else 'N/A'
        top_pct = (df[col].value_counts().iloc[0] / len(df)) * 100 if n_unique > 0 else 0

        insights.append({
            'icon': 'fa-tags',
            'category': f'Column: {col}',
            'color': 'teal',
            'text': f"**{col}** has **{n_unique} unique values**. "
                    f"The most common is **'{top_val}'** ({top_pct:.1f}% of records)."
        })

    # Date column insights
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    for col in date_cols[:2]:
        date_range = df[col].max() - df[col].min()
        insights.append({
            'icon': 'fa-calendar',
            'category': f'Time Range: {col}',
            'color': 'rose',
            'text': f"**{col}** spans from **{df[col].min().strftime('%Y-%m-%d')}** to "
                    f"**{df[col].max().strftime('%Y-%m-%d')}** ({date_range.days} days)."
        })

    # Outlier detection
    for col in numeric_cols[:3]:
        series = df[col].dropna()
        if len(series) > 10:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum()
            if outliers > 0:
                pct = (outliers / len(series)) * 100
                insights.append({
                    'icon': 'fa-bolt',
                    'category': f'Outliers: {col}',
                    'color': 'red',
                    'text': f"**{col}** has **{outliers} outliers** ({pct:.1f}% of values) "
                            f"outside the typical range [{Q1 - 1.5*IQR:.2f}, {Q3 + 1.5*IQR:.2f}]."
                })

    return insights


# ──────────────────────────────────────────────
# 3. AUTO CHART GENERATION
# ──────────────────────────────────────────────

def generate_charts(df, session_id):
    """Auto-generate relevant charts based on column types. Returns list of chart filenames."""
    chart_dir = f'static/agent_{session_id}'
    if not os.path.exists(chart_dir):
        os.makedirs(chart_dir)

    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # 1. Distribution histograms for numeric columns
    for col in numeric_cols[:4]:
        try:
            fig = px.histogram(df, x=col, marginal='box',
                             title=f'Distribution of {col}',
                             color_discrete_sequence=['#6366f1'],
                             template='plotly_white')
            fig.update_layout(bargap=0.05)
            fname = f'dist_{col.replace(" ", "_").lower()}.html'
            fig.write_html(f'{chart_dir}/{fname}')
            charts.append({'file': fname, 'title': f'Distribution: {col}', 'type': 'distribution'})
        except Exception:
            pass

    # 2. Correlation heatmap
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto='.2f',
                          color_continuous_scale='RdBu_r',
                          title='Correlation Matrix',
                          template='plotly_white',
                          aspect='auto')
            fname = 'correlation_matrix.html'
            fig.write_html(f'{chart_dir}/{fname}')
            charts.append({'file': fname, 'title': 'Correlation Matrix', 'type': 'correlation'})
        except Exception:
            pass

    # 3. Scatter plot for top 2 correlated numeric columns
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            np.fill_diagonal(corr_matrix.values, 0)
            max_idx = np.unravel_index(np.abs(corr_matrix.values).argmax(), corr_matrix.shape)
            col_x = corr_matrix.columns[max_idx[0]]
            col_y = corr_matrix.columns[max_idx[1]]

            fig = px.scatter(df, x=col_x, y=col_y, trendline='ols',
                           title=f'{col_x} vs {col_y} (Top Correlation)',
                           template='plotly_white',
                           color_discrete_sequence=['#8b5cf6'])
            fname = 'scatter_top_corr.html'
            fig.write_html(f'{chart_dir}/{fname}')
            charts.append({'file': fname, 'title': f'{col_x} vs {col_y}', 'type': 'scatter'})
        except Exception:
            pass

    # 4. Category bar charts
    for col in cat_cols[:3]:
        try:
            if df[col].nunique() <= 30:
                counts = df[col].value_counts().head(15)
                fig = px.bar(x=counts.index, y=counts.values,
                           labels={'x': col, 'y': 'Count'},
                           title=f'Top Values: {col}',
                           template='plotly_white',
                           color=counts.values,
                           color_continuous_scale='Viridis')
                fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                fname = f'bar_{col.replace(" ", "_").lower()}.html'
                fig.write_html(f'{chart_dir}/{fname}')
                charts.append({'file': fname, 'title': f'Category: {col}', 'type': 'bar'})
        except Exception:
            pass

    # 5. Pie chart for first low-cardinality categorical
    for col in cat_cols:
        if 2 <= df[col].nunique() <= 10:
            try:
                fig = px.pie(df, names=col, title=f'Distribution: {col}',
                           template='plotly_white',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fname = f'pie_{col.replace(" ", "_").lower()}.html'
                fig.write_html(f'{chart_dir}/{fname}')
                charts.append({'file': fname, 'title': f'Pie: {col}', 'type': 'pie'})
                break  # Only one pie
            except Exception:
                pass

    # 6. Time series if date columns exist
    for dcol in date_cols[:1]:
        if numeric_cols:
            try:
                ts_col = numeric_cols[0]
                ts_df = df.dropna(subset=[dcol, ts_col]).sort_values(dcol)
                fig = px.line(ts_df, x=dcol, y=ts_col,
                            title=f'{ts_col} over Time',
                            template='plotly_white',
                            color_discrete_sequence=['#06b6d4'])
                fname = 'timeseries.html'
                fig.write_html(f'{chart_dir}/{fname}')
                charts.append({'file': fname, 'title': f'Time Series: {ts_col}', 'type': 'timeseries'})
            except Exception:
                pass

    # 7. Box plots for numeric by category
    if cat_cols and numeric_cols:
        try:
            best_cat = None
            for c in cat_cols:
                if 2 <= df[c].nunique() <= 10:
                    best_cat = c
                    break
            if best_cat:
                num_col = numeric_cols[0]
                fig = px.box(df, x=best_cat, y=num_col,
                           title=f'{num_col} by {best_cat}',
                           template='plotly_white',
                           color=best_cat,
                           color_discrete_sequence=px.colors.qualitative.Pastel)
                fname = 'boxplot_cat.html'
                fig.write_html(f'{chart_dir}/{fname}')
                charts.append({'file': fname, 'title': f'{num_col} by {best_cat}', 'type': 'box'})
        except Exception:
            pass

    return charts


# ──────────────────────────────────────────────
# 4. TREND PREDICTION
# ──────────────────────────────────────────────

def predict_trends(df):
    """Predict trends for numeric columns. Returns predictions list."""
    predictions = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Time-based trends
    if date_cols and numeric_cols:
        dcol = date_cols[0]
        for ncol in numeric_cols[:3]:
            try:
                ts_df = df[[dcol, ncol]].dropna().sort_values(dcol)
                if len(ts_df) < 10:
                    continue

                # Convert dates to ordinal for regression
                ts_df['date_ord'] = ts_df[dcol].map(pd.Timestamp.toordinal)
                X = ts_df['date_ord'].values.reshape(-1, 1)
                y = ts_df[ncol].values

                # Linear regression
                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                slope = model.coef_[0]

                direction = "increasing" if slope > 0 else "decreasing"
                daily_change = slope  # per day
                strength = "strongly" if abs(r2) > 0.5 else "weakly" if abs(r2) > 0.2 else "very weakly"

                # Forecast next 30 data points
                last_date = ts_df[dcol].max()
                future_dates = pd.date_range(start=last_date, periods=31, freq='D')[1:]
                future_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
                future_pred = model.predict(future_ord)

                predictions.append({
                    'column': ncol,
                    'trend': direction,
                    'strength': strength,
                    'r2': round(r2, 4),
                    'daily_change': round(daily_change, 4),
                    'current_mean': round(y[-10:].mean(), 2),
                    'predicted_mean_30d': round(future_pred.mean(), 2),
                    'text': f"**{ncol}** is {strength} {direction} over time (R² = {r2:.3f}). "
                            f"Currently averaging **{y[-10:].mean():,.2f}**, predicted to reach "
                            f"~**{future_pred.mean():,.2f}** in the next 30 days."
                })
            except Exception:
                pass

    # Row-index trends (if no dates)
    if not date_cols and numeric_cols:
        for ncol in numeric_cols[:3]:
            try:
                series = df[ncol].dropna().reset_index(drop=True)
                if len(series) < 10:
                    continue

                X = np.arange(len(series)).reshape(-1, 1)
                y = series.values

                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                slope = model.coef_[0]

                direction = "increasing" if slope > 0 else "decreasing"
                strength = "strongly" if abs(r2) > 0.5 else "weakly" if abs(r2) > 0.2 else "very weakly"

                # Forecast next 20% of data length
                n_forecast = max(10, int(len(series) * 0.2))
                future_X = np.arange(len(series), len(series) + n_forecast).reshape(-1, 1)
                future_pred = model.predict(future_X)

                predictions.append({
                    'column': ncol,
                    'trend': direction,
                    'strength': strength,
                    'r2': round(r2, 4),
                    'daily_change': round(slope, 4),
                    'current_mean': round(y[-10:].mean(), 2),
                    'predicted_mean_30d': round(future_pred.mean(), 2),
                    'text': f"**{ncol}** shows a {strength} {direction} trend (R² = {r2:.3f}). "
                            f"Recent average: **{y[-10:].mean():,.2f}**, projected next segment: "
                            f"~**{future_pred.mean():,.2f}**."
                })
            except Exception:
                pass

    if not predictions:
        predictions.append({
            'column': 'N/A',
            'trend': 'stable',
            'strength': '',
            'r2': 0,
            'daily_change': 0,
            'current_mean': 0,
            'predicted_mean_30d': 0,
            'text': "Not enough sequential numeric data to detect meaningful trends."
        })

    return predictions


# ──────────────────────────────────────────────
# 4.5. ADVANCED ML (Anomalies & Clustering)
# ──────────────────────────────────────────────

def calculate_data_quality(df_clean, original_shape):
    """Calculate a composite data quality score out of 100."""
    score = 100
    metrics = []
    
    # Missing values penalty
    total_cells = original_shape[0] * original_shape[1]
    missing_cells = df_clean.isna().sum().sum()
    if missing_cells > 0:
        penalty = min(30, (missing_cells / total_cells) * 100 * 2)
        score -= penalty
        metrics.append(f"Missing Values Penalty: -{penalty:.1f} pts")
    else:
        metrics.append("No Missing Values: +0 pts (Perfect)")

    # Duplicates penalty
    dup_ratio = df_clean.duplicated().sum() / len(df_clean)
    if dup_ratio > 0:
        penalty = min(20, dup_ratio * 100 * 1.5)
        score -= penalty
        metrics.append(f"Duplicate Row Penalty: -{penalty:.1f} pts")
    
    # Skewness penalty
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    high_skew_cols = sum(1 for c in numeric_cols if abs(df_clean[c].skew()) > 2)
    if high_skew_cols > 0:
        penalty = min(20, high_skew_cols * 2)
        score -= penalty
        metrics.append(f"High Skewness Penalty ({high_skew_cols} cols): -{penalty:.1f} pts")

    return max(0, int(score)), metrics

def compute_advanced_ml(df, session_id):
    """Run Isolation Forest for anomaly detection and K-Means for clustering."""
    chart_dir = f'static/agent_{session_id}'
    os.makedirs(chart_dir, exist_ok=True)
    
    results = {'anomalies': [], 'cluster_chart': None, 'feature_importance': []}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2 or len(df) < 50:
        return results
        
    try:
        # Preprocessing
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 1. Anomaly Detection (Isolation Forest)
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(X_scaled)
        anomaly_count = (outliers == -1).sum()
        anomaly_pct = (anomaly_count / len(df)) * 100
        
        results['anomalies'] = {
            'count': int(anomaly_count),
            'percentage': round(anomaly_pct, 2),
            'text': f"Detected **{anomaly_count} anomalous rows** ({anomaly_pct:.1f}% of data) using Isolation Forest. These rows deviate significantly from the rest of the dataset."
        }
        
        # 2. Advanced Clustering with PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Auto-determine K using a simple heuristic (between 2 and 5)
        k = min(5, max(2, len(df) // 100))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_plot = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Cluster': [f"Group {c+1}" for c in clusters],
            'Is_Anomaly': ['Anomaly' if o == -1 else 'Normal' for o in outliers]
        })
        
        # Add tooltips if possible
        for i, col in enumerate(numeric_cols[:3]):
            df_plot[col] = df[col].values
            
        fig = px.scatter(df_plot, x='PCA1', y='PCA2', color='Cluster', symbol='Is_Anomaly',
                         title=f'Auto-Detected Segments (K={k}) & Anomalies via PCA',
                         hover_data=numeric_cols[:3],
                         template='plotly_white',
                         color_discrete_sequence=px.colors.qualitative.Bold)
        
        fname = 'advanced_clustering_pca.html'
        fig.write_html(os.path.join(chart_dir, fname))
        results['cluster_chart'] = {'file': fname, 'title': 'Machine Learning: Segment Analysis', 'type': 'pca_cluster'}
        
        # 3. Auto Model Selection & XAI (Predicting the last numeric column using others)
        if len(numeric_cols) >= 3:
            target_col = numeric_cols[-1]
            features = numeric_cols[:-1]
            
            X_ml = X[features]
            y_ml = X[target_col]
            
            # True AutoML: Compare multiple models
            models = {
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'Linear Regression': LinearRegression()
            }
            
            best_model_name = None
            best_model = None
            best_score = -float('inf')
            
            for name, model in models.items():
                model.fit(X_ml, y_ml)
                score = model.score(X_ml, y_ml)
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = model
            
            # Explainable AI (XAI) using Extracted Importances
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
            else:
                importances = np.abs(best_model.coef_)
                if sum(importances) > 0:
                    importances = importances / sum(importances)
                    
            feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:5]
            
            if feat_imp:
                top_feat = feat_imp[0][0]
                top_score = feat_imp[0][1]
                
                results['feature_importance'] = {
                    'target': target_col,
                    'model_used': best_model_name,
                    'r2_score': round(best_score, 3),
                    'importance': [{'feature': f, 'score': f"{round(s * 100, 1)}%"} for f, s in feat_imp],
                    'text': f"Ran **True AutoML** (Random Forest, Gradient Boosting, Linear Regression) to predict **{target_col}**. " + 
                            f"The winning algorithm was **{best_model_name}** (Accuracy: {best_score:.2f}). " +
                            f"**Explainable AI Insights:** **{top_feat}** determines {top_score*100:.1f}% of the outcome."
                }
            
    except Exception as e:
        print(f"Advanced ML failed: {e}")
        pass
        
    return results


def autonomous_hypothesis_testing(df, target_col=None):
    """
    Acts as an Autonomous Hypothesis Generator and Advanced Statistical Testing Suite.
    Formulates a hypothesis about the target metric across binary groups, tests it independently, and returns conclusions.
    """
    if not target_col or target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        # Fall back to picking the highest variance numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None
        target_col = df[numeric_cols].var().idxmax()
        
    results = []
    
    # 1. Search for binary categorical columns to test against target
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool', 'int']).columns
    for col in cat_cols:
        if col == target_col: continue
        unique_vals = df[col].dropna().unique()
        
        # Test if perfectly binary
        if len(unique_vals) == 2:
            group1 = df[df[col] == unique_vals[0]][target_col].dropna()
            group2 = df[df[col] == unique_vals[1]][target_col].dropna()
            
            if len(group1) > 5 and len(group2) > 5:
                stat, p_value = ttest_ind(group1, group2, equal_var=False)
                mean_diff = group1.mean() - group2.mean()
                
                is_significant = p_value < 0.05
                winner = unique_vals[0] if mean_diff > 0 else unique_vals[1]
                
                if is_significant:
                    conclusion = f"**Statistically Significant (p={p_value:.3f}):** The metric **{target_col}** is fundamentally different between **{unique_vals[0]}** and **{unique_vals[1]}**. Specifically, {winner} drives higher numbers."
                else:
                    conclusion = f"The variance of **{target_col}** between **{unique_vals[0]}** and **{unique_vals[1]}** is just random noise (p={p_value:.3f}). Do not build strategies relying on this split."
                    
                results.append({
                    'hypothesis': f"Does [{target_col}] depend on [{col}]?",
                    'test_type': 'Welch’s T-Test',
                    'significant': bool(is_significant),
                    'conclusion': conclusion
                })
                
                # We only need 1 or 2 high-quality hypotheses to avoid overwhelming the user
                if len(results) >= 2:
                    break
                    
    return results

def detect_industry_context(df):
    """
    Context-Aware AI: Detects industry domain based on dataframe headers 
    and returns tailored KPI recommendations & narrative lens.
    """
    cols = " ".join(df.columns).lower()
    
    contexts = {
        'Finance & Banking': {'keywords': ['transaction', 'balance', 'credit', 'account', 'fraud', 'loan', 'salary'], 
                              'kpis': ['Default Risk', 'Transaction Volume', 'Average Balance']},
        'E-Commerce & Retail': {'keywords': ['product', 'price', 'sales', 'revenue', 'customer', 'cart', 'discount'], 
                                'kpis': ['Customer Lifetime Value (CLTV)', 'Conversion Rate', 'Average Order Value']},
        'Healthcare & Medical': {'keywords': ['patient', 'diagnosis', 'blood', 'treatment', 'hospital', 'dose', 'symptom'], 
                                 'kpis': ['Recovery Rate', 'Patient Admission Frequency', 'Treatment Efficacy']},
        'Transportation & Logistics': {'keywords': ['trip', 'origin', 'destination', 'driver', 'fare', 'vehicle', 'mile'], 
                                       'kpis': ['Fleet Utilization', 'Average Delivery Time', 'Cost per Mile']},
        'Human Resources': {'keywords': ['employee', 'salary', 'department', 'manager', 'hire', 'attrition', 'turnover'], 
                            'kpis': ['Employee Churn Rate', 'Average Tenure', 'Salary Band Variance']}
    }
    
    for industry, data in contexts.items():
        if any(kw in cols for kw in data['keywords']):
            return {'industry': industry, 'recommended_kpis': data['kpis']}
            
    return {'industry': 'General / Cross-Industry', 'recommended_kpis': ['Revenue Growth', 'Target Variable Variance', 'Efficiency Ratios']}

# ──────────────────────────────────────────────
# 5. FULL ANALYSIS PIPELINE
# ──────────────────────────────────────────────

def analyze_csv(filepath):
    """
    Full AI agent pipeline:
    1. Load CSV
    2. Clean data
    3. Generate insights
    4. Generate charts
    5. Predict trends

    Returns dict with all results.
    """
    session_id = str(uuid.uuid4())[:8]

    # Load
    df = pd.read_csv(filepath)

    # Clean
    df_clean, cleaning_report = clean_data(df)

    # Calculate Data Quality
    quality_score, quality_metrics = calculate_data_quality(df_clean, df.shape)

    # Summary stats
    summary = {
        'rows': len(df_clean),
        'cols': len(df_clean.columns),
        'columns': df_clean.columns.tolist(),
        'dtypes': {col: str(df_clean[col].dtype) for col in df_clean.columns},
        'missing_total': int(df_clean.isna().sum().sum()),
        'memory_mb': round(df_clean.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'preview': df_clean.head(10).to_dict(orient='records'),
        'quality_score': quality_score,
        'quality_metrics': quality_metrics
    }

    # Insights
    insights = generate_insights(df_clean)

    # Charts
    charts = generate_charts(df_clean, session_id)

    # Trends
    predictions = predict_trends(df_clean)
    
    # Advanced ML (Anomalies & Clusters)
    advanced_ml = compute_advanced_ml(df_clean, session_id)
    if advanced_ml and 'cluster_chart' in advanced_ml and advanced_ml['cluster_chart']:
        charts.append(advanced_ml['cluster_chart'])
        
    # Context-Aware AI & Automated Statistical Testing (Startup-Grade Features)
    industry_context = detect_industry_context(df_clean)
    
    target_var = None
    if advanced_ml and advanced_ml.get('feature_importance'):
        target_var = advanced_ml['feature_importance']['target']
        
    hypotheses = autonomous_hypothesis_testing(df_clean, target_var)
    if hypotheses:
        advanced_ml['hypotheses'] = hypotheses
        
    # AI Executive Summary Generator & Data Storytelling
    is_good = quality_score > 70
    biz_text = f"Analyzed a **{industry_context['industry']}** dataset with {len(df_clean):,} records. Overall data quality is {'excellent' if is_good else 'poor'} (Score: {quality_score}/100). "
    
    if target_var:
        biz_text += f"Key driver analysis reveals that **{advanced_ml['feature_importance']['importance'][0]['feature']}** is the primary root cause influencing **{target_var}**. "
        
    if hypotheses and [h for h in hypotheses if h['significant']]:
        sig_hypo = [h for h in hypotheses if h['significant']][0]
        biz_text += f"We autonomously tested and proved: {sig_hypo['conclusion'].split(':** ')[1]} "
        
    biz_text += f"**Decision Engine Recommendation:** Given this is a {industry_context['industry']} context, your teams should immediately build dashboards monitoring {', '.join(industry_context['recommended_kpis'])}."
        
    tech_text = f"Processed {len(df_clean)}x{len(df_clean.columns)} dimension matrix. Imputed {summary['missing_total']} missing values. "
    if target_var:
        tech_text += f"AutoML selected {advanced_ml['feature_importance']['model_used']} (R²={advanced_ml['feature_importance']['r2_score']}). Extracted SHAP-like feature weights. "
    if advanced_ml and advanced_ml.get('anomalies'):
        tech_text += f"Unsupervised Isolation Forest detected {advanced_ml['anomalies']['percentage']}% high-dimensional outliers. PCA mapping and K-Means segmentation executed."

    executive_summary = {
        'industry_context': industry_context['industry'],
        'business': biz_text,
        'technical': tech_text
    }

    return {
        'session_id': session_id,
        'summary': summary,
        'cleaning_report': cleaning_report,
        'insights': insights,
        'charts': charts,
        'predictions': predictions,
        'advanced_ml': advanced_ml,
        'executive_summary': executive_summary
    }

