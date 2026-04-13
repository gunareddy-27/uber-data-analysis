import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier, IsolationForest
)
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
    classification_report, mean_squared_error
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

# ─── Model Paths ───
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


def _ensure_models_dir():
    os.makedirs('models', exist_ok=True)


# ═══════════════════════════════════════════════
# DATA LOADING & ADVANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # Parse dates
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')
    df['END_DATE'] = pd.to_datetime(df['END_DATE'], errors='coerce')

    # Drop rows with missing critical columns
    df.dropna(subset=['START_DATE', 'END_DATE', 'MILES'], inplace=True)

    # ── Core Time Features ──
    df['Hour'] = df['START_DATE'].dt.hour
    df['Weekday'] = df['START_DATE'].dt.dayofweek
    df['Month'] = df['START_DATE'].dt.month
    df['DayOfMonth'] = df['START_DATE'].dt.day
    df['Trip_Duration'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 60

    # ── Advanced Engineered Features ──
    # Cyclic encoding for hour (captures 23→0 continuity)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    # Cyclic encoding for weekday
    df['Weekday_Sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['Weekday_Cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

    # Cyclic encoding for month
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Interaction features
    df['Miles_x_Hour'] = df['MILES'] * df['Hour']
    df['Miles_x_Weekday'] = df['MILES'] * df['Weekday']
    df['Hour_Sq'] = df['Hour'] ** 2
    df['Miles_Sq'] = df['MILES'] ** 2
    df['Log_Miles'] = np.log1p(df['MILES'])

    # Rush hour flag
    df['Is_Rush_Hour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9) |
                          (df['Hour'] >= 16) & (df['Hour'] <= 19)).astype(int)

    # Weekend flag
    df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)

    # Time of day bucket (Morning/Afternoon/Evening/Night)
    df['Time_Bucket'] = pd.cut(df['Hour'], bins=[-1, 6, 12, 18, 24],
                               labels=[0, 1, 2, 3]).astype(int)

    # Remove extreme outliers (> 5 hours or negative)
    df = df[(df['Trip_Duration'] > 0) & (df['Trip_Duration'] < 300)]

    print(f"✅ Data prepared: {len(df)} rows, {len(df.columns)} features")
    return df


# ═══════════════════════════════════════════════
# MODEL 1: TRIP DURATION (Ensemble Regressor)
# ═══════════════════════════════════════════════

def train_trip_duration_model(df):
    print("🔧 Training Trip Duration Model (Advanced Ensemble)...")

    FEATURE_COLS = [
        'MILES', 'Hour', 'Weekday', 'Month',
        'Hour_Sin', 'Hour_Cos', 'Weekday_Sin', 'Weekday_Cos',
        'Miles_x_Hour', 'Miles_x_Weekday', 'Hour_Sq', 'Miles_Sq',
        'Log_Miles', 'Is_Rush_Hour', 'Is_Weekend', 'Time_Bucket'
    ]

    df_model = df[FEATURE_COLS + ['Trip_Duration']].dropna()

    X = df_model[FEATURE_COLS]
    y = df_model['Trip_Duration']

    # Proper train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensemble: GradientBoosting + RandomForest Voting
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    # Voting Ensemble (weighted average of both models)
    ensemble = VotingRegressor(
        estimators=[('gb', gb_model), ('rf', rf_model)],
        weights=[0.6, 0.4]  # GB gets more weight (usually better)
    )

    ensemble.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = ensemble.predict(X_train_scaled)
    y_pred_test = ensemble.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test  R²: {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.2f} min")
    print(f"  Test RMSE: {test_rmse:.2f} min")

    # Cross-validation score
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Compute Prediction Interval Statistics (for research-grade uncertainty)
    # Using RMSE as a proxy for prediction interval width (assuming normality of residuals)
    prediction_std_err = test_rmse
    
    # [RESEARCH] Model Benchmarking Engine
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    
    print("🔬 Benchmarking against Baseline & Neural Models...")
    
    # 1. Baseline: Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_r2 = r2_score(y_test, ridge.predict(X_test_scaled))
    
    # 2. Neural: MLP Regressor
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.01, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    mlp_r2 = r2_score(y_test, mlp.predict(X_test_scaled))
    
    # Save benchmark 
    metrics['benchmarks'] = {
        'ensemble_r2': round(test_r2, 4),
        'linear_baseline_r2': round(ridge_r2, 4),
        'neural_network_r2': round(mlp_r2, 4),
        'neural_upside': round(test_r2 - mlp_r2, 4)
    }

    # Save metadata with all metrics
    _ensure_models_dir()
    with open(MODEL_PATH, 'wb') as f:
        payload = {
            'model': ensemble,
            'prediction_std_err': test_rmse,
            'feature_names': FEATURE_COLS,
            'metrics': metrics
        }
        pickle.dump(payload, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    return ensemble, scaler, metrics

def run_counterfactual(input_df, model_payload, scaler, target_feature, alternative_value):
    """
    [RESEARCH] Causal Counterfactual Analysis.
    Predicts 'What if?' by shifting a single feature and comparing the result.
    """
    try:
        model = model_payload['model']
        feature_names = model_payload['feature_names']
        
        # Original Prediction
        X_base = scaler.transform(input_df)
        base_pred = model.predict(X_base)[0]
        
        # Modified Prediction
        input_mod = input_df.copy()
        input_mod[target_feature] = alternative_value
        
        # Recompute dependent features (e.g. if we changed Hour, we must change Hour_Sin/Cos)
        if target_feature == 'Hour':
            hour = alternative_value
            input_mod['Hour_Sin'] = np.sin(2 * np.pi * hour / 24)
            input_mod['Hour_Cos'] = np.cos(2 * np.pi * hour / 24)
            input_mod['Miles_x_Hour'] = input_df['MILES'] * hour
            input_mod['Hour_Sq'] = hour ** 2
            
        X_mod = scaler.transform(input_mod)
        mod_pred = model.predict(X_mod)[0]
        
        diff = mod_pred - base_pred
        return {
            'original': round(float(base_pred), 2),
            'counterfactual': round(float(mod_pred), 2),
            'impact': round(float(diff), 2),
            'percentage_shift': round(float(diff/base_pred*100), 1)
        }
    except:
        return None
    """
    [XAI] Research Feature: Sensitivity Analysis.
    Perturbs each feature to see its individual contribution to the final prediction.
    """
    try:
        model = model_payload['model']
        feature_names = model_payload['feature_names']
        
        # Scaling the input
        X_base = scaler.transform(input_df)
        base_pred = model.predict(X_base)[0]
        
        explanations = []
        for i, col in enumerate(feature_names):
            # Perturb feature by 10%
            X_mod = X_base.copy()
            X_mod[0, i] *= 1.1 # 10% shift
            mod_pred = model.predict(X_mod)[0]
            
            diff = mod_pred - base_pred
            explanations.append({
                'feature': col,
                'weight': round(float(diff), 4),
                'direction': 'increases' if diff > 0 else 'decreases'
            })
            
        # Top 3 drivers
        top_drivers = sorted(explanations, key=lambda x: abs(x['weight']), reverse=True)[:3]
        return top_drivers
    except:
        return []

def run_monte_carlo_duration(input_df, model_payload, scaler, n_iter=1000):
    """
    [RESEARCH] Probabilistic Forecasting.
    Uses Monte Carlo sampling based on model variance to produce a confidence distribution.
    """
    try:
        model = model_payload['model']
        std_err = model_payload.get('prediction_std_err', 3.6)
        
        X_scaled = scaler.transform(input_df)
        point_estimate = model.predict(X_scaled)[0]
        
        # Sample from normal distribution centered at prediction
        samples = np.random.normal(point_estimate, std_err, n_iter)
        samples = np.maximum(samples, point_estimate * 0.1) # Duration must be > 0
        
        return {
            'mean': round(float(point_estimate), 2),
            'lower_95': round(float(np.percentile(samples, 2.5)), 2),
            'upper_95': round(float(np.percentile(samples, 97.5)), 2),
            'probability_mass': [float(s) for s in np.histogram(samples, bins=10)[0]]
        }
    except:
        return None


# ═══════════════════════════════════════════════
# MODEL 2: DEMAND PREDICTION
# ═══════════════════════════════════════════════

def train_demand_model(df):
    print("🔧 Training Demand Prediction Model (Advanced)...")

    # Aggregate demand by Hour + Weekday + Month
    df_demand = df.groupby(['Hour', 'Weekday', 'Month']).agg(
        Trip_Count=('Trip_Duration', 'size'),
        Avg_Miles=('MILES', 'mean'),
        Avg_Duration=('Trip_Duration', 'mean')
    ).reset_index()

    # Add engineered features
    df_demand['Hour_Sin'] = np.sin(2 * np.pi * df_demand['Hour'] / 24)
    df_demand['Hour_Cos'] = np.cos(2 * np.pi * df_demand['Hour'] / 24)
    df_demand['Weekday_Sin'] = np.sin(2 * np.pi * df_demand['Weekday'] / 7)
    df_demand['Weekday_Cos'] = np.cos(2 * np.pi * df_demand['Weekday'] / 7)
    df_demand['Is_Rush_Hour'] = ((df_demand['Hour'] >= 7) & (df_demand['Hour'] <= 9) |
                                  (df_demand['Hour'] >= 16) & (df_demand['Hour'] <= 19)).astype(int)
    df_demand['Is_Weekend'] = (df_demand['Weekday'] >= 5).astype(int)

    FEATURES = ['Hour', 'Weekday', 'Month', 'Hour_Sin', 'Hour_Cos',
                'Weekday_Sin', 'Weekday_Cos', 'Is_Rush_Hour', 'Is_Weekend']

    X = df_demand[FEATURES]
    y = df_demand['Trip_Count']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.85,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics = {
        'r2': round(test_r2, 4),
        'mae': round(test_mae, 2),
        'rmse': round(test_rmse, 2),
        'samples': len(df_demand)
    }
    print(f"  Demand Model Test R²: {test_r2:.4f}")
    print(f"  Demand Model Test MAE: {test_mae:.2f}")

    # Location Intelligence: best START per (Hour, Weekday)
    loc_demand = df.groupby(['Hour', 'Weekday', 'START']).size().reset_index(name='Frequency')
    best_locs = loc_demand.sort_values(
        ['Hour', 'Weekday', 'Frequency'], ascending=[True, True, False]
    )
    best_locs = best_locs.drop_duplicates(subset=['Hour', 'Weekday'], keep='first')

    # Save
    _ensure_models_dir()
    with open(DEMAND_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'metrics': metrics}, f)
    with open('models/best_locations_lookup.pkl', 'wb') as f:
        pickle.dump(best_locs, f)

    return model, best_locs, metrics

    return model, best_locs


# ═══════════════════════════════════════════════
# MODEL 3: CLASSIFICATION (Category & Purpose)
# ═══════════════════════════════════════════════

def train_classification_models(df):
    print("🔧 Training Classification Models (Category & Purpose)...")

    df_clf = df.dropna(subset=['CATEGORY', 'PURPOSE', 'MILES'])

    FEATURE_COLS = [
        'MILES', 'Hour', 'Weekday', 'Month',
        'Hour_Sin', 'Hour_Cos', 'Weekday_Sin', 'Weekday_Cos',
        'Miles_x_Hour', 'Log_Miles', 'Is_Rush_Hour', 'Is_Weekend', 'Time_Bucket'
    ]

    X = df_clf[FEATURE_COLS]

    # ── Category Model (GradientBoosting) ──
    le_cat = LabelEncoder()
    y_cat = le_cat.fit_transform(df_clf['CATEGORY'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    model_cat = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.85,
        random_state=42
    )
    model_cat.fit(X_train, y_train)

    cat_train_acc = accuracy_score(y_train, model_cat.predict(X_train))
    cat_test_acc = accuracy_score(y_test, model_cat.predict(X_test))
    print(f"  Category Train Accuracy: {cat_train_acc:.4f}")
    print(f"  Category Test  Accuracy: {cat_test_acc:.4f}")

    # ── Purpose Model (Ensemble) ──
    le_purp = LabelEncoder()
    y_purp = le_purp.fit_transform(df_clf['PURPOSE'])

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X, y_purp, test_size=0.2, random_state=42, stratify=y_purp
    )

    gb_clf = GradientBoostingClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42
    )
    rf_clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    model_purp = VotingClassifier(
        estimators=[('gb', gb_clf), ('rf', rf_clf)],
        voting='soft',
        weights=[0.6, 0.4]
    )
    model_purp.fit(X_train_p, y_train_p)

    from sklearn.metrics import precision_recall_fscore_support
    
    def get_clf_metrics(y_true, y_pred):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        return {'accuracy': round(acc, 4), 'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f, 4)}

    cat_metrics = get_clf_metrics(y_test, model_cat.predict(X_test))
    purp_metrics = get_clf_metrics(y_test_p, model_purp.predict(X_test_p))

    print(f"  Category Test  Accuracy: {cat_metrics['accuracy']}")
    print(f"  Purpose Test  Accuracy: {purp_metrics['accuracy']}")

    # Save
    _ensure_models_dir()
    with open(CATEGORY_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model_cat, 'metrics': cat_metrics}, f)
    with open(PURPOSE_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model_purp, 'metrics': purp_metrics}, f)
    with open(CATEGORY_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_cat, f)
    with open(PURPOSE_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_purp, f)

    return model_cat, model_purp, le_cat, le_purp, {'category': cat_metrics, 'purpose': purp_metrics}


# ═══════════════════════════════════════════════
# MODEL 4: LOCATION PREDICTION (Smart Routing)
# ═══════════════════════════════════════════════

def train_location_model(df):
    print("🔧 Training Location Prediction Model (Advanced)...")

    df_loc = df.dropna(subset=['STOP', 'Hour', 'Weekday', 'MILES', 'PURPOSE', 'CATEGORY'])

    # Encode targets and features
    le_stop = LabelEncoder()
    y = le_stop.fit_transform(df_loc['STOP'].astype(str))

    le_cat = LabelEncoder()
    X_cat = le_cat.fit_transform(df_loc['CATEGORY'].astype(str))

    le_purp = LabelEncoder()
    X_purp = le_purp.fit_transform(df_loc['PURPOSE'].astype(str))

    # Extended Feature Set
    X = pd.DataFrame({
        'Hour': df_loc['Hour'].values,
        'Weekday': df_loc['Weekday'].values,
        'Month': df_loc['Month'].values,
        'MILES': df_loc['MILES'].values,
        'CATEGORY_ENCODED': X_cat,
        'PURPOSE_ENCODED': X_purp,
        'Hour_Sin': df_loc['Hour_Sin'].values,
        'Hour_Cos': df_loc['Hour_Cos'].values,
        'Is_Rush_Hour': df_loc['Is_Rush_Hour'].values,
        'Is_Weekend': df_loc['Is_Weekend'].values,
        'Log_Miles': df_loc['Log_Miles'].values,
    })

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.85,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Location Train Accuracy: {train_acc:.4f}")
    print(f"  Location Test  Accuracy: {test_acc:.4f}")

    # Save
    _ensure_models_dir()
    with open(LOCATION_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(STOP_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_stop, f)
    with open(LOC_CAT_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_cat, f)
    with open(LOC_PURP_ENCODER_PATH, 'wb') as f:
        pickle.dump(le_purp, f)

    return model, le_stop, le_cat, le_purp


# ═══════════════════════════════════════════════
# MODEL 5: ANOMALY DETECTION
# ═══════════════════════════════════════════════

def train_anomaly_detection(df):
    print("🔧 Training Anomaly Detection Model (Isolation Forest)...")

    feature_cols = ['MILES', 'Trip_Duration', 'Hour', 'Weekday']
    X = df[feature_cols].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.06,
        n_estimators=200,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    anomaly_count = (model.predict(X_scaled) == -1).sum()
    print(f"  Anomalies detected: {anomaly_count} / {len(X)} ({anomaly_count/len(X)*100:.1f}%)")

    _ensure_models_dir()
    with open(ANOMALY_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': feature_cols}, f)

    return model


# ═══════════════════════════════════════════════
# MODEL 6: TIME-SERIES FORECAST (ARIMA)
# ═══════════════════════════════════════════════

def train_time_series_forecast(df):
    print("🔧 Training Time-Series Forecast Model (ARIMA)...")
    if ARIMA is None:
        print("  ⚠ Statsmodels not installed. Skipping.")
        return None

    df_ts = df.copy()
    df_ts['Date'] = df_ts['START_DATE'].dt.date
    series = df_ts.groupby('Date').size()

    if len(series) < 10:
        print("  ⚠ Not enough time-series data. Skipping.")
        return None

    try:
        # Try multiple ARIMA orders and pick the best (low AIC)
        best_aic = float('inf')
        best_model = None
        best_order = (5, 1, 0)

        for p in [3, 5, 7]:
            for d in [0, 1]:
                for q in [0, 1, 2]:
                    try:
                        m = ARIMA(series.values, order=(p, d, q))
                        fit = m.fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_model = fit
                            best_order = (p, d, q)
                    except Exception:
                        continue

        if best_model is None:
            model = ARIMA(series.values, order=(5, 1, 0))
            best_model = model.fit()
            best_order = (5, 1, 0)

        print(f"  Best ARIMA order: {best_order}, AIC: {best_aic:.2f}")

        _ensure_models_dir()
        with open(FORECAST_MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        return best_model

    except Exception as e:
        print(f"  ⚠ ARIMA Training failed: {e}")
        return None


# ═══════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════

def load_models():
    models = {}

    def load_pickle(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, Exception):
            return None

    def extract_model_and_metrics(data, suffix):
        if isinstance(data, dict):
            models[f'{suffix}_model'] = data.get('model')
            models[f'{suffix}_metrics'] = data.get('metrics')
            if 'prediction_std_err' in data: models[f'{suffix}_std_err'] = data['prediction_std_err']
            if 'feature_names' in data: models[f'{suffix}_features'] = data['feature_names']
            return True
        return False

    # 1. Duration Model
    duration_data = load_pickle(MODEL_PATH)
    if not extract_model_and_metrics(duration_data, 'duration'):
        models['duration_model'] = duration_data
    models['duration_scaler'] = load_pickle(SCALER_PATH)
    print("  Loaded Duration Model.")

    # 2. Location Model
    models['location_model'] = load_pickle(LOCATION_MODEL_PATH)
    models['stop_encoder'] = load_pickle(STOP_ENCODER_PATH)
    models['loc_cat_encoder'] = load_pickle(LOC_CAT_ENCODER_PATH)
    models['loc_purp_encoder'] = load_pickle(LOC_PURP_ENCODER_PATH)
    print("  Loaded Location Model.")

    # 3. Anomaly & Forecast
    anomaly_data = load_pickle(ANOMALY_MODEL_PATH)
    if not extract_model_and_metrics(anomaly_data, 'anomaly'):
        models['anomaly_model'] = anomaly_data
    models['forecast_model'] = load_pickle(FORECAST_MODEL_PATH)
    print("  Loaded Anomaly and Forecast Models.")

    # 4. Demand
    demand_data = load_pickle(DEMAND_MODEL_PATH)
    if not extract_model_and_metrics(demand_data, 'demand'):
        models['demand_model'] = demand_data
    models['best_locations'] = load_pickle('models/best_locations_lookup.pkl')
    print("  Loaded Demand and Revenue Optimizer Models.")

    # 5. Classification
    cat_data = load_pickle(CATEGORY_MODEL_PATH)
    if not extract_model_and_metrics(cat_data, 'category'):
        models['category_model'] = cat_data
        
    purp_data = load_pickle(PURPOSE_MODEL_PATH)
    if not extract_model_and_metrics(purp_data, 'purpose'):
        models['purpose_model'] = purp_data
        
    models['category_encoder'] = load_pickle(CATEGORY_ENCODER_PATH)
    models['purpose_encoder'] = load_pickle(PURPOSE_ENCODER_PATH)
    print("  Loaded Classification Models.")

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
        print("\n🎯 All models trained successfully!")
    else:
        print("Dataset not found!")
