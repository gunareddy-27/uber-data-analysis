# 🚖 Uber Analytics Pro & AI Data Analyst Agent

![Uber Analytics Pro Overview](https://img.shields.io/badge/Status-Active-success)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-black)
![scikit-learn](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange)

**Uber Analytics Pro** is a comprehensive, AI-powered Flask web application designed to analyze transportation datasets, predict ride-sharing trends, and act as a fully autonomous **AI Data Analyst Agent** capable of processing and analyzing *any* uploaded CSV file. 

The application uses a modern **Tailwind CSS** Single Page Application (SPA) architecture combined with a robust Python Machine Learning backend.

---

## ✨ Key Features

### 1. 🤖 AI Data Analyst Agent (Auto-Insights Generator)
Upload any CSV (max 50MB) and let the autonomous AI agent to completely analyze the dataset in seconds:
- **Auto Data Cleaning:** Intelligently imputes missing values (median/mode), detects datetimes, and removes duplicated or entirely empty rows.
- **Plain English Insights:** Analyzes data distributions, extreme skewness, top correlates, data quality issues, and cardinality in human-readable language.
- **Auto Data Visualization:** Generates relevant dynamic Plotly charts (Histograms, Correlation Heatmaps, Scatter Plots, Pie Charts, and Time Series).
- **Trend Predictions:** Harnesses Linear Regressions to analyze directional trends over time and forecast a 30-day projection.
- **PDF Export:** Click **"Download Report PDF"** for a fully formatted, print-ready document of the agent's findings.

**Advanced ML Capabilities within the Agent:**
- **Data Quality Scoring:** Algorithmically scores the dataset (out of 100) based on duplicates, missing values, and skewness penalties.
- **Isolation Forest Anomaly Detection:** Statistically detects and isolates high-dimensional outliers within the data.
- **PCA + K-Means Segmentation:** Compresses datasets into a 2D PCA space and automatically detects natural groupings (clusters).
- **Random Forest Auto-ML:** Selects a target variable and trains a Random Forest model on the fly to determine the Top 5 most important predictive features.

### 2. 🏢 Ride-Sharing Data Warehouse (Uber Specific)
A dedicated module for analyzing Uber/Ride-sharing logistics:
- **Revenue Simulation:** Estimates trip fares based on distance and dynamically simulated time-of-day surge multipliers.
- **Geographic Tracking:** Tracks total revenue broken down by starting City.
- **Demand & Peak Hour Analytics:** Generates demand heatmaps (Hour x Weekday), Surge Pricing Trends, and dynamic Revenue Sunburst visualizations.

### 3. 🧠 AI Trip Predictor
Predictive Machine Learning models explicitly trained on your ride data:
- **Duration Predictor:** Predicts travel time (minutes) based on mileage and the time of the week.
- **Demand Forecaster:** Predicts overall platform ride volume based on the active hour and day.
- **Intent Classifier:** Classifies if an upcoming trip is for Business/Personal and matches the probable Category.
- **Smart Routing Predictor:** Suggests the probable Destination of a ride based on starting time and initial details.

### 4. 📊 Dashboard & Data View 
- **Real-Time KPIs:** Tracks top-level metrics like Total Trips, Mileage, Average Speed, Busiest Hours, and Estimated Tax Deductibles.
- **Live Filtering & Exporting:** Instantly query trip data by date ranges, categories, and purposes — and export back to CSV or JSON format.

---

## 📂 Modules & Architecture

### Backend (Python)
- `app.py` — The core Flask backend. Handles application routing, file uploads, API endpoints (`/api/warehouse`, `/api/agent/analyze`), user authentication, and orchestrates the ML models.
- `ai_agent.py` — Contains the state-of-the-art Auto Insights pipeline. Handles dataset cleaning, Plotly auto-generation, ML models (`IsolationForest`, `PCA`, `KMeans`, `RandomForestRegressor`), and JSON payload packaging for the frontend.
- `warehouse.py` — The Ride-Sharing Data Warehousing layer. Processes Uber data, estimates simulated base fares, calculates surge modifiers, and renders specialized Plotly HTML figures.
- `predictor.py` — The centralized training module. Prepares data, trains, and serializes the 5 foundational AI Predictor models to memory whenever new data is added.
- `visualization.py` & `analysis.py` — Core statistical libraries for plotting fundamental hourly, daily, and monthly trip frequencies inside the main Dashboard.
- `clustering.py` — Supports auxiliary trip demographic clustering routines (spatial-temporal grouping).

### Frontend (HTML / JS / CSS)
- `templates/spa.html` — A fully responsive, dark-mode equipped, Tailwind CSS Single Page Application. It uses JavaScript DOM manipulation for seamless tab switching, AJAX asynchronous `fetch` requests for the AI Agent processing, and embeds dynamic interactive iframes designed specifically for Plotly outputs. 

---

## 🚀 Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Setup
Clone the repository, create a virtual environment, and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/gunareddy-27/uber-data-analysis.git
cd uber-data-analysis

# Install dependencies
pip install -r requirements.txt

# Or manually install core packages:
pip install flask flask-sqlalchemy flask-login pandas numpy scikit-learn plotly statsmodels werkzeug
```

### 3. Running the App
Initiate the Flask server. Model training happens automatically on initial load:

```bash
python app.py
```
**Access the dashboard globally at:** `http://127.0.0.1:5000`

---

## 🔒 Security
- Data uploads are restricted to a `50MB` threshold enforced natively in Flask to protect server RAM.
- Secure filename handling (`werkzeug.utils.secure_filename`) prevents directory traversal attacks during the AI Agent upload phase.
- Built-in session tracking via `Flask-Login` protects analysis dashboards strictly for authenticated users.

---

*Powered by Python, Scikit-Learn, Plotly, and Flask.*
