# 🚖 Uber Analytics Pro & AI Data Analyst Agent

![Uber Analytics Pro Overview](https://img.shields.io/badge/Status-Active-success)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-black)
![scikit-learn](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange)

**Uber Analytics Pro** is a comprehensive, AI-powered Flask web application designed to analyze transportation datasets, predict ride-sharing trends, and act as a fully autonomous **AI Data Analyst Agent** capable of processing and analyzing *any* uploaded CSV file. 

The application uses a modern **Tailwind CSS** Single Page Application (SPA) architecture combined with a robust Python Machine Learning backend.

---

## ✨ Key Features

### 1. 🤖 Uber Analytics Pro: Autonomous AI Data Agent
The application has been upgraded to a fully **Autonomous Multi-Agent System** that runs a zero-click data intelligence pipeline:
- **Autonomous Multi-Agent Orchestration:** Deploys specialized sub-agents for **Cleaning, AutoML, Anomaly Detection, Hypothesis Testing, and Executive Reporting** in a coordinated sequence.
- **One-Click Autonomous Pipeline:** A complete data-to-insight lifecycle with a visual progress tracker, status updates, and stage-by-side execution logic.
- **Smart Alerting & Anomaly Center:** Real-time monitoring using unsupervised `IsolationForest`. Detects high-dimensional outliers and triggers pulse-animated UI alerts in a dedicated Notification Center.
- **Predictive Business Scenario Simulation:** Interactive "What-if" simulations (e.g., "Increase demand by 20%") that project revenue impacts and platform efficiency using predictive regression models.
- **Continuous Learning Feedback Loop:** Users can rate agent insights (👍/👎), which are logged and used by the agent to fine-tune future recommendation weights.
- **Run Versioning & Drift Detection:** Automatically compares new datasets against historical baselines stored in JSON metadata logs to detect statistical drift and volume anomalies.
- **Advanced AutoML & XAI:** Compares multiple algorithms (Random Forest, Gradient Boosting) and uses **Explainable AI** to reveal the top 5 drivers behind any prediction.
- **Autonomous Hypothesis Engine:** Formulates and runs Welch's T-Tests across categorical groups to prove statistical significance (p-values) automatically.
- **Comprehensive Data Documentation:** Auto-generates a full Data Dictionary with inferred column meanings and cardinality assessments.
- **Data Quality Scoring:** Compiles a 1-100 quality score based on duplicates, missing values, and distribution skewness.

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
- `ai_agent.py` — Contains the state-of-the-art Auto Insights pipeline. Handles dataset cleaning, AutoML iteration (`GradientBoosting`, `RandomForest`, `LinearRegression`), `IsolationForest`, `PCA`/`KMeans`, autonomous `scipy` Hypothesis testing, and JSON payload packaging for the frontend.
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

# Install dependencies (pinned to stable NumPy 1.x)
pip install -r requirements.txt

# Note: NumPy < 2.0 is required to maintain compatibility with legacy binary wheels.
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
