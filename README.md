# 🌫️ PEARLS AQI PREDICTOR

### End-to-End Automated AQI Forecasting System (MLOps Project)

🔗 Live App:
https://projcet-mot8ocxkoihxmufu4xnbhn.streamlit.app/

---

## 🚀 Project Overview

I built a fully automated Machine Learning system that predicts the next **72 hours of Air Quality Index (AQI)** without running manually.

Unlike basic ML projects that only train a model once, this system:

* Collects real-time pollution data from live API
* Updates features every hour in feature store
* Retrains models daily in model registry
* Automatically selects the best model
* Deploys predictions to a live web app

Everything runs automatically using CI/CD pipelines.

---

## 🏗️ Complete System Architecture

Air Quality API
→ Hourly Feature Pipeline
→ MongoDB Feature Store
→ Data Validation
→ Daily Training Pipeline
→ Model Evaluation & Comparison
→ Model Registry
→ GitHub Repository
→ CI/CD (GitHub Actions)
→ Automated Deployment
→ Streamlit Web App
→ End User sees AQI Forecast

---

## 🔄 Feature Pipeline (Runs Every Hour)

* Fetches real-time pollution data
* Cleans missing values
* Creates:

  * Lag features (1h, 3h, 6h)
  * Rolling averages
  * Time-based features
* Stores processed features in MongoDB Atlas

This ensures training always uses fresh data, and predictions are based on the latest environmental conditions.

---

## 🌧️ Advanced Rain Impact Modeling (Extra Feature Added)

In addition to standard AQI forecasting, I implemented a **Rain Impact Analysis Feature**, which makes this project more realistic and environment-aware.

### What I Added:

* Trained models using rain-related environmental data
* Integrated atmospheric molecule behavior patterns
* Compared AQI during:

  * Rain conditions
  * Non-rain conditions
* Built logic to show how rainfall improves AQI

### Why This Is Important:

During rain, airborne dust particles and pollutants settle down due to water droplets binding with pollution molecules. This naturally reduces PM2.5 concentration.

Instead of just showing prediction, the system:

* Demonstrates how weather directly impacts pollution
* Compares AQI with rain vs without rain
* Provides practical environmental insights
* Makes the forecasting system more intelligent and real-world aware

This feature shows understanding beyond basic ML — it reflects environmental modeling and contextual prediction.

Recruiter Impact:
This demonstrates the ability to integrate domain knowledge (weather + air chemistry) into machine learning systems rather than building a generic forecasting model.

---

## 🧠 Training Pipeline (Runs Daily)

* Loads latest features from MongoDB Atlas
* Trains multiple models:

  * Linear Regression
  * Ridge Regression
  * Random Forest
  * Gradient Boosting
* Compares performance using:

  * RMSE
  * MAE
  * R²
* Automatically selects best model
* Saves it in Model Registry with timestamp

No manual retraining needed.

---

## 📦 Model Registry

* Stores models
* Keeps track of best-performing model
* Ensures reproducibility

---

## ⚙️ CI/CD Automation (GitHub Actions)

* Hourly feature updates
* Daily model retraining
* Code testing
* Automatic deployment

---

## 🌐 Streamlit Web Application

* Loads latest trained model
* Fetches latest features
* Predicts next 72 hours AQI
* Displays interactive charts
* Converts PM2.5 to official AQI values

There are total 6 tabs included in the application:

### Tab 1: Overview

* Shows overall AQI of Karachi.
* Displays dust particle values (PM2.5, PM10) and their conversion into AQI.
* Provides health precautions according to the current AQI level.
* Shows special advice for sensitive groups based on the current AQI.

### Tab 2: Live Map

* Displays a live map of Karachi with different areas.
* Shows color indicators on the map to represent AQI levels (e.g., red means poor AQI).
* Users can move the mouse over a specific area (e.g., Shahrah-e-Faisal) to check its AQI value.

### Tab 3: Forecast

* Predicts the overall AQI of Karachi.
* Shows a 3-day AQI forecast automatically.
* Displays the previous month’s AQI for comparison.
* Includes a slider to check prediction for 1, 2, or 3 days.

### Tab 4: Area Comparison

* Shows AQI for different regions of Karachi.
* Since Karachi is a large city, AQI may vary by region.
* Users can select a region from a dropdown menu to view its AQI on the map.

### Tab 5: Health & Alerts

* Shows the practical impact of environmental conditions like rain on AQI.
* Compares AQI with rain and without rain.
* Uses trained AQI calculation engine and environmental data.
* Displays rain forecast areas and their effect on air quality.

### Tab 6: Model Insights

* Shows different trained models.
* Provides better visualization for model performance and comparison.

---

## 🛠 Tech Stack

Python
Scikit-learn
MongoDB Atlas
Streamlit
GitHub Actions
Open-Meteo API

---

## 💡 Why This Project Is Strong

* End-to-end ML system (not just model training)
* Feature Store implementation
* Model Registry with versioning
* Automated retraining
* CI/CD pipeline
* Live cloud deployment
* Real-time + forecasting
* Weather-aware AQI modeling
* Environmental impact comparison (Rain vs Non-Rain AQI)

This gives me complete understanding of:

* Machine Learning
* MLOps
* Automation and CI/CD pipelines
* Pipeline architecture and running different pipelines
* Cloud-based architecture
* Production systems
* Practical solving problem by doing this project

---

## 👩‍💻 About Me

Sadia Ali
LinkedIn Profile: https://www.linkedin.com/in/sadia-ali-ce/


