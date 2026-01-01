# Flood Prediction Web App

A Streamlit-based web application for real-time flood risk prediction using a hybrid machine learning model combining **LSTM** (Long Short-Term Memory) neural network and **Markov Chain** probabilistic modeling.

This app fetches recent water level data from a Supabase database, analyzes historical patterns, forecasts flood risk for the next 12 hours, visualizes past water levels, and sends automated email alerts when a flood is predicted.

## Features

- Secure login page (username/password protected)
- Real-time water level data retrieval from Supabase
- Hybrid forecasting model:
  - LSTM for sequence-based prediction
  - Markov Chain for state transition probabilities
  - Ensemble weighting (80% LSTM + 20% Markov)
- 12-hour ahead flood risk forecast
- Visual bar chart of past 12 hours water levels
- Automatic Gmail alert system when flood is predicted
- Clean, user-friendly interface with responsive layout

## Water Level Classification

| Water Level (cm) | Class | Meaning             |
|------------------|-------|----------------------|
| < 84             | 0     | NO FLOOD             |
| 84 – 104         | 1     | ALMOST FLOODED       |
| ≥ 105            | 2     | FLOOD                |

## Project Structure

```
.
├── LSTM.keras                  # Trained LSTM model for time series prediction
├── transition_matrix.csv       # Markov Chain transition probabilities
├── flood.py                    # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Result.rar                  # (Archived results / reports)
├── venezia-high-waters-master.zip  # (Possibly dataset source - Venice high water data)
└── README.md                   # This file
```

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```
streamlit
tensorflow
pandas
numpy
matplotlib
seaborn
supabase
requests
```

## Setup Instructions

1. **Replace placeholders in `flood.py`:**
   ```python
   API_URL = 'your-supabase-project-url'
   API_KEY = 'your-supabase-anon-key-or-service-key'
   ```
   → Replace with your actual Supabase project URL and API key.

2. **Email Configuration (for alerts):**
   ```python
   server.login("sundayabraham81@gmail.com", "YOUR_APP_PASSWORD")
   ```
   → Use a Gmail **App Password** (not your regular password). Enable 2FA on Gmail and generate an app password.

3. **Supabase Setup:**
   - Ensure your Supabase table is named `maintable`
   - Required columns: `level` (float/int), `created_at` (timestamp)

4. **Run the app:**

```bash
streamlit run flood.py
```

5. **Login Credentials:**
   - Username: `flood prediction`
   - Password: `computerengineering`

## How It Works

1. On login, the app fetches the latest 12 water level records from Supabase.
2. Data is preprocessed with lag features and temporal encodings.
3. The hybrid model predicts water level class for each of the next 12 hours.
4. Results are displayed in a table with human-readable flood status.
5. A bar chart shows actual water levels from the past 12 hours.
6. If a flood (class 2) is predicted in the next 12 hours, an email alert is sent immediately.

## Disclaimer

- This is an educational/prototype flood early warning system.
- Predictions are based on historical patterns and model assumptions.
- Not intended for critical operational use without further validation and calibration.
- Model performance depends heavily on data quality and local hydrology.

## Future Improvements

- Add confidence intervals to predictions
- Support multiple monitoring stations
- Integrate weather API data (rainfall, tide)
- Dashboard with historical trends and alerts log
- Deploy using Streamlit Community Cloud or Docker

## License

This project is for educational and research purposes. Feel free to modify and adapt for your own flood monitoring needs.

Made with ❤️ for disaster risk reduction and community safety.
