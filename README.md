MonteSight — 1-Year Monte Carlo Stock Forecaster

A Streamlit application for probability-driven stock price forecasting.

📌 Overview

MonteSight is a lightweight, fast, and intuitive Monte Carlo simulation app that forecasts stock prices over a 1-year horizon.
It uses:

Historical price data from yfinance

Daily log returns to estimate drift & volatility

Thousands of simulated price paths

Probability-based price targets (≥66% hit probability)

Clean, user-friendly Streamlit UI

The goal:
Help any investor—beginner or expert—quickly understand the range of likely future prices for a selected stock.

✨ Key Features
✔ 1-Year Monte Carlo Forecasting

Simulates thousands of price paths for the next 252 trading days.

✔ Probability of Hitting Price Targets

Computes and displays only price levels with ≥ 66% probability.

✔ Visual Insights

Includes:

Price path visualization

Terminal price distribution

Probability table

Percentile band (17th/50th/83rd percentiles)

✔ Clean Streamlit UI

No clutter — simple controls, intuitive visuals, and plain-language summaries.

✔ Efficient & Vectorized

Simulation uses numpy vectorization for fast, scalable performance.

📁 Project Structure
monte_sight/
├─ app.py
├─ config/
│  └─ settings.py
├─ core/
│  ├─ utils.py
│  ├─ data_loader.py
│  ├─ simulation.py
│  └─ analytics.py
├─ ui/
│  ├─ layout.py
│  ├─ components.py
│  └─ plots.py
├─ tests/
│  ├─ test_utils.py
│  ├─ test_data_loader.py
│  ├─ test_simulation.py
│  ├─ test_analytics.py
│  └─ test_integration.py
├─ PRD.md
├─ DATA_MODEL.md
├─ PIPELINE_SPEC.md
├─ AGENT_INSTRUCTIONS.md
├─ BACKLOG.md
├─ SPRINT_PLAN.md
└─ ARCHITECTURE.md

🚀 Getting Started
1. Clone the Repository
git clone https://github.com/yourusername/monte_sight.git
cd monte_sight

2. Create a Virtual Environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run the App
streamlit run app.py


Then open the local URL (usually http://localhost:8501
).

⚙️ How It Works

MonteSight uses a Geometric Brownian Motion (GBM) model driven by:

Daily drift (μ)

Daily volatility (σ)

Random shocks drawn from a normal distribution

Simulation Steps

Fetch historical adjusted-close prices with yfinance

Compute daily log returns

Estimate drift and volatility

Simulate N price paths over 252 days

Analyze terminal prices to compute:

Price-hit probabilities

17/50/83 percentile band

Probability-filtered price targets

🧠 Example Insights Provided by the App

You will see outputs like:

“This stock has a 72% chance of finishing above $540 within the next year.”

“Most likely range (1σ-equivalent): $410 – $620.”

“Median projected price after 1 year: $503.”

📊 Screenshots (optional placeholders)

Add screenshots here once the UI is completed.

🧪 Testing

All tests live under /tests.

Run them with:

pytest


(If pytest is not installed, add it to requirements.)

📦 Deployment

MonteSight can be deployed to:

Streamlit Cloud

Hugging Face Spaces

Your own containerized server

Just ensure requirements.txt is present.

🔧 Tech Stack

Python 3.10+

Streamlit

yfinance

pandas

numpy

matplotlib or plotly

🛑 Limitations

Monte Carlo models assume a simplified view of markets

Forecasts are probabilistic, not predictions

Results rely heavily on past volatility (which can change)

🗺 Roadmap (Post-MVP)

Multi-year forecasts

Volatility-regime model upgrades

Heston or Jump-Diffusion models

Portfolio-level simulations

API version of the forecasting engine

PDF export of results

📬 Contact / Contributions

Pull requests welcome.
For issues or feedback, open a GitHub Issue in the repo.
