# SentinelAI

**Intelligent Log Anomaly Detection System**

SentinelAI is a premium Streamlit dashboard for detecting anomalies in system logs using an unsupervised machine learning ensemble. It combines structured log parsing, feature engineering, anomaly scoring, and optional Gemini-powered AI explanations to help engineers identify suspicious events faster.

## Features

- **Log ingestion from multiple formats**: `.log`, `.txt`, `.csv`, `.json`, and `.jsonl`.
- **Automatic parsing** of Apache/Nginx-style logs, syslog-style logs, and raw text logs.
- **Feature extraction** from:
  - log severity levels,
  - HTTP status codes,
  - response size,
  - response time,
  - timestamp hour,
  - error keywords,
  - TF-IDF text features from log messages.
- **Ensemble anomaly detection** using:
  - Isolation Forest,
  - One-Class SVM.
- **Adaptive thresholding** with user-controlled false positive rate target.
- **Interactive dashboard** with:
  - KPI cards,
  - anomaly timeline,
  - score distribution chart,
  - log level breakdown,
  - feature radar chart,
  - severity heatmap,
  - annotated log tables.
- **AI Security Analyst panel** powered by Gemini 1.5 Flash for anomaly explanations.
- **Downloadable results** as CSV.

## Project Structure

- `app.py` — Main Streamlit dashboard.
- `data_processing.py` — Log parsing and feature engineering.
- `model.py` — Ensemble anomaly detection logic.
- `explainer.py` — Gemini-based anomaly explanation module.
- `requirements.txt` — Python dependencies.
- `sample_logs.log` — Sample log file for testing.

## How It Works

1. Upload a log file or use the built-in sample logs.
2. SentinelAI parses the file and standardizes the log data.
3. Features are extracted from both structured fields and log text.
4. The anomaly detection ensemble scores each log line.
5. A threshold is applied based on the selected false positive rate target.
6. The dashboard highlights suspicious log lines and visualizes patterns.
7. For flagged anomalies, the AI analyst can generate a short threat explanation and mitigation step.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sentinelai.git
cd sentinelai
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

For AI explanations, set your Gemini API key:

```bash
GEMINIAPIKEY=your_api_key_here
```

You can place it in a `.env` file or set it directly in your environment.

## Running the App

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser.
2. Upload a log file or enable sample data.
3. Adjust the sidebar controls:
   - **FPR Target**
   - **Contamination Prior**
   - chart toggles
4. Review detected anomalies in the dashboard.
5. Use the AI Security Analyst panel for deeper insights on suspicious entries.
6. Download the full results as CSV if needed.

## Supported Log Formats

- Apache / Nginx access logs
- Syslog-style logs
- CSV logs
- JSON lines / NDJSON
- Plain text logs

## Detection Signals

SentinelAI flags logs based on signals such as:

- ERROR / CRITICAL / FATAL severity
- HTTP 5xx responses
- Slow response times
- Repeated failed login attempts
- Connection errors
- Rare or unusual message vocabulary
- Off-hours activity
- Circuit breaker / OOM / disk failure patterns

## Sample Output

The dashboard includes:
- total log count,
- anomaly count,
- estimated detection accuracy,
- estimated false positive rate,
- anomaly score distributions,
- top suspicious log lines.

## AI Analyst

The AI Security Analyst uses Gemini 1.5 Flash to explain why a log line is suspicious and suggest one immediate mitigation step. Results are cached to reduce repeated API usage.

## Requirements

Main libraries used:
- Streamlit
- pandas
- numpy
- scikit-learn
- plotly
- scipy
- google-genai
- python-dotenv

## License

This project is built for a hackathon submission. Add your preferred license here if needed.

## Author

Built for Tensor Hackathon 2026.

## Website

https://tr-161-innovatex-anamoly-detection-tool.onrender.com
