# SentimentIQ — AI Sentiment Analyzer

An AI-powered web app that classifies text (reviews, tweets, feedback) as **Positive**, **Negative**, or **Neutral** using Hugging Face Transformers, Flask, and SQLite.

Built by **Ritesh** as a college mini-project.

---

## Features

- Real-time sentiment analysis via Hugging Face `distilbert-base-uncased-finetuned-sst-2-english`
- Confidence score with animated progress bar
- Pie chart & bar chart (Chart.js) for sentiment distribution
- Recent analysis history table
- SQLite database — no setup required
- Premium dark-mode UI with smooth animations

---

## Project Structure

```
sentiment_analyzer/
├── app.py              ← Flask backend (all routes)
├── requirements.txt    ← Python dependencies
├── sentiment.db        ← SQLite database (auto-created)
├── templates/
│   └── index.html      ← Frontend dashboard
└── venv/               ← Virtual environment
```

---

## Setup & Run

### 1. Activate virtual environment

```bash
# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: `torch` + `transformers` will download ~250 MB on first install. The AI model (~70 MB) is downloaded automatically on first run.

### 3. Start the app

```bash
python app.py
```

Open your browser at → **http://127.0.0.1:5000**

---

## API Endpoints

| Method | Endpoint    | Description                        |
|--------|-------------|------------------------------------|
| GET    | `/`         | Serves the dashboard               |
| POST   | `/analyze`  | Analyze text → `{sentiment, confidence}` |
| GET    | `/stats`    | Sentiment counts (Pos/Neg/Neu/Total) |
| GET    | `/history`  | Last 10 analyses                   |
| DELETE | `/clear`    | Clear all history                  |
| GET    | `/health`   | Health check                       |

---

## Tech Stack

| Layer     | Technology |
|-----------|-----------|
| Frontend  | HTML5, CSS3, JavaScript, Chart.js |
| Backend   | Python, Flask, Flask-CORS |
| AI Model  | Hugging Face Transformers (DistilBERT) |
| Database  | SQLite (built into Python) |
