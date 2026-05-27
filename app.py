from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import sqlite3
import logging
import os

# ─── App Setup ──────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── AI Model (loaded once on startup) ──────────────────────────────────────

log.info("Loading AI model… this may take a moment on first run.")
try:
    analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    log.info("AI model loaded successfully.")
except Exception as e:
    log.error(f"Failed to load AI model: {e}")
    analyzer = None

# ─── Database ────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "sentiment.db")


def get_db():
    """Return a fresh thread-safe SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT    NOT NULL,
                sentiment TEXT    NOT NULL,
                confidence REAL   NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    log.info("Database ready.")


init_db()

# ─── Helper ──────────────────────────────────────────────────────────────────

def normalize_label(raw_label: str) -> str:
    """Map raw model labels to Positive / Negative / Neutral."""
    label = raw_label.upper()
    if label in ("POSITIVE", "LABEL_1"):
        return "POSITIVE"
    if label in ("NEGATIVE", "LABEL_0"):
        return "NEGATIVE"
    return "NEUTRAL"

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": analyzer is not None})


@app.route("/analyze", methods=["POST"])
def analyze():
    if analyzer is None:
        return jsonify({"error": "AI model not available. Please restart the server."}), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long. Please keep it under 2000 characters."}), 400

    # Truncate to 512 tokens safety limit for the model
    truncated = text[:1500]

    try:
        result = analyzer(truncated)[0]
    except Exception as e:
        log.error(f"Model inference error: {e}")
        return jsonify({"error": "Analysis failed. Please try again."}), 500

    sentiment = normalize_label(result["label"])
    confidence = round(result["score"], 4)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO reviews (text, sentiment, confidence, timestamp) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (text, sentiment, confidence)
        )
        conn.commit()

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence,
        "text": text[:100] + ("…" if len(text) > 100 else "")
    })


@app.route("/stats")
def stats():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT sentiment, COUNT(*) as count FROM reviews GROUP BY sentiment"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]

    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for row in rows:
        if row["sentiment"] in counts:
            counts[row["sentiment"]] = row["count"]

    return jsonify({**counts, "TOTAL": total})


@app.route("/history")
def history():
    limit = min(int(request.args.get("limit", 10)), 50)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, text, sentiment, confidence, timestamp FROM reviews ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()

    return jsonify([
        {
            "id": r["id"],
            "text": r["text"],
            "sentiment": r["sentiment"],
            "confidence": r["confidence"],
            "created_at": r["timestamp"]
        }
        for r in rows
    ])


@app.route("/clear", methods=["DELETE"])
def clear_history():
    with get_db() as conn:
        conn.execute("DELETE FROM reviews")
        conn.commit()
    return jsonify({"message": "History cleared."})


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)