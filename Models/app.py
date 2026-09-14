from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)

MODELS_DIR = Path(__file__).resolve().parent

# Paths are relative to this file's own location, so it works no matter
# which directory you run "python app.py" from.
BASE_DIR = Path(__file__).resolve().parent.parent  # project root (one level up from models/)
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"

print(f"Loading model from: {MODEL_PATH}")
pipe = joblib.load(MODEL_PATH)

# Recover the exact feature columns/types the model was trained on
df_ref = pd.read_csv(FEATURES_PATH)
DROP_COLS = ["player_id", "name", "current_market_value_eur"]
FEATURE_COLUMNS = [c for c in df_ref.columns if c not in DROP_COLS]
NUM_COLS = df_ref[FEATURE_COLUMNS].select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = df_ref[FEATURE_COLUMNS].select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns expected:", NUM_COLS)
print("Categorical columns expected:", CAT_COLS)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(MODELS_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Player value prediction API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        # Derive the per-90 / contribution features the same way notebook 02 did
        minutes = float(data.get("last_season_minutes", 0) or 0)
        goals = float(data.get("last_season_goals", 0) or 0)
        assists = float(data.get("last_season_assists", 0) or 0)
        minutes_90 = minutes / 90 if minutes > 0 else 0
        goals_per_90 = (goals / minutes_90) if minutes_90 > 0 else 0
        assists_per_90 = (assists / minutes_90) if minutes_90 > 0 else 0

        row = {col: data.get(col) for col in FEATURE_COLUMNS}
        row["goals_per_90"] = goals_per_90
        row["assists_per_90"] = assists_per_90
        row["goal_contributions"] = goals + assists

        input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        for col in NUM_COLS:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
        input_df[NUM_COLS] = input_df[NUM_COLS].fillna(0)
        input_df[CAT_COLS] = input_df[CAT_COLS].fillna("Unknown").astype(str)

        log_pred = pipe.predict(input_df)[0]
        predicted_value = float(np.expm1(log_pred))

        return jsonify({"predicted_market_value_eur": round(predicted_value, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5500)