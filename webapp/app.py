"""Local demo server for the FL Edge-IoT IDS project.

Run with:  python app.py
Then open: http://127.0.0.1:5000

Serves the real results (results/dashboard_*.json, copied into
webapp/data/ at build time) and a single page that animates real
traffic being classified by the real trained model, plus the project's
actual comparison charts. No Mininet, no Linux-only features -- this
runs the same way on Windows, macOS or Linux.
"""
from flask import Flask, send_from_directory, jsonify
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/data/<name>.json")
def data(name):
    safe_name = os.path.basename(name) + ".json"
    path = os.path.join(DATA_DIR, safe_name)
    if not os.path.isfile(path):
        return jsonify({"error": f"no such dataset: {name}"}), 404
    return send_from_directory(DATA_DIR, safe_name)


if __name__ == "__main__":
    print("\n  FL Edge-IoT IDS -- local demo")
    print("  Open http://127.0.0.1:5000 in your browser\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
