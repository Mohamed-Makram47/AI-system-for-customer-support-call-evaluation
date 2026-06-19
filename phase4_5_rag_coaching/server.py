"""
EJUST Call Evaluation Dashboard - Local Server
Run from phase4_5_rag_coaching/ directory:
    python server.py
Then open: http://localhost:8000
"""

import http.server
import json
import os
import urllib.parse
from pathlib import Path

PORT = 8000
BASE_DIR = Path(__file__).parent

# Paths relative to this script
DATA_DIR        = BASE_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
COACHING_DIR    = DATA_DIR / "coaching"
RESULTS_FILE    = DATA_DIR / "experiments" / "call_class_gt_results.json"
DASHBOARD_FILE  = BASE_DIR / "dashboard.html"

# Load results JSON once at startup
print(f"Loading results from: {RESULTS_FILE}")
with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    ALL_RESULTS = {entry["call_id"]: entry for entry in json.load(f)}
print(f"  Loaded {len(ALL_RESULTS)} call results.")


class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  [{self.address_string()}] {format % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status)

    def send_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error_json(404, f"File not found: {path.name}")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")

        # ── Serve dashboard ──────────────────────────────────────
        if path in ("", "/", "/dashboard", "/index.html"):
            self.send_file(DASHBOARD_FILE, "text/html; charset=utf-8")
            return

        # ── Serve logo ───────────────────────────────────────────
        if path == "/ejust_logo.png":
            self.send_file(BASE_DIR / "ejust_logo.png", "image/png")
            return

        # ── API: result by call_id ────────────────────────────────
        # GET /api/result/<call_id>
        if path.startswith("/api/result/"):
            call_id = path[len("/api/result/"):]
            call_id = urllib.parse.unquote(call_id)
            if call_id in ALL_RESULTS:
                self.send_json(ALL_RESULTS[call_id])
            else:
                self.send_error_json(404, f"call_id '{call_id}' not found in results.")
            return

        # ── API: transcript by call_id ────────────────────────────
        # GET /api/transcript/<call_id>
        if path.startswith("/api/transcript/"):
            call_id  = urllib.parse.unquote(path[len("/api/transcript/"):])
            filepath = TRANSCRIPTS_DIR / f"{call_id}.json"
            if not filepath.exists():
                self.send_error_json(404, f"Transcript not found: {call_id}.json")
                return
            with open(filepath, "r", encoding="utf-8") as f:
                self.send_json(json.load(f))
            return

        # ── API: coaching by call_id ──────────────────────────────
        # GET /api/coaching/<call_id>
        if path.startswith("/api/coaching/"):
            call_id  = urllib.parse.unquote(path[len("/api/coaching/"):])
            filepath = COACHING_DIR / f"{call_id}.json"
            if not filepath.exists():
                self.send_error_json(404, f"Coaching not found: {call_id}.json")
                return
            with open(filepath, "r", encoding="utf-8") as f:
                self.send_json(json.load(f))
            return

        # ── API: list all known call_ids ──────────────────────────
        if path == "/api/calls":
            self.send_json(sorted(ALL_RESULTS.keys()))
            return

        # ── 404 fallback ──────────────────────────────────────────
        self.send_error_json(404, f"Unknown route: {path}")


if __name__ == "__main__":
    print(f"\n{'='*52}")
    print(f"  EJUST Call Evaluation Dashboard")
    print(f"  Server running at http://localhost:{PORT}")
    print(f"  Open your browser and go to http://localhost:{PORT}")
    print(f"{'='*52}\n")
    with http.server.HTTPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
