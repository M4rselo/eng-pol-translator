import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

WEBAPP_DIR = Path(__file__).resolve().parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

import loader

app = Flask(__name__, template_folder=str(WEBAPP_DIR / "templates"),
            static_folder=str(WEBAPP_DIR / "static"))

VERSIONS = [v for v, cfg in loader.VERSION_CONFIG.items()
            if (loader.CHECKPOINT_DIR / cfg["checkpoint"]).exists()]
if not VERSIONS:
    sys.exit("No checkpoints found in appdata/checkpoints - see README, section 'Download the model'.")


def _clamp_float(val, default, lo, hi):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _clamp_int(val, default, lo, hi):
    try:
        v = int(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _parse_common(data):
    version = str(data.get("version") or VERSIONS[-1])
    text = str(data.get("text") or "")
    self_ref = data.get("self_ref") or "na"
    addr_ref = data.get("addr_ref") or "na"
    if self_ref not in loader.SELF_OPTS:
        self_ref = "na"
    if addr_ref not in loader.ADDR_OPTS:
        addr_ref = "na"
    alpha = _clamp_float(data.get("alpha", 0.6), 0.6, 0.0, 3.0)
    num_k = _clamp_int(data.get("num_k", 5), 5, 1, 20)
    return version, text, self_ref, addr_ref, alpha, num_k


@app.route("/")
def index():
    ui_configs = {v: loader.ui_config(v) for v in VERSIONS}
    return render_template("index.html", versions=VERSIONS, ui_configs=ui_configs)


@app.route("/api/versions")
def api_versions():
    return jsonify({v: loader.ui_config(v) for v in VERSIONS})


@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    """Cheap, model-free tokenization for the live token counter -- no
    encoder/decoder forward pass, safe to call much more eagerly than
    /api/translate."""
    data = request.get_json(force=True, silent=True) or {}
    version = str(data.get("version") or VERSIONS[-1])
    text = str(data.get("text") or "")
    try:
        version = loader.normalize_version(version)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        result = loader.tokenize_preview(version, text)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Tokenization failed: {exc}"}), 500
    return jsonify(result)


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(force=True, silent=True) or {}
    version, text, self_ref, addr_ref, alpha, num_k = _parse_common(data)
    try:
        version = loader.normalize_version(version)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        result = loader.translate(version, text, self_ref, addr_ref, alpha, num_k, want_attention=False)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Translation failed: {exc}"}), 500
    result["version"] = version
    return jsonify(result)


@app.route("/api/insight", methods=["POST"])
def api_insight():
    data = request.get_json(force=True, silent=True) or {}
    version, text, self_ref, addr_ref, alpha, num_k = _parse_common(data)
    if not text.strip():
        return jsonify({"error": "Empty input."}), 400
    try:
        version = loader.normalize_version(version)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        result = loader.translate(version, text, self_ref, addr_ref, alpha, num_k, want_attention=True)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Translation failed: {exc}"}), 500
    result["version"] = version
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json(force=True, silent=True) or {}
    version, text, _self_ref, _addr_ref, alpha, num_k = _parse_common(data)
    if not text.strip():
        return jsonify({"error": "Empty input."}), 400
    try:
        version = loader.normalize_version(version)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        result = loader.compare_all(version, text, alpha, num_k)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Comparison failed: {exc}"}), 500
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False, threaded=True)
