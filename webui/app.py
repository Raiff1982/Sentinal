


import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to Python path for sentinal imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
    logger.info(f"Added workspace root to Python path: {WORKSPACE_ROOT}")

# Add Aegis directory to Python path
AEGIS_ROOT = os.path.join(WORKSPACE_ROOT, 'aegis2.0_patched', 'aegis2.0')
if os.path.exists(AEGIS_ROOT) and AEGIS_ROOT not in sys.path:
    sys.path.insert(0, AEGIS_ROOT)
    logger.info(f"Added Aegis root to Python path: {AEGIS_ROOT}")

# Import Flask components
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, abort
import json
from webui.auth import bp as auth_bp
from datetime import datetime, timedelta

# Import Sentinel components
try:
    from sentinal.hoax_filter import HoaxFilter
    from sentinal.ai_base import AIBase
    logger.info("Successfully imported Sentinel components")
except ImportError as e:
    logger.error(f"Failed to import Sentinel components: {e}")
    raise

# Import Aegis components from the patched version
AEGIS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'aegis2.0_patched', 'aegis2.0'))
if not os.path.exists(AEGIS_ROOT):
    raise RuntimeError(f"Aegis root path not found: {AEGIS_ROOT}")

# Ensure Aegis path is in Python path
if AEGIS_ROOT not in sys.path:
    sys.path.insert(0, AEGIS_ROOT)
    
try:
    from aegis_timescales import NexusMemory
    from sentinel_council import get_council
except ImportError as e:
    print(f"Failed to import Aegis components: {e}")
    print(f"Python path: {sys.path}")
    raise

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'devsecretkey')
app.register_blueprint(auth_bp)
hf = HoaxFilter()
ai = AIBase()

# Initialize Sentinel Council with persistence
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "sentinel_memory.json")
council = get_council(persistence_path=MEMORY_PATH)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "user_interactions.jsonl")

def save_interaction(data):
    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

chat_history = []

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('index.html', user=session.get('user'))

@app.route('/scan', methods=['POST'])
def scan():
    text = request.form.get('text')
    file = request.files.get('file')
    if file:
        text = file.read().decode('utf-8')
        
    # Check with sentinel council first
    council_input = {
        "text": text,
        "intent": "content scan",
        "_signals": {
            "bio": {"stress": 0.2},  # Lower default stress for scanning
            "env": {"context_risk": 0.3}  # Lower default risk for scanning
        }
    }
    council_out = council.dispatch(council_input)
    meta_report = next((r for r in council_out.get("reports", []) if r.get("agent") == "MetaJudge"), {})
    decision = meta_report.get("details", {}).get("decision", "BLOCK")
    
    if decision == "BLOCK":
        return jsonify({
            'error': "Content scan blocked by sentinel council",
            'council_report': council_out
        }), 403
    
    # Proceed with scan if not blocked
    scan_model = request.form.get('scanModel', None)
    if scan_model:
        custom_ai = AIBase(model_names=[os.path.join(os.path.dirname(__file__), scan_model) if os.path.isdir(os.path.join(os.path.dirname(__file__), scan_model) ) else scan_model])
        ai_result = custom_ai.analyze(text)
    else:
        ai_result = ai.analyze(text)
    result = hf.score(text)
    entry = {
        'type': 'scan',
        'user': text,
        'ai': ai_result,
        'result': result.verdict,
        'council_decision': decision,
        'council_report': council_out,
        'red_flag_hits': result.red_flag_hits,
        'source_score': result.source_score,
        'scan_model': scan_model
    }
    chat_history.append(entry)
    save_interaction(entry)
    analytics_path = os.path.join(os.path.dirname(__file__), "analytics.json")
    try:
        if os.path.exists(analytics_path):
            with open(analytics_path, "r", encoding="utf-8") as f:
                analytics = json.load(f)
        else:
            analytics = {"scan": {}, "chat": {}}
        model = scan_model or "default"
        analytics["scan"].setdefault(model, 0)
        analytics["scan"][model] += 1
        with open(analytics_path, "w", encoding="utf-8") as f:
            json.dump(analytics, f)
    except Exception as e:
        print(f"Analytics error: {e}")
    explanation = f"Model: {scan_model or 'default'} | Majority label: {ai_result[0]['label']} | Avg score: {ai_result[0]['score']:.2f}"
    return jsonify({
        'ai_result': ai_result,
        'red_flag_hits': result.red_flag_hits,
        'source_score': result.source_score,
        'verdict': result.verdict,
        'explanation': explanation,
        'chat_history': chat_history[-10:]
    })

@app.route('/api/sentinel/status', methods=['GET'])
def sentinel_status():
    """Get the current status of the sentinel council."""
    try:
        # Request a simple decision to check council health
        out = council.dispatch({"text": "status check"})
        memory_snapshot = out.get("memory_snapshot", {})
        
        return jsonify({
            "status": "ok",
            "agents": len(out.get("reports", [])),
            "memory_entries": len(memory_snapshot),
            "reports": out.get("reports", [])
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/sentinel/decide', methods=['POST'])
def sentinel_decide():
    """Request a decision from the sentinel council."""
    try:
        # Get input data from request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        # Add signals from request or use defaults
        signals = data.get("signals", {})
        if not signals.get("bio"):
            signals["bio"] = {"stress": 0.0}
        if not signals.get("env"):
            signals["env"] = {"context_risk": 0.0}
            
        # Create council input
        council_input = {
            "text": data.get("text", ""),
            "intent": data.get("intent", ""),
            "_signals": signals,
            "_last_decision": data.get("last_decision")
        }
        
        # Get council decision
        out = council.dispatch(council_input)
        
        # Extract decision and relevant details
        reports = out.get("reports", [])
        meta_report = next((r for r in reports if r.get("agent") == "MetaJudge"), {})
        timescale_report = next((r for r in reports if r.get("agent") == "TimeScaleCoordinator"), {})
        
        return jsonify({
            "decision": meta_report.get("details", {}).get("decision"),
            "severity": meta_report.get("severity", 0.0),
            "timescale_fusion": timescale_report.get("details", {}),
            "input_audit": out.get("input_audit", {}),
            "reports": reports,
            "graph": out.get("explainability_graph", {})
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/sentinel/memory', methods=['GET'])
def sentinel_memory():
    """Get the current state of the sentinel memory with optional filtering."""
    try:
        # Get optional query parameters
        limit = request.args.get('limit', type=int)
        agent = request.args.get('agent')
        decision = request.args.get('decision')
        time_window = request.args.get('time_window', type=int)  # In minutes
        
        # Get memory entries
        entries = council.memory.entries
        
        # Apply filters
        if agent:
            entries = [e for e in entries if any(r['agent'] == agent for r in e.get('reports', []))]
        if decision:
            entries = [e for e in entries if any(r.get('details', {}).get('decision') == decision 
                                               for r in e.get('reports', []))]
        if time_window:
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(minutes=time_window)
            entries = [e for e in entries if datetime.fromisoformat(e['timestamp']) > cutoff]
        if limit:
            entries = entries[-limit:]
        
        # Get memory audit
        audit = council.memory.audit()
        
        return jsonify({
            "status": "ok",
            "memory": entries,
            "audit": audit,
            "filters_applied": {
                "limit": limit,
                "agent": agent,
                "decision": decision,
                "time_window": time_window
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    text = request.form.get('text')
    chat_model = request.form.get('chatModel', None)
    
    # Ask sentinel council for decision
    council_input = {
        "text": text,
        "intent": "chat response",
        "_signals": {
            "bio": {"stress": 0.3},  # Default moderate stress for chat
            "env": {"context_risk": 0.4}  # Default moderate risk for chat
        }
    }
    council_out = council.dispatch(council_input)
    meta_report = next((r for r in council_out.get("reports", []) if r.get("agent") == "MetaJudge"), {})
    decision = meta_report.get("details", {}).get("decision", "BLOCK")
    
    if decision == "BLOCK":
        return jsonify({
            'error': "Request blocked by sentinel council",
            'council_report': council_out
        }), 403
    
    # Proceed with chat if not blocked
    if chat_model:
        custom_ai = AIBase(llm_names=[os.path.join(os.path.dirname(__file__), chat_model) if os.path.isdir(os.path.join(os.path.dirname(__file__), chat_model)) else chat_model])
        ai_result = custom_ai.analyze(text)
        llm_response = custom_ai.chat(text)
    else:
        ai_result = ai.analyze(text)
        llm_response = ai.chat(text)
        
    result = hf.score(text)
    entry = {
        'type': 'chat',
        'user': text,
        'ai': ai_result,
        'llm': llm_response,
        'council_decision': decision,
        'result': result.verdict,
        'chat_model': chat_model
    }
    chat_history.append(entry)
    save_interaction(entry)
    analytics_path = os.path.join(os.path.dirname(__file__), "analytics.json")
    try:
        if os.path.exists(analytics_path):
            with open(analytics_path, "r", encoding="utf-8") as f:
                analytics = json.load(f)
        else:
            analytics = {"scan": {}, "chat": {}}
        model = chat_model or "default"
        analytics["chat"].setdefault(model, 0)
        analytics["chat"][model] += 1
        with open(analytics_path, "w", encoding="utf-8") as f:
            json.dump(analytics, f)
    except Exception as e:
        print(f"Analytics error: {e}")
    explanation = f"Model: {chat_model or 'default'} | Majority label: {ai_result[0]['label']} | Avg score: {ai_result[0]['score']:.2f}"
    return jsonify({'chat_history': chat_history[-10:], 'explanation': explanation})

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html', user=session.get('user'))

@app.route('/admin/users')
@admin_required
def admin_users():
    return render_template('admin_users.html', user=session.get('user'))

@app.route('/admin/batch', methods=['GET', 'POST'])
@admin_required
def admin_batch():
    if request.method == 'GET':
        return render_template('batch_scan.html', user=session.get('user'))
    data = request.get_json()
    texts = data.get('texts', [])
    model = data.get('model', None)
    results = []
    for text in texts:
        custom_ai = AIBase(model_names=[model], llm_names=[model])
        scan_result = custom_ai.analyze(text)
        chat_result = custom_ai.chat(text)
        explanation = f"Model: {model} | Majority label: {scan_result[0]['label']} | Avg score: {scan_result[0]['score']:.2f}"
        results.append({
            'scan': scan_result,
            'chat': chat_result,
            'explanation': explanation
        })
    return jsonify({'results': results})

@app.route('/admin/analytics_data')
@admin_required
def admin_analytics_data():
    analytics_path = os.path.join(os.path.dirname(__file__), "analytics.json")
    if os.path.exists(analytics_path):
        with open(analytics_path, "r", encoding="utf-8") as f:
            analytics = json.load(f)
    else:
        analytics = {"scan": {}, "chat": {}}
    return jsonify(analytics)

@app.route('/admin/feedback', methods=['GET', 'POST'])
@admin_required
def admin_feedback():
    feedback_path = os.path.join(os.path.dirname(__file__), "feedback_labels.jsonl")
    if request.method == 'GET':
        return render_template('feedback_label.html', user=session.get('user'))
    data = request.get_json()
    entry = {"text": data.get("text"), "label": data.get("label")}
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return jsonify({"status": "Feedback saved."})

@app.route('/admin/retrain', methods=['POST'])
@admin_required
def admin_retrain():
    import subprocess
    result = subprocess.run([os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "python.exe"), os.path.join(os.path.dirname(__file__), "train_and_deploy.py")], capture_output=True, text=True)
    status = "Retraining complete." if result.returncode == 0 else f"Error: {result.stderr}"
    return render_template('admin_dashboard.html', user=session.get('user'), retrain_status=status)


# --- Flask run configuration ---
# To run this app in development:
# 1. Set environment variable: $env:FLASK_APP = "webui/app.py" (PowerShell)
# 2. Optionally: $env:FLASK_ENV = "development"
# 3. Run: flask run
# Or, run directly as a script:
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
