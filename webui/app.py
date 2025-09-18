


import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to Python path for imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
    logger.info(f"Added workspace root to Python path: {WORKSPACE_ROOT}")

# Import Flask components
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, abort
import json
from webui.auth import bp as auth_bp
from datetime import datetime, timedelta

# Import Aegis components
try:
    from aegis import (
        Sentinel,
        AegisCouncil,
        MetaJudgeAgent,
        NexusExplainStore
    )
    from aegis.sentinel_council import get_council
    logger.info("Successfully imported Aegis components")
except ImportError as e:
    logger.error(f"Failed to import Aegis components: {e}")
    logger.error(f"Python path: {sys.path}")
    raise

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'devsecretkey')
app.register_blueprint(auth_bp)

# Initialize Sentinel with persistence
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "data")  # Create a data directory
os.makedirs(MEMORY_PATH, exist_ok=True)
MEMORY_FILE = os.path.join(MEMORY_PATH, "sentinel_memory.jsonl")
DATASET_PATH = os.path.join(MEMORY_PATH, "user_interactions.jsonl")

try:
    # Initialize Sentinel with NexusExplainStore for persistence
    explain_store = NexusExplainStore(persistence_path=MEMORY_FILE)
    sentinel = Sentinel(explain_store=explain_store)
    
    # Initialize diagnostics
    diagnostics = sentinel.get_diagnostics()
    
    # Run initial health check
    health = diagnostics.run_preflight_checks()
    if health.status != "healthy":
        logger.warning(f"System health check: {health.status} - {health.message}")
    else:
        logger.info("System health check passed")
    
    logger.info("Successfully initialized Sentinel")
except Exception as e:
    logger.error(f"Failed to initialize Sentinel: {e}")
    raise

# Start metrics collection in background
def collect_metrics_loop():
    while True:
        try:
            metrics = diagnostics.collect_metrics()
            time.sleep(60)  # Collect every minute
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            time.sleep(60)

metrics_thread = threading.Thread(target=collect_metrics_loop, daemon=True)
metrics_thread.start()

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
    return render_template('index.html')

@app.route('/diagnostics')
def show_diagnostics():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('diagnostics.html')

# Diagnostics API endpoints
@app.route('/api/diagnostics', methods=['GET'])
def get_diagnostics_data():
    try:
        # Get resource metrics
        resources = diagnostics.get_resource_metrics()
        
        # Get component health
        components = diagnostics.check_critical_components()
        
        # Get active warnings
        warnings = diagnostics.get_active_warnings()
        
        # Get security metrics
        security = {
            "blocked_requests": diagnostics.security_metrics.blocked_requests,
            "warnings": len(diagnostics.security_metrics.recent_warnings),
            "allowed_requests": diagnostics.security_metrics.allowed_requests
        }
        
        # Get audit metrics from audit trail component health
        audit = components.get("audit_trail", {}).get("metrics", {})
        
        return jsonify({
            "resources": {
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "disk_usage": resources.disk_usage,
                "response_times": resources.response_times
            },
            "components": components,
            "warnings": warnings,
            "security": security,
            "audit": audit
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
    return render_template('index.html', user=session.get('user'))

@app.route('/scan', methods=['POST'])
def scan():
    text = request.form.get('text')
    file = request.files.get('file')
    if file:
        text = file.read().decode('utf-8')
    
    # Check with Sentinel first for safety
    scan_model = request.form.get('scanModel', None)
    context = {
        'intent': 'content scan',
        'model': scan_model or 'default',
        'risk_level': 'low'  # Content scanning is lower risk
    }

    result = sentinel.check(text, context)
    
    if not result.allow:
        return jsonify({
            'error': "Content scan blocked by Sentinel",
            'reason': result.reason,
            'severity': result.severity
        }), 403
    
    # Content is safe, get detailed analysis
    analysis = sentinel.analyze(text, context)
    
    entry = {
        'type': 'scan',
        'user': text,
        'scan_model': scan_model,
        'result': analysis,
        'timestamp': datetime.now().isoformat()
    }
    chat_history.append(entry)
    save_interaction(entry)
    
    # Update analytics
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
        logger.error(f"Analytics error: {e}")
        
    explanation = f"Model: {scan_model or 'default'} | Risk Score: {analysis['risk_score']:.2f} | Confidence: {analysis['confidence']:.2f}"
    
    return jsonify({
        'analysis': analysis,
        'explanation': explanation,
        'chat_history': chat_history[-10:]
    })

@app.route('/api/sentinel/status', methods=['GET'])
def sentinel_status():
    """Get the current status of the Sentinel system."""
    try:
        # Get basic system info
        status = sentinel.get_status()
        
        # Try a simple check to verify operation
        test_context = {
            'intent': 'system check',
            'risk_level': 'low'
        }
        result = sentinel.check('status check', test_context)
        
        return jsonify({
            "status": "ok",
            "system": {
                "version": status['version'],
                "explain_store": status['explain_store'],
                "memory_size": status['memory_size'],
                "persistence_path": status['persistence_path']
            },
            "health_check": {
                "allow": result.allow,
                "severity": result.severity,
                "last_check": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
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
    """Get the current state of the Sentinel's explain store with optional filtering."""
    try:
        # Get optional query parameters
        limit = request.args.get('limit', type=int)
        time_window = request.args.get('time_window', type=int)  # In minutes
        
        # Get entries from explain store
        status = sentinel.get_status()
        entries = sentinel.get_history(limit=limit)
        
        # Apply time window filter if needed
        if time_window:
            cutoff = datetime.now() - timedelta(minutes=time_window)
            entries = [
                e for e in entries 
                if isinstance(e.get('timestamp'), (str, datetime)) and 
                (datetime.fromisoformat(e['timestamp']) if isinstance(e['timestamp'], str) else e['timestamp']) > cutoff
            ]
        
        # Get memory stats
        stats = {
            "total_entries": len(entries),
            "store_size": status['memory_size'],
            "persistence_path": status['persistence_path']
        }
        
        return jsonify({
            "status": "ok",
            "history": entries,
            "stats": stats,
            "filters_applied": {
                "limit": limit,
                "time_window": time_window
            }
        })
    except Exception as e:
        logger.error(f"Memory access error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    text = request.form.get('text')
    chat_model = request.form.get('chatModel', None)
    
    # Check with Sentinel first for safety
    context = {
        'intent': 'chat response',
        'model': chat_model or 'default',
        'risk_level': 'medium'  # Chat has moderate risk level
    }
    
    result = sentinel.check(text, context)
    
    if not result.allow:
        return jsonify({
            'error': "Chat request blocked by Sentinel",
            'reason': result.reason,
            'severity': result.severity
        }), 403
    
    # Get response from Sentinel
    response = sentinel.respond(text, context)
    
    entry = {
        'type': 'chat',
        'user': text,
        'response': response,
        'chat_model': chat_model,
        'timestamp': datetime.now().isoformat()
    }
    chat_history.append(entry)
    save_interaction(entry)
    
    # Update analytics
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
        logger.error(f"Analytics error: {e}")
        
    explanation = f"Model: {chat_model or 'default'} | Risk Score: {response['risk_score']:.2f} | Safety Level: {response['safety_level']}"
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
    
    # Context for batch scanning
    context = {
        'intent': 'batch scan',
        'model': model or 'default',
        'risk_level': 'low'
    }
    
    for text in texts:
        result = sentinel.check(text, context)
        if result.allow:
            analysis = sentinel.analyze(text, context)
            response = sentinel.respond(text, context)
            explanation = f"Model: {model or 'default'} | Risk Score: {analysis['risk_score']:.2f} | Safety: {analysis['safety_level']}"
        else:
            analysis = None
            response = None
            explanation = f"Blocked by Sentinel | Reason: {result.reason} | Severity: {result.severity:.2f}"
            
        results.append({
            'allowed': result.allow,
            'analysis': analysis,
            'response': response, 
            'explanation': explanation
        })
        
    return jsonify({'results': results})

@app.route('/api/sentinel/diagnostics/health', methods=['GET'])
@admin_required
def get_system_health():
    """Get current system health status."""
    try:
        health = diagnostics.run_preflight_checks()
        return jsonify({
            "status": health.status,
            "message": health.message,
            "timestamp": health.timestamp,
            "details": health.details
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/sentinel/diagnostics/metrics', methods=['GET'])
@admin_required
def get_system_metrics():
    """Get current system metrics."""
    try:
        metrics = diagnostics.collect_metrics()
        report = diagnostics.get_performance_report()
        return jsonify({
            "current_metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_usage": metrics.disk_usage,
                "explain_store_size": metrics.explain_store_size,
                "active_threads": metrics.active_threads,
                "response_times": metrics.response_times
            },
            "performance_report": report
        })
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return jsonify({
            "error": str(e)
        }), 500

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
