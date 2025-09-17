


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request, jsonify
from sentinal.hoax_filter import HoaxFilter
from sentinal.ai_base import AIBase
import json
from flask import session, redirect, url_for, abort
from webui.auth import bp as auth_bp

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'devsecretkey')
app.register_blueprint(auth_bp)
hf = HoaxFilter()
ai = AIBase()

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
    scan_model = request.form.get('scanModel', None)
    if scan_model:
        custom_ai = AIBase(model_names=[os.path.join(os.path.dirname(__file__), scan_model) if os.path.isdir(os.path.join(os.path.dirname(__file__), scan_model) ) else scan_model])
        ai_result = custom_ai.analyze(text)
    else:
        ai_result = ai.analyze(text)
    result = hf.score(text)
    entry = {'type': 'scan', 'user': text, 'ai': ai_result, 'result': result.verdict,
             'red_flag_hits': result.red_flag_hits, 'source_score': result.source_score, 'scan_model': scan_model}
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

@app.route('/chat', methods=['POST'])
def chat():
    text = request.form.get('text')
    chat_model = request.form.get('chatModel', None)
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
