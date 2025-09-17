
from flask import Flask, render_template, request, jsonify
from sentinal.hoax_filter import HoaxFilter
from sentinal.ai_base import AIBase

import os
import json
from flask import session, redirect, url_for
from flask import abort


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
    # Advanced logic: use selected model for analysis
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
    return jsonify({
        'ai_result': ai_result,
        'red_flag_hits': result.red_flag_hits,
        'source_score': result.source_score,
        'verdict': result.verdict,
        'chat_history': chat_history[-10:]  # last 10 exchanges
    })


@app.route('/chat', methods=['POST'])
def chat():
    text = request.form.get('text')
    chat_model = request.form.get('chatModel', None)
    # Advanced logic: use selected model for chat
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
    return jsonify({'chat_history': chat_history[-10:]})

if __name__ == '__main__':
    app.run(debug=True)

# --- Admin-only routes ---
def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html', user=session.get('user'))

@app.route('/admin/users')
@admin_required
def admin_users():
    return render_template('admin_users.html', user=session.get('user'))

@app.route('/admin/logs')
@admin_required
def admin_logs():
    return render_template('admin_logs.html', user=session.get('user'))
