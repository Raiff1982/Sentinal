
from flask import Flask, render_template, request, jsonify
from sentinal.hoax_filter import HoaxFilter
import os

app = Flask(__name__)
hf = HoaxFilter()

chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    text = request.form.get('text')
    file = request.files.get('file')
    if file:
        text = file.read().decode('utf-8')
    result = hf.score(text)
    chat_history.append({'user': text, 'result': result.verdict})
    return jsonify({
        'red_flag_hits': result.red_flag_hits,
        'source_score': result.source_score,
        'verdict': result.verdict,
        'chat_history': chat_history[-10:]  # last 10 exchanges
    })

@app.route('/chat', methods=['POST'])
def chat():
    text = request.form.get('text')
    result = hf.score(text)
    chat_history.append({'user': text, 'result': result.verdict})
    return jsonify({'chat_history': chat_history[-10:]})

if __name__ == '__main__':
    app.run(debug=True)
