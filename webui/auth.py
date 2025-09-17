from flask import Blueprint, render_template, redirect, url_for, request, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

bp = Blueprint('auth', __name__)

# Simple in-memory user store for demo
users = {
    'admin': {'password': generate_password_hash('admin123'), 'role': 'admin'},
    'user': {'password': generate_password_hash('user123'), 'role': 'user'}
}

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_obj = users.get(username)
        if user_obj and check_password_hash(user_obj['password'], password):
            session['user'] = username
            session['role'] = user_obj['role']
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@bp.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('auth.login'))

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form.get('role', 'user')
        if username in users:
            flash('Username already exists')
        else:
            users[username] = {'password': generate_password_hash(password), 'role': role}
            flash('Registration successful')
            return redirect(url_for('auth.login'))
    return render_template('register.html')
