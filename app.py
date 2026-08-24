import os
import math
import re
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load trained model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
VEC_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')

print("Loading Machine Learning model and vectorizer...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
print("ML Model loaded successfully!")

def calculate_entropy(password):
    if not password:
        return 0.0
    pool_size = 0
    if re.search(r'[a-z]', password):
        pool_size += 26
    if re.search(r'[A-Z]', password):
        pool_size += 26
    if re.search(r'[0-9]', password):
        pool_size += 10
    if re.search(r'[^a-zA-Z0-9]', password):
        pool_size += 32
    if pool_size == 0:
        pool_size = 256
    entropy = len(password) * math.log2(pool_size)
    return round(entropy, 1)

def get_suggestions(password):
    suggestions = []
    if len(password) < 8:
        suggestions.append("Increase length to at least 10–12 characters.")
    if not re.search(r'[A-Z]', password):
        suggestions.append("Include uppercase letters (A-Z).")
    if not re.search(r'[a-z]', password):
        suggestions.append("Include lowercase letters (a-z).")
    if not re.search(r'[0-9]', password):
        suggestions.append("Add numeric digits (0-9).")
    if not re.search(r'[^a-zA-Z0-9]', password):
        suggestions.append("Include special symbols (!@#$%^&* etc.).")
    if len(set(password)) < len(password) / 2:
        suggestions.append("Avoid repeating characters excessively.")
    return suggestions

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Password Strength Detector - Live Demo</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0b0f19;
            --card-bg: rgba(18, 26, 44, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --weak: #ef4444;
            --medium: #f59e0b;
            --strong: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(59, 130, 246, 0.12) 0%, transparent 40%);
        }

        .container {
            width: 100%;
            max-width: 680px;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 9999px;
            background: rgba(6, 182, 212, 0.1);
            border: 1px solid rgba(6, 182, 212, 0.3);
            color: var(--accent-cyan);
            font-size: 0.825rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .title {
            font-size: 2.25rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.975rem;
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .input-group {
            position: relative;
            margin-bottom: 1.5rem;
        }

        .label {
            display: block;
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
        }

        .password-input {
            width: 100%;
            padding: 1rem 3.5rem 1rem 1.25rem;
            background: rgba(10, 15, 29, 0.6);
            border: 1.5px solid var(--border-color);
            border-radius: 14px;
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            outline: none;
            transition: all 0.25s ease;
        }

        .password-input:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 4px rgba(6, 182, 212, 0.15);
        }

        .toggle-btn {
            position: absolute;
            right: 1rem;
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 0.25rem;
            font-size: 1.1rem;
            transition: color 0.2s;
        }

        .toggle-btn:hover {
            color: #ffffff;
        }

        /* Meter */
        .meter-container {
            margin-bottom: 1.5rem;
        }

        .meter-label-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .status-text {
            font-weight: 700;
            font-size: 1.1rem;
            transition: color 0.3s ease;
        }

        .entropy-badge {
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.05);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
        }

        .meter-track {
            height: 10px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            overflow: hidden;
            display: flex;
            gap: 4px;
            padding: 2px;
        }

        .meter-bar {
            height: 100%;
            flex: 1;
            border-radius: 999px;
            background: transparent;
            transition: background-color 0.4s ease, transform 0.3s ease;
        }

        /* Probabilities Section */
        .grid-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.85rem;
            text-align: center;
        }

        .stat-name {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stat-val {
            font-size: 1.2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Features Analysis */
        .features-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .feature-chip {
            font-size: 0.8rem;
            padding: 0.35rem 0.75rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.4rem;
            transition: all 0.2s ease;
        }

        .feature-chip.active {
            background: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.3);
            color: var(--strong);
        }

        /* Suggestions */
        .suggestions-box {
            background: rgba(245, 158, 11, 0.05);
            border: 1px solid rgba(245, 158, 11, 0.2);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
        }

        .suggestions-title {
            font-size: 0.875rem;
            font-weight: 700;
            color: #fbbf24;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .suggestions-list {
            list-style: none;
            padding-left: 0;
        }

        .suggestions-list li {
            font-size: 0.85rem;
            color: #d1d5db;
            margin-bottom: 0.35rem;
            position: relative;
            padding-left: 1.2rem;
        }

        .suggestions-list li::before {
            content: "•";
            position: absolute;
            left: 0.3rem;
            color: #fbbf24;
        }

        .actions {
            display: flex;
            gap: 0.75rem;
        }

        .btn {
            flex: 1;
            padding: 0.85rem 1rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            border: none;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: #ffffff;
            box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3);
        }

        .btn-primary:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.09);
        }

        .samples-title {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
        }

        .sample-buttons {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .sample-btn {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            padding: 0.35rem 0.7rem;
            border-radius: 8px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .sample-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <span class="badge">Machine Learning Model</span>
        <h1 class="title">Password Strength Classifier</h1>
        <p class="subtitle">Character-level TF-IDF Vectorization & Logistic Regression ML Model</p>
    </div>

    <div class="card">
        <div class="input-group">
            <label class="label" for="passwordInput">ENTER PASSWORD TO EVALUATE</label>
            <div class="input-wrapper">
                <input type="password" id="passwordInput" class="password-input" placeholder="Type a password..." autocomplete="off">
                <button type="button" id="toggleVisibility" class="toggle-btn" title="Toggle visibility">👁️</button>
            </div>
        </div>

        <div class="meter-container">
            <div class="meter-label-row">
                <span id="statusText" class="status-text" style="color: var(--text-muted);">Enter password</span>
                <span id="entropyBadge" class="entropy-badge">0.0 bits entropy</span>
            </div>
            <div class="meter-track">
                <div id="bar1" class="meter-bar"></div>
                <div id="bar2" class="meter-bar"></div>
                <div id="bar3" class="meter-bar"></div>
            </div>
        </div>

        <!-- Probabilities -->
        <div class="grid-stats">
            <div class="stat-card">
                <div class="stat-name">Weak (0)</div>
                <div id="probWeak" class="stat-val" style="color: var(--weak);">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-name">Medium (1)</div>
                <div id="probMedium" class="stat-val" style="color: var(--medium);">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-name">Strong (2)</div>
                <div id="probStrong" class="stat-val" style="color: var(--strong);">0%</div>
            </div>
        </div>

        <!-- Character variety chips -->
        <div class="features-row">
            <div id="chipLen" class="feature-chip">📏 Length &ge; 8</div>
            <div id="chipUpper" class="feature-chip">🔠 Uppercase</div>
            <div id="chipLower" class="feature-chip">🔡 Lowercase</div>
            <div id="chipNum" class="feature-chip">🔢 Digits</div>
            <div id="chipSym" class="feature-chip">✨ Special Symbol</div>
        </div>

        <!-- Suggestions box -->
        <div id="suggestionsBox" class="suggestions-box" style="display: none;">
            <div class="suggestions-title">⚡ Suggestions to Improve Strength</div>
            <ul id="suggestionsList" class="suggestions-list"></ul>
        </div>

        <div class="actions">
            <button id="generateBtn" class="btn btn-primary">⚡ Generate Strong Password</button>
        </div>

        <div class="samples-title">TRY PRESET SAMPLE PASSWORDS</div>
        <div class="sample-buttons">
            <button class="sample-btn" onclick="testSample('123456')">123456</button>
            <button class="sample-btn" onclick="testSample('password123')">password123</button>
            <button class="sample-btn" onclick="testSample('megzy123')">megzy123</button>
            <button class="sample-btn" onclick="testSample('kino3434')">kino3434</button>
            <button class="sample-btn" onclick="testSample('AVYq1lDE4MgAZfNt')">AVYq1lDE4MgAZfNt</button>
            <button class="sample-btn" onclick="testSample('K9#mQ!8$pLz2@vR5')">K9#mQ!8$pLz2@vR5</button>
        </div>
    </div>
</div>

<script>
    const passwordInput = document.getElementById('passwordInput');
    const toggleVisibility = document.getElementById('toggleVisibility');
    const statusText = document.getElementById('statusText');
    const entropyBadge = document.getElementById('entropyBadge');
    const bar1 = document.getElementById('bar1');
    const bar2 = document.getElementById('bar2');
    const bar3 = document.getElementById('bar3');

    const probWeak = document.getElementById('probWeak');
    const probMedium = document.getElementById('probMedium');
    const probStrong = document.getElementById('probStrong');

    const chipLen = document.getElementById('chipLen');
    const chipUpper = document.getElementById('chipUpper');
    const chipLower = document.getElementById('chipLower');
    const chipNum = document.getElementById('chipNum');
    const chipSym = document.getElementById('chipSym');

    const suggestionsBox = document.getElementById('suggestionsBox');
    const suggestionsList = document.getElementById('suggestionsList');
    const generateBtn = document.getElementById('generateBtn');

    toggleVisibility.addEventListener('click', () => {
        const type = passwordInput.type === 'password' ? 'text' : 'password';
        passwordInput.type = type;
        toggleVisibility.textContent = type === 'password' ? '👁️' : '🙈';
    });

    let debounceTimer;
    passwordInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(evaluatePassword, 150);
    });

    function testSample(pwd) {
        passwordInput.value = pwd;
        evaluatePassword();
    }

    async function evaluatePassword() {
        const password = passwordInput.value;
        if (!password) {
            resetUI();
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ password })
            });
            const data = await response.json();
            updateUI(data);
        } catch (err) {
            console.error(err);
        }
    }

    function resetUI() {
        statusText.textContent = "Enter password";
        statusText.style.color = "var(--text-muted)";
        entropyBadge.textContent = "0.0 bits entropy";
        bar1.style.backgroundColor = "transparent";
        bar2.style.backgroundColor = "transparent";
        bar3.style.backgroundColor = "transparent";

        probWeak.textContent = "0%";
        probMedium.textContent = "0%";
        probStrong.textContent = "0%";

        [chipLen, chipUpper, chipLower, chipNum, chipSym].forEach(c => c.classList.remove('active'));
        suggestionsBox.style.display = 'none';
    }

    function updateUI(data) {
        // Label & Color
        statusText.textContent = `Strength: ${data.label.toUpperCase()}`;
        statusText.style.color = data.color;
        entropyBadge.textContent = `${data.entropy} bits entropy`;

        // Bars
        bar1.style.backgroundColor = "transparent";
        bar2.style.backgroundColor = "transparent";
        bar3.style.backgroundColor = "transparent";

        if (data.prediction === 0) {
            bar1.style.backgroundColor = "var(--weak)";
        } else if (data.prediction === 1) {
            bar1.style.backgroundColor = "var(--medium)";
            bar2.style.backgroundColor = "var(--medium)";
        } else {
            bar1.style.backgroundColor = "var(--strong)";
            bar2.style.backgroundColor = "var(--strong)";
            bar3.style.backgroundColor = "var(--strong)";
        }

        // Probabilities
        probWeak.textContent = `${(data.probabilities.weak * 100).toFixed(1)}%`;
        probMedium.textContent = `${(data.probabilities.medium * 100).toFixed(1)}%`;
        probStrong.textContent = `${(data.probabilities.strong * 100).toFixed(1)}%`;

        // Feature chips
        chipLen.classList.toggle('active', data.features.length_ok);
        chipUpper.classList.toggle('active', data.features.has_uppercase);
        chipLower.classList.toggle('active', data.features.has_lowercase);
        chipNum.classList.toggle('active', data.features.has_numbers);
        chipSym.classList.toggle('active', data.features.has_special);

        // Suggestions
        if (data.suggestions && data.suggestions.length > 0) {
            suggestionsBox.style.display = 'block';
            suggestionsList.innerHTML = data.suggestions.map(s => `<li>${s}</li>`).join('');
        } else {
            suggestionsBox.style.display = 'none';
        }
    }

    generateBtn.addEventListener('click', () => {
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=";
        let pwd = "";
        const array = new Uint32Array(16);
        window.crypto.getRandomValues(array);
        for (let i = 0; i < 16; i++) {
            pwd += chars[array[i] % chars.length];
        }
        passwordInput.value = pwd;
        evaluatePassword();
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    password = data.get('password', '')
    
    if not password:
        return jsonify({
            'prediction': 0,
            'label': 'Weak',
            'color': '#ef4444',
            'probabilities': {'weak': 0, 'medium': 0, 'strong': 0},
            'entropy': 0,
            'features': {},
            'suggestions': []
        })
    
    # ML Prediction
    vec_input = vectorizer.transform([password])
    prediction = int(model.predict(vec_input)[0])
    probabilities = model.predict_proba(vec_input)[0]
    
    label_map = {0: 'Weak', 1: 'Medium', 2: 'Strong'}
    color_map = {0: '#ef4444', 1: '#f59e0b', 2: '#10b981'}
    
    entropy = calculate_entropy(password)
    suggestions = get_suggestions(password)
    
    features = {
        'length_ok': len(password) >= 8,
        'has_uppercase': bool(re.search(r'[A-Z]', password)),
        'has_lowercase': bool(re.search(r'[a-z]', password)),
        'has_numbers': bool(re.search(r'[0-9]', password)),
        'has_special': bool(re.search(r'[^a-zA-Z0-9]', password))
    }
    
    return jsonify({
        'password': password,
        'prediction': prediction,
        'label': label_map.get(prediction, 'Unknown'),
        'color': color_map.get(prediction, '#ef4444'),
        'probabilities': {
            'weak': float(probabilities[0]),
            'medium': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            'strong': float(probabilities[2]) if len(probabilities) > 2 else 0.0
        },
        'entropy': entropy,
        'features': features,
        'suggestions': suggestions
    })

if __name__ == '__main__':
    print("Starting Flask web server on http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
