/* ─── SymptomAI Frontend Script (Enhanced) ───────────────── */

const API_BASE = window.location.origin;

// ─── Navigation ──────────────────────────────────────────────

const navbar = document.getElementById('navbar');
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');

window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 50);
});

navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
});

navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => navLinks.classList.remove('open'));
});


// ─── EHR History Chips ───────────────────────────────────────

document.querySelectorAll('.ehr-history-chip').forEach(chip => {
    chip.addEventListener('click', () => chip.classList.toggle('active'));
});

function getEHRContext() {
    const age = document.getElementById('ehrAge')?.value;
    const sex = document.getElementById('ehrSex')?.value;
    const activeChips = document.querySelectorAll('.ehr-history-chip.active');
    const history = Array.from(activeChips).map(c => c.dataset.condition);
    return {
        age: age ? parseInt(age) : null,
        sex: sex || null,
        medical_history: history.length > 0 ? history : null,
    };
}


// ─── Symptom Tags ────────────────────────────────────────────

const symptomInput = document.getElementById('symptomInput');
const tags = document.querySelectorAll('.symptom-tag');

tags.forEach(tag => {
    tag.addEventListener('click', () => {
        tag.classList.toggle('active');
        updateTextFromTags();
    });
});

function updateTextFromTags() {
    const activeTags = document.querySelectorAll('.symptom-tag.active');
    const tagTexts = Array.from(activeTags).map(t => t.dataset.symptom);
    let cleaned = symptomInput.value.trim();
    document.querySelectorAll('.symptom-tag').forEach(t => {
        cleaned = cleaned.replace(new RegExp('\\b' + t.dataset.symptom + '\\b', 'gi'), '');
    });
    cleaned = cleaned.replace(/\s{2,}/g, ' ').trim();
    const parts = cleaned ? [cleaned, ...tagTexts] : tagTexts;
    symptomInput.value = parts.join(' ');
}


// ─── Symptom Analysis ────────────────────────────────────────

async function analyzeSymptoms() {
    const symptoms = symptomInput.value.trim();
    if (!symptoms || symptoms.length < 3) {
        showError('Please describe your symptoms in more detail.');
        return;
    }

    const resultsContent = document.getElementById('resultsContent');
    const placeholder = document.getElementById('resultsPlaceholder');
    const analyzeBtn = document.getElementById('analyzeBtn');

    placeholder.style.display = 'none';
    resultsContent.style.display = 'block';
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '⏳ Analyzing...';
    resultsContent.innerHTML = `
        <div class="loading-spinner">
            <div class="spinner"></div>
            <div class="loading-text">Analyzing symptom patterns...</div>
        </div>`;

    const ehr = getEHRContext();

    // Try Flask backend first, fall back to client-side ML engine
    let data = null;
    try {
        const body = { symptoms, top_k: 5 };
        if (ehr.age) body.age = ehr.age;
        if (ehr.sex) body.sex = ehr.sex;
        if (ehr.medical_history) body.medical_history = ehr.medical_history;

        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!response.ok) throw new Error('Backend unavailable');
        data = await response.json();
    } catch (_) {
        // Backend unavailable → use client-side ML engine
        if (mlEngine && mlEngine.ready) {
            const predictions = mlEngine.predict(
                symptoms, 5, ehr.age, ehr.sex, ehr.medical_history
            );
            data = {
                predictions,
                input_symptoms: symptoms,
                ehr_context: (ehr.age || ehr.sex || ehr.medical_history) ? ehr : null,
                disclaimer: "This is an AI-based screening tool for informational purposes only. " +
                    "It is NOT a substitute for professional medical advice, diagnosis, or treatment.",
            };
        }
    }

    if (data) {
        renderResults(data);
    } else {
        resultsContent.innerHTML = `
            <div class="results-placeholder">
                <div class="results-placeholder-icon">⏳</div>
                <div class="results-placeholder-text" style="color: var(--accent-amber);">
                    ML engine is still loading, please try again in a moment.
                </div>
            </div>`;
    }

    analyzeBtn.disabled = false;
    analyzeBtn.textContent = '🔍 Analyze Symptoms';
}

function showError(message) {
    const resultsContent = document.getElementById('resultsContent');
    document.getElementById('resultsPlaceholder').style.display = 'none';
    resultsContent.style.display = 'block';
    resultsContent.innerHTML = `
        <div class="results-placeholder">
            <div class="results-placeholder-icon">⚠️</div>
            <div class="results-placeholder-text" style="color: var(--accent-amber);">${message}</div>
        </div>`;
}

function renderResults(data) {
    const resultsContent = document.getElementById('resultsContent');
    const predictions = data.predictions;
    if (!predictions || predictions.length === 0) {
        resultsContent.innerHTML = `
            <div class="results-placeholder">
                <div class="results-placeholder-icon">🤷</div>
                <div class="results-placeholder-text">No matching conditions found. Try providing more details.</div>
            </div>`;
        return;
    }

    const top = predictions[0];
    const severityClass = `severity-${top.severity.toLowerCase()}`;

    let html = `
        <div class="top-diagnosis">
            <div class="top-diagnosis-label">🏆 Primary Diagnosis</div>
            <div class="top-diagnosis-name">${top.disease}</div>
            <div class="top-diagnosis-confidence">Confidence: <strong>${(top.confidence * 100).toFixed(1)}%</strong></div>
            <div class="top-diagnosis-meta">
                <span class="meta-badge ${severityClass}">${top.severity} Severity</span>
                <span class="meta-badge" style="background:rgba(59,130,246,0.15);color:#60a5fa;border:1px solid rgba(59,130,246,0.3);">${top.category}</span>
            </div>
            ${top.description ? `<p style="font-size:0.88rem;color:var(--text-secondary);margin-top:0.8rem;line-height:1.5;">${top.description}</p>` : ''}
            ${top.seek_care ? `<p class="seek-care-note">📋 ${top.seek_care}</p>` : ''}
        </div>`;

    // Show EHR context if provided
    if (data.ehr_context) {
        const ctx = data.ehr_context;
        let badges = [];
        if (ctx.age) badges.push(`Age: ${ctx.age}`);
        if (ctx.sex) badges.push(`Sex: ${ctx.sex}`);
        if (ctx.medical_history) badges.push(`History: ${ctx.medical_history.join(', ')}`);
        if (badges.length > 0) {
            html += `<div style="margin-bottom:1rem;">${badges.map(b => `<span class="ehr-context-badge">🏥 ${b}</span>`).join(' ')}</div>`;
        }
    }

    if (predictions.length > 1) {
        html += `<h4 style="font-size:0.85rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.8rem;">Differential Diagnoses</h4>`;
        html += '<div class="differential-list">';
        for (let i = 1; i < predictions.length; i++) {
            const pred = predictions[i];
            const sevClass = `severity-${pred.severity.toLowerCase()}`;
            const barWidth = (pred.confidence * 100).toFixed(1);
            html += `
                <div class="differential-item">
                    <div class="differential-header">
                        <span class="differential-name">${pred.disease}</span>
                        <span class="differential-score">${barWidth}%</span>
                    </div>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill" style="width:0%;" data-width="${barWidth}"></div>
                    </div>
                    ${pred.description ? `<div class="differential-description">${pred.description}</div>` : ''}
                    <div style="display:flex;gap:0.5rem;margin-top:0.5rem;flex-wrap:wrap;">
                        <span class="meta-badge ${sevClass}" style="font-size:0.7rem;">${pred.severity}</span>
                        <span class="meta-badge" style="font-size:0.7rem;background:rgba(59,130,246,0.1);color:#60a5fa;border:1px solid rgba(59,130,246,0.2);">${pred.category}</span>
                    </div>
                    ${pred.seek_care ? `<p class="seek-care-note" style="font-size:0.75rem;">📋 ${pred.seek_care}</p>` : ''}
                </div>`;
        }
        html += '</div>';
    }

    html += `<div class="results-disclaimer">⚠️ <strong>Important:</strong> ${data.disclaimer}</div>`;
    resultsContent.innerHTML = html;

    setTimeout(() => {
        document.querySelectorAll('.confidence-bar-fill').forEach(bar => {
            bar.style.width = bar.dataset.width + '%';
        });
    }, 100);
}


// ─── Charts ──────────────────────────────────────────────────

function initCharts() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.plugins.legend.labels.boxWidth = 15;
    Chart.defaults.plugins.legend.labels.padding = 15;

    const barCtx = document.getElementById('barChart');
    if (barCtx) {
        new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['Rule-Based Tool', 'Human Physicians', 'Proposed ML System'],
                datasets: [
                    {
                        label: 'M1 Accuracy (%)',
                        data: [32.4, 88.2, 91.7],
                        backgroundColor: ['rgba(239,68,68,0.7)', 'rgba(59,130,246,0.7)', 'rgba(20,184,166,0.7)'],
                        borderColor: ['rgba(239,68,68,1)', 'rgba(59,130,246,1)', 'rgba(20,184,166,1)'],
                        borderWidth: 2, borderRadius: 8,
                    },
                    {
                        label: 'F1-Score',
                        data: [41, 89, 87],
                        backgroundColor: ['rgba(239,68,68,0.3)', 'rgba(59,130,246,0.3)', 'rgba(20,184,166,0.3)'],
                        borderColor: ['rgba(239,68,68,0.8)', 'rgba(59,130,246,0.8)', 'rgba(20,184,166,0.8)'],
                        borderWidth: 2, borderRadius: 8,
                    },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: {
                    y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { callback: v => v + '%' } },
                    x: { grid: { display: false } },
                },
            },
        });
    }

    const radarCtx = document.getElementById('radarChart');
    if (radarCtx) {
        new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: ['M1 Accuracy', 'F1-Score', 'NDCG', 'Ranking Quality', 'Diagnostic Balance'],
                datasets: [
                    { label: 'ML System', data: [91.7, 87, 93, 95, 88], borderColor: 'rgba(20,184,166,1)', backgroundColor: 'rgba(20,184,166,0.15)', borderWidth: 2, pointBackgroundColor: 'rgba(20,184,166,1)', pointRadius: 4 },
                    { label: 'Human Physicians', data: [88.2, 89, 82, 80, 90], borderColor: 'rgba(59,130,246,1)', backgroundColor: 'rgba(59,130,246,0.1)', borderWidth: 2, pointBackgroundColor: 'rgba(59,130,246,1)', pointRadius: 4 },
                    { label: 'Rule-Based Tool', data: [32.4, 41, 45, 35, 38], borderColor: 'rgba(239,68,68,1)', backgroundColor: 'rgba(239,68,68,0.08)', borderWidth: 2, pointBackgroundColor: 'rgba(239,68,68,1)', pointRadius: 4 },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: { r: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.06)' }, angleLines: { color: 'rgba(255,255,255,0.06)' }, pointLabels: { font: { size: 11 } }, ticks: { display: false, stepSize: 20 } } },
            },
        });
    }
}


// ─── NDCG Dashboard ──────────────────────────────────────────

// Fallback data from paper results (used when backend is unavailable, e.g. GitHub Pages)
const NDCG_FALLBACK = {
    ndcg_score: 0.93,
    comparison: { our_system_paper: 0.93, our_system_live: 0.84, human_physicians: 0.82, rule_based_tool: 0.45 },
};

function renderNDCG(data) {
    const score = data.ndcg_score;
    const circumference = 314.16;
    const offset = circumference * (1 - score);
    const gaugeFill = document.getElementById('ndcgGaugeFill');
    const gaugeValue = document.getElementById('ndcgGaugeValue');
    if (gaugeFill && gaugeValue) {
        gaugeValue.textContent = score.toFixed(2);
        setTimeout(() => { gaugeFill.style.strokeDashoffset = offset; }, 300);
    }
    const compList = document.getElementById('ndcgComparisonList');
    if (compList && data.comparison) {
        const items = [
            { name: '🏆 Our System (Paper)', score: data.comparison.our_system_paper, color: '#14b8a6' },
            { name: '🤖 Our System (Live)', score: data.comparison.our_system_live, color: '#8b5cf6' },
            { name: '👨‍⚕️ Human Physicians', score: data.comparison.human_physicians, color: '#3b82f6' },
            { name: '📋 Rule-Based Tool', score: data.comparison.rule_based_tool, color: '#ef4444' },
        ];
        compList.innerHTML = items.map(item => `
            <div class="ndcg-comparison-item">
                <div class="ndcg-comparison-header">
                    <span class="ndcg-comparison-name">${item.name}</span>
                    <span class="ndcg-comparison-score" style="color:${item.color};">${item.score.toFixed(2)}</span>
                </div>
                <div class="confidence-bar-track">
                    <div class="confidence-bar-fill" style="width:${item.score * 100}%;background:${item.color};"></div>
                </div>
            </div>`).join('');
    }
}

async function loadNDCGDashboard() {
    try {
        const response = await fetch(`${API_BASE}/api/ndcg`);
        if (!response.ok) throw new Error('API unavailable');
        renderNDCG(await response.json());
    } catch (err) {
        renderNDCG(NDCG_FALLBACK);
    }
}


// ─── Bias Detection Panel ────────────────────────────────────

const BIAS_FALLBACK = {
    'Endocrine': { accuracy: 1.0, samples: 5 },
    'Gastrointestinal': { accuracy: 0.875, samples: 8 },
    'Dermatological': { accuracy: 0.75, samples: 4 },
    'Cardiovascular': { accuracy: 0.667, samples: 6 },
    'Mental Health': { accuracy: 0.667, samples: 3 },
    'Musculoskeletal': { accuracy: 0.6, samples: 5 },
    'Neurological': { accuracy: 0.571, samples: 7 },
    'Respiratory': { accuracy: 0.5, samples: 6 },
    'Renal': { accuracy: 0.5, samples: 2 },
    'Infectious Disease': { accuracy: 0.5, samples: 6 },
};

function renderBias(report) {
    if (!report || Object.keys(report).length === 0) return;

    const sorted = Object.entries(report).sort((a, b) => b[1].accuracy - a[1].accuracy);
    const accs = sorted.map(([, v]) => v.accuracy);
    const avgAcc = accs.reduce((a, b) => a + b, 0) / accs.length;
    const minAcc = Math.min(...accs);
    const maxAcc = Math.max(...accs);
    const spread = maxAcc - minAcc;

    const barsContainer = document.getElementById('biasBarsContainer');
    barsContainer.innerHTML = sorted.map(([cat, stats]) => {
        const pct = (stats.accuracy * 100).toFixed(1);
        const color = stats.accuracy >= 0.7 ? '#14b8a6' : stats.accuracy >= 0.4 ? '#f59e0b' : '#ef4444';
        return `
            <div class="bias-bar-item">
                <div class="bias-bar-header">
                    <span class="bias-bar-category">${cat}</span>
                    <span class="bias-bar-accuracy" style="color:${color};">${pct}%</span>
                </div>
                <div class="bias-bar-track">
                    <div class="bias-bar-fill" style="width:${pct}%;background:${color};"></div>
                </div>
                <span class="bias-bar-samples">${stats.samples} test samples</span>
            </div>`;
    }).join('');

    const summaryEl = document.getElementById('biasSummaryContent');
    const alertClass = spread > 0.4 ? 'warning' : 'success';
    const alertIcon = spread > 0.4 ? '⚠️' : '✅';
    const alertMsg = spread > 0.4
        ? `Significant accuracy disparity detected (${(spread * 100).toFixed(0)}% spread). Some disease categories may be underrepresented in training data, indicating potential data bias.`
        : `Accuracy is relatively balanced across categories (${(spread * 100).toFixed(0)}% spread). Minimal bias detected.`;

    summaryEl.innerHTML = `
        <div class="bias-alert ${alertClass}">${alertIcon} ${alertMsg}</div>
        <p style="font-size:0.85rem;color:var(--text-secondary);line-height:1.6;margin-bottom:1rem;">
            The paper recommends "removing hidden data biases and improving inclusion of all population groups"
            as a key future work direction. This panel monitors accuracy fairness across disease categories.
        </p>
        <div class="bias-stat-grid">
            <div class="bias-stat-item">
                <div class="bias-stat-value" style="color:var(--accent-teal);">${(avgAcc * 100).toFixed(1)}%</div>
                <div class="bias-stat-label">Avg. Accuracy</div>
            </div>
            <div class="bias-stat-item">
                <div class="bias-stat-value" style="color:${spread > 0.4 ? 'var(--accent-amber)' : 'var(--accent-green)'};"> ${(spread * 100).toFixed(1)}%</div>
                <div class="bias-stat-label">Accuracy Spread</div>
            </div>
            <div class="bias-stat-item">
                <div class="bias-stat-value" style="color:var(--accent-green);">${(maxAcc * 100).toFixed(1)}%</div>
                <div class="bias-stat-label">Best Category</div>
            </div>
            <div class="bias-stat-item">
                <div class="bias-stat-value" style="color:var(--accent-red);">${(minAcc * 100).toFixed(1)}%</div>
                <div class="bias-stat-label">Worst Category</div>
            </div>
        </div>`;
}

async function loadBiasPanel() {
    try {
        const response = await fetch(`${API_BASE}/api/bias`);
        if (!response.ok) throw new Error('API unavailable');
        const data = await response.json();
        renderBias(data.bias_report);
    } catch (err) {
        renderBias(BIAS_FALLBACK);
    }
}


// ─── Scroll Animations ───────────────────────────────────────

function initScrollAnimations() {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
    );
    document.querySelectorAll('.animate-on-scroll').forEach(el => observer.observe(el));
}


// ─── Keyboard Shortcut ──────────────────────────────────────

symptomInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyzeSymptoms();
});


// ─── Initialize ──────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initScrollAnimations();
    loadNDCGDashboard();
    loadBiasPanel();
    // Initialize client-side ML engine (for static hosting fallback)
    if (typeof mlEngine !== 'undefined') mlEngine.initialize();
});
