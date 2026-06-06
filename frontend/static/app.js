/**
 * ICU Early Warning System — Frontend Logic
 */
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('patient-form');
    const btnPredict = document.getElementById('btn-predict');
    const inputSection = document.getElementById('input-section');
    const resultsSection = document.getElementById('results-section');
    const btnReset = document.getElementById('btn-reset');

    // Arc length for gauge (half circle)
    const GAUGE_ARC = 251.2;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        btnPredict.classList.add('loading');
        btnPredict.querySelector('.btn-text').textContent = 'Analyzing...';

        const data = {
            age: parseFloat(document.getElementById('age').value),
            gender: document.querySelector('input[name="gender"]:checked').value,
            heart_rate_mean: parseFloat(document.getElementById('heart_rate_mean').value),
            heart_rate_min: parseFloat(document.getElementById('heart_rate_min').value),
            heart_rate_max: parseFloat(document.getElementById('heart_rate_max').value),
            systolic_bp_mean: parseFloat(document.getElementById('systolic_bp_mean').value),
            systolic_bp_min: parseFloat(document.getElementById('systolic_bp_min').value),
            systolic_bp_max: parseFloat(document.getElementById('systolic_bp_max').value),
            diastolic_bp_mean: parseFloat(document.getElementById('diastolic_bp_mean').value),
            diastolic_bp_min: parseFloat(document.getElementById('diastolic_bp_min').value),
            diastolic_bp_max: parseFloat(document.getElementById('diastolic_bp_max').value),
            admit_hour: parseInt(document.getElementById('admit_hour').value),
            admit_dayofweek: parseInt(document.getElementById('admit_dayofweek').value),
        };

        // Optional labs — send only when provided (blank => server uses median).
        const lactate = document.getElementById('lactate_max').value;
        const creatinine = document.getElementById('creatinine_max').value;
        if (lactate !== '') data.lactate_max = parseFloat(lactate);
        if (creatinine !== '') data.creatinine_max = parseFloat(creatinine);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            if (result.error) {
                alert('Error: ' + result.error);
                return;
            }

            displayResults(result);
        } catch (err) {
            alert('Connection error: ' + err.message);
        } finally {
            btnPredict.classList.remove('loading');
            btnPredict.querySelector('.btn-text').textContent = 'Assess Deterioration Risk';
        }
    });

    btnReset.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        inputSection.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    function displayResults(result) {
        inputSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Gauge
        const score = result.risk_score;
        const fillLength = score * GAUGE_ARC;
        const gaugeFill = document.getElementById('gauge-fill');
        gaugeFill.style.stroke = result.risk_color;
        // Animate after small delay
        setTimeout(() => {
            gaugeFill.setAttribute('stroke-dasharray', `${fillLength} ${GAUGE_ARC}`);
        }, 100);

        // Value
        const gaugeValue = document.getElementById('gauge-value');
        gaugeValue.style.color = result.risk_color;
        animateCounter(gaugeValue, 0, Math.round(score * 100), 1000);

        // Badge
        const badge = document.getElementById('risk-badge');
        badge.textContent = result.risk_level;
        badge.style.background = result.risk_color + '20';
        badge.style.color = result.risk_color;
        badge.style.border = `1px solid ${result.risk_color}40`;

        // Risk card border
        const riskCard = document.getElementById('risk-card');
        riskCard.style.borderColor = result.risk_color + '40';
        riskCard.style.boxShadow = `0 0 40px ${result.risk_color}10`;

        // Prediction text
        const prediction = document.getElementById('risk-prediction');
        prediction.textContent = result.prediction;
        prediction.style.color = result.risk_color;

        // Alerts
        const alertsCard = document.getElementById('alerts-card');
        const alertsList = document.getElementById('alerts-list');
        alertsList.innerHTML = '';
        if (result.clinical_alerts && result.clinical_alerts.length > 0) {
            alertsCard.classList.remove('hidden');
            result.clinical_alerts.forEach(alert => {
                const div = document.createElement('div');
                div.className = `alert-item ${alert.level}`;
                div.innerHTML = `
                    <span class="alert-icon">${alert.icon}</span>
                    <span class="alert-text">${alert.text}</span>
                `;
                alertsList.appendChild(div);
            });
        } else {
            alertsCard.classList.add('hidden');
        }

        // Key factors
        const factorsList = document.getElementById('factors-list');
        factorsList.innerHTML = '';
        if (result.key_factors) {
            const maxImp = Math.max(...result.key_factors.map(f => f.importance));
            result.key_factors.forEach(factor => {
                const pct = (factor.importance / maxImp * 100).toFixed(0);
                const div = document.createElement('div');
                div.className = 'factor-item';
                div.innerHTML = `
                    <span class="factor-name">${factor.name}</span>
                    <div class="factor-bar-container">
                        <div class="factor-bar" style="width: 0%"></div>
                    </div>
                    <span class="factor-value">${formatValue(factor.value)}</span>
                `;
                factorsList.appendChild(div);
                // Animate bar
                setTimeout(() => {
                    div.querySelector('.factor-bar').style.width = pct + '%';
                }, 200);
            });
        }

        // Computed indicators
        const indicatorsGrid = document.getElementById('indicators-grid');
        indicatorsGrid.innerHTML = '';
        const indicators = result.computed_indicators;
        const indicatorDefs = [
            { key: 'shock_index', label: 'Shock Index', unit: 'HR/SBP', warn: v => v > 0.9 },
            { key: 'MAP', label: 'MAP', unit: 'mmHg', warn: v => v < 65 },
            { key: 'pulse_pressure', label: 'Pulse Pressure', unit: 'mmHg', warn: v => v < 25 },
            { key: 'HR_range', label: 'HR Range', unit: 'bpm', warn: v => v > 40 },
            { key: 'SBP_range', label: 'SBP Range', unit: 'mmHg', warn: v => v > 50 },
            { key: 'BP_range', label: 'DBP Range', unit: 'mmHg', warn: v => v > 30 },
        ];
        indicatorDefs.forEach(def => {
            const val = indicators[def.key];
            const isWarn = def.warn(val);
            const div = document.createElement('div');
            div.className = 'indicator-item';
            div.innerHTML = `
                <div class="indicator-label">${def.label}</div>
                <div class="indicator-value" style="color: ${isWarn ? 'var(--orange)' : 'var(--text-primary)'}">${val}</div>
                <div class="indicator-unit">${def.unit}</div>
            `;
            indicatorsGrid.appendChild(div);
        });

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function animateCounter(el, start, end, duration) {
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            const current = Math.round(start + (end - start) * eased);
            el.textContent = current + '%';
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }

    function formatValue(val) {
        if (typeof val === 'number') {
            return val >= 1000 ? val.toLocaleString() :
                   val % 1 !== 0 ? val.toFixed(2) : val.toString();
        }
        return val;
    }
});
