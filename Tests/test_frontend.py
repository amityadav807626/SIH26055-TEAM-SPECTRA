from pathlib import Path


def test_frontend_files_exist_and_have_expected_hooks():
    root = Path(__file__).resolve().parents[1] / 'Frontend'
    html = (root / 'index.html').read_text(encoding='utf-8')
    js = (root / 'app.js').read_text(encoding='utf-8')
    css = (root / 'styles.css').read_text(encoding='utf-8')
    assert 'page-live' in html and 'page-performance' in html
    assert '/api/simulation/live' in js
    assert '/api/experiments' in js
    assert '.radar' in css


def test_frontend_chart_scaling_and_controls_are_presentation_safe():
    root = Path(__file__).resolve().parents[1] / 'Frontend'
    html = (root / 'index.html').read_text(encoding='utf-8')
    js = (root / 'app.js').read_text(encoding='utf-8')
    css = (root / 'styles.css').read_text(encoding='utf-8')
    assert 'value="260"><span id="slotsValue">260</span>' in html
    assert 'yMin:0,yMax:100' in js
    assert 'system detection probability (%)' in js
    assert 'preserveAspectRatio="none"' in js
    assert '#runBenchmark:disabled' in css
    assert 'background:#071820!important' in css
    assert 'reward-zero' in css

def test_frontend_has_live_execution_feedback_and_random_default():
    root = Path(__file__).resolve().parents[1] / 'Frontend'
    html = (root / 'index.html').read_text(encoding='utf-8')
    js = (root / 'app.js').read_text(encoding='utf-8')
    css = (root / 'styles.css').read_text(encoding='utf-8')
    assert 'scanOverlay' in html
    assert '0 = RANDOM' in html
    assert 'decisionLatencyChip' in html
    assert 'Decision Latency:' in js
    assert 'Dwell Valid' in js
    assert 'agile_hopping' in html
    assert 'showScanOverlay' in js
    assert 'SCAN COMPLETE' in js
    assert 'scan-overlay' in css


def test_frontend_completed_live_card_and_benchmark_variance():
    root = Path(__file__).resolve().parents[1] / 'Frontend'
    js = (root / 'app.js').read_text(encoding='utf-8')
    assert 'P_D:r.P_D,tp:r.tp' in js
    assert 'BENCHMARK RESULTS · MEAN ± SD' in js
    assert "r.P_D_std" in js
    assert "r.avg_intercept_time_std" in js
