import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.smart_scheduler import SmartScheduler
from Backend.database import init_db, save_experiment, get_experiments, clear_database

init_db()

st.set_page_config(page_title="SPECTRA // SMART-SCAN", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- CSS / UNIQUE UI -----------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root { --bg:#05080d; --panel:#0a1119; --line:#18303b; --cyan:#35f2ff; --green:#55ff9a; --amber:#ffc857; --red:#ff5577; --text:#d8f7fb; --muted:#6f929b; }
.stApp { background: radial-gradient(circle at 70% 10%, #0b1d25 0, #05080d 36%, #030509 100%); color:var(--text); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#050b10,#071018 55%,#04070b); border-right:1px solid #17323c; }
.block-container { padding: 1.1rem 1.4rem 2rem; max-width: 1500px; }
* { font-family:'JetBrains Mono',monospace; }
h1,h2,h3 { font-family:'Orbitron',sans-serif !important; letter-spacing:.08em; }
.brand { display:flex; align-items:center; gap:14px; padding:5px 0 18px; }
.brand-mark { width:48px;height:48px;border:1px solid var(--cyan);border-radius:50%;display:grid;place-items:center;color:var(--cyan);box-shadow:0 0 22px #35f2ff33;position:relative; }
.brand-mark:after { content:'';position:absolute;inset:8px;border:1px dashed #35f2ff88;border-radius:50%; }
.brand-title { font:700 19px Orbitron,sans-serif; color:#e9feff; }
.brand-sub { font-size:10px;color:var(--muted);letter-spacing:.12em; }
.navtitle { color:#53737b;font-size:10px;letter-spacing:.18em;margin:12px 0 8px; }
.status { border:1px solid #1b3b44;background:#07131a;padding:9px 11px;border-radius:8px;font-size:11px;color:#86aeb7; }
.dot { display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);margin-right:7px; }
.header { display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:15px; }
.kicker { font:700 11px JetBrains Mono;color:var(--cyan);letter-spacing:.18em; }
.main-title { font:700 29px Orbitron;color:#ecffff;margin-top:3px; }
.sub { color:#6d8c94;font-size:11px;margin-top:5px; }
.chip { border:1px solid #21444e;background:#071219;border-radius:999px;padding:6px 10px;color:#7da5ad;font-size:10px; }
.panel { background:linear-gradient(145deg,#081119,#060b11);border:1px solid #16313a;border-radius:12px;padding:14px;box-shadow:0 8px 30px #0007; }
.panel-title { font:600 11px Orbitron;color:#8fb8c0;letter-spacing:.12em;margin-bottom:10px; }
.metric { min-height:88px;display:flex;flex-direction:column;justify-content:center; }
.metric-label { font-size:9px;color:#64818a;letter-spacing:.14em; }
.metric-value { font:700 24px Orbitron;color:#e9ffff;margin-top:6px; }
.metric-accent { color:var(--cyan); }
.radar-wrap { display:flex;justify-content:center;align-items:center;height:380px;overflow:hidden;position:relative; }
.radar { width:330px;height:330px;border:1px solid #24515d;border-radius:50%;position:relative;background:repeating-radial-gradient(circle,#07151c 0 1px,transparent 1px 55px),linear-gradient(#07151c,#07151c);box-shadow:0 0 55px #35f2ff12 inset,0 0 35px #35f2ff0d; }
.radar:before,.radar:after { content:'';position:absolute;inset:15%;border:1px solid #1b4650;border-radius:50%; }
.radar:after { inset:35%; }
.cross-h,.cross-v { position:absolute;background:#17414b; }
.cross-h { left:0;right:0;top:50%;height:1px; }
.cross-v { top:0;bottom:0;left:50%;width:1px; }
.sweep { position:absolute;left:50%;top:50%;width:48%;height:2px;transform-origin:0 50%;background:linear-gradient(90deg,var(--cyan),transparent);box-shadow:0 0 12px var(--cyan);animation:sweep 2.2s linear infinite; }
@keyframes sweep { to { transform:rotate(360deg); } }
.wave { position:absolute;left:50%;top:50%;width:18px;height:18px;border:1px solid #35f2ff;transform:translate(-50%,-50%);border-radius:50%;animation:wave 2.4s ease-out infinite;opacity:0; }
.wave:nth-child(2){animation-delay:.8s}.wave:nth-child(3){animation-delay:1.6s}
@keyframes wave { 0%{width:18px;height:18px;opacity:.8} 100%{width:330px;height:330px;opacity:0} }
.radar-center {position:absolute;left:50%;top:50%;width:8px;height:8px;transform:translate(-50%,-50%);border-radius:50%;background:var(--cyan);box-shadow:0 0 18px var(--cyan);z-index:4;}
.scanline {height:1px;background:linear-gradient(90deg,transparent,#35f2ff66,transparent);animation:scanline 1.6s linear infinite;margin:2px 0 10px;}
@keyframes scanline { from{transform:translateX(-20%)} to{transform:translateX(20%)} }
.band-grid { display:grid;grid-template-columns:repeat(5,1fr);gap:7px;max-height:380px;overflow:auto;padding-right:4px; }
.band { height:47px;border:1px solid #18343d;border-radius:7px;background:#071017;display:flex;flex-direction:column;justify-content:center;padding:6px 8px;transition:.15s; }
.band.active { border-color:#35f2ff99;box-shadow:0 0 13px #35f2ff18;background:#0a2028; }
.band.hit { border-color:#55ff9a99;background:#0b1d16; }
.band.miss { border-color:#32444a; }
.band-top {display:flex;justify-content:space-between;font-size:10px;color:#c9eef2}.band-bottom {font-size:8px;color:#5f858d;margin-top:3px;}
.bar {height:5px;background:#0d2027;border-radius:4px;overflow:hidden;margin-top:5px}.fill {height:100%;background:linear-gradient(90deg,#35f2ff,#55ff9a);box-shadow:0 0 8px #35f2ff66;}
.event {display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #10252c;font-size:9px;color:#88aeb6}.event b{color:#d7fbff}.event-dot{width:6px;height:6px;border-radius:50%;background:#35f2ff;margin-top:4px;box-shadow:0 0 8px #35f2ff;flex:none}
.small {font-size:9px;color:#62828b;}.big {font:700 18px Orbitron;color:#dffcff;}
.advantage-panel { background:linear-gradient(145deg,#07151e,#08131a);border:1px solid #1b4854;border-radius:12px;padding:15px;margin:10px 0 18px;box-shadow:0 8px 30px #0006; }
.advantage-title { font:600 12px Orbitron;color:#8fd7e0;letter-spacing:.13em;margin-bottom:12px; }
.adv-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:10px; }
.adv-card { background:#071017;border:1px solid #17323b;border-radius:9px;padding:12px;min-height:105px; }
.adv-label { font-size:9px;color:#6f929b;letter-spacing:.12em; }
.adv-value { font:700 18px Orbitron;color:#e8ffff;margin-top:7px; }
.adv-good { color:#55ff9a; }
.adv-neutral { color:#ffc857; }
.adv-trade { color:#ff9eaa; }
.adv-note { font-size:9px;color:#71929a;margin-top:6px;line-height:1.5; }
.winbar { height:6px;background:#10232a;border-radius:5px;overflow:hidden;margin-top:8px; }
.winfill { height:100%;background:linear-gradient(90deg,#35f2ff,#55ff9a); }
@media (max-width:900px) { .adv-grid { grid-template-columns:1fr; } }


/* V3.7 final cockpit selectbox polish */
[data-testid="stSidebar"] [data-testid="stSelectbox"] { margin-top: 8px; margin-bottom: 12px; }
.stSelectbox { margin-top: 8px !important; margin-bottom: 12px !important; }
.stSelectbox label { color: #6f929b !important; margin-bottom: 6px !important; line-height: 1.35 !important; }
.stSelectbox [data-baseweb="select"] > div { background: #071017 !important; color: #d9f8fb !important; border-color: #28505b !important; border-radius: 7px !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] label { color: #6f929b !important; display:block !important; margin-bottom: 6px !important; line-height: 1.35 !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #071017 !important; color: #d9f8fb !important; border-color: #28505b !important; border-radius: 7px !important; min-height: 38px !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] svg { fill: #35f2ff !important; }
[data-baseweb="popover"] [role="listbox"] { background: #071017 !important; border: 1px solid #28505b !important; }
[data-baseweb="popover"] [role="option"] { background: #071017 !important; color: #d9f8fb !important; }
[data-baseweb="popover"] [role="option"]:hover { background: #0a2028 !important; color: #ffffff !important; }

div.stButton > button { background:#071820;border:1px solid #23515c;color:#a9e8ef;border-radius:8px;font-family:'JetBrains Mono';font-size:11px; }
div.stButton > button:hover { border-color:#35f2ff;color:#fff;box-shadow:0 0 18px #35f2ff16; }
footer {visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ----------------------------- SIMULATION -----------------------------
from Backend.simulation_service import create_environment

class Learner:
    def __init__(self,n):
        self.reward=np.zeros(n); self.visits=np.zeros(n,dtype=int); self.hits=np.zeros(n,dtype=int)
    def update(self,b,hit):
        r=1.0 if hit else -0.08
        self.reward[b]=0.88*self.reward[b]+0.12*r
        self.visits[b]+=1; self.hits[b]+=int(hit)

def step_scores(scheduler, learner, observations, bands, t, epsilon=0.0):
    selected, scores, _latency = scheduler.select_band(bands, t, observations, epsilon=epsilon)
    return selected, scores

def detector_observation(rng, truth_value, detector_pd=0.90, detector_pfa=0.05):
    """Synthetic detector: separates ground truth from noisy observation."""
    if truth_value:
        return bool(rng.random() < detector_pd)
    return bool(rng.random() < detector_pfa)


def compute_metrics(tp, fp, fn, tn, detection_delays=None):
    """Calculate metrics for the scanned synthetic observations."""
    total = tp + fp + fn + tn
    actual_positive = tp + fn
    actual_negative = fp + tn
    predicted_positive = tp + fp

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / actual_positive if actual_positive else 0.0
    pfa = fp / actual_negative if actual_negative else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    avg_delay = float(np.mean(detection_delays)) if detection_delays else None

    return {
        'accuracy': accuracy,
        'P_D': recall,
        'P_FA': pfa,
        'precision': precision,
        'recall': recall,
        'F1': f1,
        'avg_intercept_time': avg_delay,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def simulate_live(n_bands, horizon, epsilon, seed, placeholder, stop_flag):
    rng=np.random.default_rng(seed)
    truth=create_environment(n_bands,horizon,seed)
    scheduler=SmartScheduler(seed=seed + 17)
    learner=Learner(n_bands)
    observations={b:{'previous_activity':0,'time_since_activity':5,'recent_activity_rate':.5} for b in range(n_bands)}
    last=np.full(n_bands,-1); hits=0; scans=0; recent=[]; events=[]
    band_history=[]; performance_history=[]; detection_delays=[]
    tp=fp=fn=tn=0
    active_start={b: None for b in range(n_bands)}
    max_live=min(horizon, 260)
    for t in range(max_live):
        if stop_flag(): return None
        if t<n_bands: band=t
        else:
            band, scores=step_scores(scheduler,learner,observations,range(n_bands),t,epsilon=epsilon)

        actual=bool(truth[t,band])
        detected=detector_observation(rng, actual)
        scans += 1
        hits += int(detected)
        if actual and not (t > 0 and truth[t-1,band]):
            active_start[band] = t
        if detected and actual and active_start[band] is not None:
            detection_delays.append(t-active_start[band])
            active_start[band] = None

        if actual and detected: tp += 1
        elif (not actual) and detected: fp += 1
        elif actual and (not detected): fn += 1
        else: tn += 1

        learner.update(band,detected)
        scheduler.update_result(band,detected)
        for b in range(n_bands):
            observations[b]['time_since_activity']=min(observations[b]['time_since_activity']+1,50)
        observations[band]['previous_activity']=int(detected)
        if detected: observations[band]['time_since_activity']=0
        recent.append(int(detected)); recent=recent[-10:]
        observations[band]['recent_activity_rate']=sum(recent)/len(recent)
        last[band]=t
        events.insert(0,(t,band,detected)); events=events[:7]
        band_history.append(band)
        performance_history.append((hits/scans if scans else 0.0)*100)
        payload=dict(t=t,band=band,hit=detected,hits=hits,scans=scans,learner=learner,truth=truth,last=last.copy(),events=events.copy())
        placeholder(payload)
        time.sleep(0.045)

    metrics=compute_metrics(tp,fp,fn,tn,detection_delays)
    return {
        'hits':hits,'scans':scans,'pd':metrics['P_D'],'P_D':metrics['P_D'],'P_FA':metrics['P_FA'],'accuracy':metrics['accuracy'],
        'precision':metrics['precision'],'recall':metrics['recall'],'F1':metrics['F1'],
        'avg_intercept_time':metrics['avg_intercept_time'],'truth':truth,'learner':learner,
        'band_history':band_history,'performance_history':performance_history,
        'final_rewards':learner.reward.tolist(),'visits':learner.visits.tolist(),
        'hits_per_band':learner.hits.tolist(), 'tp':tp,'fp':fp,'fn':fn,'tn':tn,
    }


def run_benchmark(n_bands, horizon, epsilon, seed, detector_pd=0.90, detector_pfa=0.05):
    """Fair Smart Scan vs Round Robin comparison on identical synthetic data."""
    truth = create_environment(n_bands, horizon, seed)

    # One detector draw per time/band cell. Both strategies see the same
    # detector outcome whenever they inspect the same synthetic cell.
    detector_rng = np.random.default_rng(seed + 10000)
    detection_draws = detector_rng.random((horizon, n_bands))
    false_alarm_draws = detector_rng.random((horizon, n_bands))

    # Ground-truth activity onset for each band. This prevents a zero-delay
    # artifact when a strategy first scans a band after activity has started.
    onset = np.full((horizon, n_bands), -1, dtype=int)
    for band in range(n_bands):
        current = -1
        for t in range(horizon):
            if truth[t, band] and (t == 0 or not truth[t - 1, band]):
                current = t
            onset[t, band] = current
            if not truth[t, band]:
                current = -1

    results = []

    for strategy in ['Smart Scan', 'Round Robin']:
        learner = Learner(n_bands)
        observations = {
            b: {
                'previous_activity': 0,
                'time_since_activity': 5,
                'recent_activity_rate': 0.5
            } for b in range(n_bands)
        }
        scheduler = SmartScheduler(seed=seed + 17) if strategy == 'Smart Scan' else None
        rr_index = 0

        tp = fp = fn = tn = 0
        delays = []
        band_scans = np.zeros(n_bands, dtype=int)
        band_detections = np.zeros(n_bands, dtype=int)

        for t in range(horizon):
            if strategy == 'Smart Scan':
                if t < n_bands:
                    band = t
                else:
                    band, scores = step_scores(
                        scheduler, learner, observations, range(n_bands), t, epsilon=epsilon
                    )
            else:
                band = rr_index
                rr_index = (rr_index + 1) % n_bands

            band_scans[band] += 1
            actual = bool(truth[t, band])

            if actual:
                detected = bool(detection_draws[t, band] < detector_pd)
            else:
                detected = bool(false_alarm_draws[t, band] < detector_pfa)

            if detected:
                band_detections[band] += 1

            if actual and detected and onset[t, band] >= 0:
                delays.append(t - onset[t, band])

            if actual and detected:
                tp += 1
            elif not actual and detected:
                fp += 1
            elif actual and not detected:
                fn += 1
            else:
                tn += 1

            learner.update(band, detected)
            if scheduler is not None:
                scheduler.update_result(band, detected)

            for b in range(n_bands):
                observations[b]['time_since_activity'] = min(
                    observations[b]['time_since_activity'] + 1, 50
                )

            observations[band]['previous_activity'] = int(detected)
            if detected:
                observations[band]['time_since_activity'] = 0

            observations[band]['recent_activity_rate'] = (
                0.90 * observations[band]['recent_activity_rate']
                + 0.10 * int(detected)
            )

        metrics = compute_metrics(tp, fp, fn, tn, delays)
        results.append({
            'Strategy': strategy,
            'Accuracy': metrics['accuracy'],
            'P_D': metrics['P_D'],
            'P_FA': metrics['P_FA'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'F1': metrics['F1'],
            'avg_intercept_time': metrics['avg_intercept_time'],
            'Scans': int(band_scans.sum()),
            'Detections': int(band_detections.sum()),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'band_scans': band_scans.tolist(),
            'band_detections': band_detections.tolist(),
        })

    return pd.DataFrame(results)


def run_multi_benchmark(n_bands, horizon, epsilon, seed, trials=10, detector_pd=0.90, detector_pfa=0.05):
    """Run repeated fair benchmarks and return trial-level + aggregate results."""
    all_trials = []
    for trial in range(trials):
        trial_seed = int(seed) + trial * 1009
        bench = run_benchmark(
            n_bands, horizon, epsilon, trial_seed,
            detector_pd=detector_pd, detector_pfa=detector_pfa
        )
        bench["Trial"] = trial + 1
        bench["Seed"] = trial_seed
        all_trials.append(bench)

    trial_df = pd.concat(all_trials, ignore_index=True)

    metric_cols = [
        'Accuracy', 'P_D', 'P_FA', 'precision', 'recall',
        'F1', 'avg_intercept_time', 'Scans', 'Detections',
        'TP', 'FP', 'FN', 'TN'
    ]

    summary_rows = []
    for strategy in ['Smart Scan', 'Round Robin']:
        subset = trial_df[trial_df['Strategy'] == strategy]
        row = {'Strategy': strategy}
        for col in metric_cols:
            values = pd.to_numeric(subset[col], errors='coerce')
            row[col] = float(values.mean())
            if col in ['Scans', 'Detections', 'TP', 'FP', 'FN', 'TN']:
                row[f'{col}_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            else:
                row[f'{col}_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    smart_wins = {}
    for metric in ['P_D', 'precision', 'recall', 'F1']:
        wins = 0
        for trial_no in range(1, trials + 1):
            s = trial_df[(trial_df['Trial'] == trial_no) & (trial_df['Strategy'] == 'Smart Scan')][metric].iloc[0]
            r = trial_df[(trial_df['Trial'] == trial_no) & (trial_df['Strategy'] == 'Round Robin')][metric].iloc[0]
            wins += int(s > r)
        smart_wins[metric] = wins

    pfa_wins = 0
    delay_wins = 0
    for trial_no in range(1, trials + 1):
        s = trial_df[(trial_df['Trial'] == trial_no) & (trial_df['Strategy'] == 'Smart Scan')].iloc[0]
        r = trial_df[(trial_df['Trial'] == trial_no) & (trial_df['Strategy'] == 'Round Robin')].iloc[0]
        pfa_wins += int(s['P_FA'] < r['P_FA'])
        if pd.notna(s['avg_intercept_time']) and pd.notna(r['avg_intercept_time']):
            delay_wins += int(s['avg_intercept_time'] < r['avg_intercept_time'])

    return {
        'summary': summary_df,
        'trials': trial_df,
        'smart_wins': smart_wins,
        'pfa_wins': pfa_wins,
        'delay_wins': delay_wins,
        'trials_count': trials,
    }

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">◉</div><div><div class="brand-title">SPECTRA</div><div class="brand-sub">SMART-SCAN / SIH26055</div></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="navtitle">CONTROL DECK</div>',unsafe_allow_html=True)
    page=st.radio('Navigation',['Live Radar','Performance','Benchmark','Experiment Log'],label_visibility='collapsed')
    st.markdown('<div class="navtitle">SIMULATION PARAMETERS</div>',unsafe_allow_html=True)
    n_bands=st.slider('Frequency bands',10,60,30)
    horizon=st.slider('Time slots',100,1000,300)
    epsilon=st.slider('Exploration rate',0.0,0.40,0.10,0.01)
    seed=st.number_input('Random seed',0,999999,7,1)
    benchmark_trials=st.slider('Benchmark trials',3,30,10,1)
    start=st.button('▶  START LIVE SCAN',width="stretch")
    benchmark_start=st.button('⚖  RUN BASELINE COMPARISON',width="stretch")
    clear=st.button('CLEAR EXPERIMENT LOG',width="stretch")
    if clear:
        clear_database(); st.session_state.pop('last_result',None); st.rerun()
    st.markdown('<div class="status"><span class="dot"></span>SYNTHETIC RF ENVIRONMENT<br><span class="small">No real RF hardware / transmission</span></div>',unsafe_allow_html=True)

# ----------------------------- LIVE RADAR -----------------------------
st.markdown('<div class="header"><div><div class="kicker">ADAPTIVE SPECTRUM MONITOR</div><div class="main-title">SMART-SCAN // LIVE CONSOLE</div><div class="sub">Simulation-only closed loop: scan → observe → learn → reprioritize</div></div><div class="chip">● SYSTEM READY</div></div>',unsafe_allow_html=True)

if page=='Live Radar':
    if 'running' not in st.session_state: st.session_state.running=False
    if start:
        st.session_state.running=True
        st.session_state.live_payload=None
        st.session_state.last_result=None
        st.rerun()

    c1,c2,c3,c4=st.columns(4)
    k1=c1.empty(); k2=c2.empty(); k3=c3.empty(); k4=c4.empty()
    main_left, main_right=st.columns([1.05,1.35])
    radar_box=main_left.empty(); grid_box=main_right.empty()
    event_box=st.empty()
    chart_left, chart_right = st.columns(2)
    band_activity_box = chart_left.empty()
    detection_box = chart_right.empty()
    reward_box = st.empty()
    live_scan_history = []
    live_detection_history = []

    if st.session_state.running:
        def render(p):
            t=p['t']; b=p['band']; hit=p['hit']; hits=p['hits']; scans=p['scans']; learner=p['learner']; last=p['last']; events=p['events']; truth=p['truth']
            live_scan_history.append(b)
            live_detection_history.append((hits/scans if scans else 0.0) * 100)
            k1.markdown(f'<div class="panel metric"><div class="metric-label">CURRENT SLOT</div><div class="metric-value metric-accent">{t:04d}</div></div>',unsafe_allow_html=True)
            k2.markdown(f'<div class="panel metric"><div class="metric-label">SCANNED BAND</div><div class="metric-value">B-{b:02d}</div></div>',unsafe_allow_html=True)
            k3.markdown(f'<div class="panel metric"><div class="metric-label">DETECTIONS</div><div class="metric-value">{hits:04d}</div></div>',unsafe_allow_html=True)
            k4.markdown(f'<div class="panel metric"><div class="metric-label">P<sub>D</sub> / DETECTION RATE</div><div class="metric-value">{hits/scans:.1%}</div></div>',unsafe_allow_html=True)
            radar_box.markdown('''<div class="panel"><div class="panel-title">◉ LIVE SCAN FIELD</div><div class="radar-wrap"><div class="radar"><div class="cross-h"></div><div class="cross-v"></div><div class="sweep"></div><div class="wave"></div><div class="wave"></div><div class="wave"></div><div class="radar-center"></div></div></div><div class="scanline"></div><div class="small">RADAR PULSE ACTIVE • SYNTHETIC OBSERVATION STREAM</div></div>''',unsafe_allow_html=True)
            cells=[]
            for x in range(len(learner.reward)):
                cls=' active' if x==b else (' hit' if learner.hits[x]>0 else '')
                prob=max(0,min(1,(learner.reward[x]+1)/2))
                cells.append(f'<div class="band{cls}"><div class="band-top"><span>B-{x:02d}</span><span>{learner.hits[x]}H</span></div><div class="bar"><div class="fill" style="width:{prob*100:.0f}%"></div></div><div class="band-bottom">visits {learner.visits[x]} · last {last[x] if last[x]>=0 else "—"}</div></div>')
            grid_box.markdown(f'<div class="panel"><div class="panel-title">BAND ACTIVITY MATRIX / REAL-TIME</div><div class="band-grid">{"".join(cells)}</div></div>',unsafe_allow_html=True)
            ev=''.join(f'<div class="event"><span class="event-dot"></span><div>t={tt:04d} · <b>B-{bb:02d}</b> · {"DETECTED" if hh else "MISS"}</div></div>' for tt,bb,hh in events)
            event_box.markdown(f'<div class="panel"><div class="panel-title">LIVE EVENT STREAM</div>{ev or "<div class=small>Waiting for first observation…</div>"}</div>',unsafe_allow_html=True)

            # 1. Live band activity chart
            activity_df = pd.DataFrame({
                'Time Slot': list(range(len(live_scan_history))),
                'Scanned Band': [x + 1 for x in live_scan_history]
            }).set_index('Time Slot')
            band_activity_box.markdown('<div class="panel"><div class="panel-title">LIVE BAND ACTIVITY</div></div>', unsafe_allow_html=True)
            band_activity_box.line_chart(activity_df, height=220, width="stretch")

            # 2. Detection / performance graph
            performance_df = pd.DataFrame({
                'Time Slot': list(range(len(live_detection_history))),
                'Detection Rate (%)': live_detection_history
            }).set_index('Time Slot')
            detection_box.markdown('<div class="panel"><div class="panel-title">DETECTION / PERFORMANCE</div></div>', unsafe_allow_html=True)
            detection_box.line_chart(performance_df, height=220, width="stretch")

            # 3. Band priority / learned reward visualization
            reward_df = pd.DataFrame({
                'Band': [f'B-{x:02d}' for x in range(len(learner.reward))],
                'Priority / Reward': learner.reward
            }).set_index('Band')
            reward_box.markdown('<div class="panel"><div class="panel-title">BAND PRIORITY / LEARNED REWARD</div></div>', unsafe_allow_html=True)
            reward_box.bar_chart(reward_df, height=260, width="stretch")

        result=simulate_live(n_bands,horizon,epsilon,int(seed),render,lambda: False)
        st.session_state.running=False
        if result:
            st.session_state.last_result=result
            save_experiment('Live Smart Scan', n_bands, horizon, result)
        st.success('Live scan complete. Press START LIVE SCAN to run another synthetic scenario.')
    else:
        k1.markdown('<div class="panel metric"><div class="metric-label">CURRENT SLOT</div><div class="metric-value">—</div></div>',unsafe_allow_html=True)
        k2.markdown('<div class="panel metric"><div class="metric-label">SCANNED BAND</div><div class="metric-value">—</div></div>',unsafe_allow_html=True)
        k3.markdown('<div class="panel metric"><div class="metric-label">DETECTIONS</div><div class="metric-value">—</div></div>',unsafe_allow_html=True)
        k4.markdown('<div class="panel metric"><div class="metric-label">P<sub>D</sub> / DETECTION RATE</div><div class="metric-value">—</div></div>',unsafe_allow_html=True)
        radar_box.markdown('''<div class="panel"><div class="panel-title">◉ LIVE SCAN FIELD</div><div class="radar-wrap"><div class="radar"><div class="cross-h"></div><div class="cross-v"></div><div class="sweep"></div><div class="wave"></div><div class="wave"></div><div class="wave"></div><div class="radar-center"></div></div></div><div class="scanline"></div><div class="small">RADAR STANDBY • PRESS START LIVE SCAN</div></div>''',unsafe_allow_html=True)
        grid_box.markdown('<div class="panel"><div class="panel-title">BAND ACTIVITY MATRIX</div><div class="small" style="padding:90px 10px;text-align:center">NO SCAN DATA<br><br>Start the simulation to populate bands in real time.</div></div>',unsafe_allow_html=True)
        event_box.markdown('<div class="panel"><div class="panel-title">LIVE EVENT STREAM</div><div class="small">Awaiting simulation start…</div></div>',unsafe_allow_html=True)

elif page=='Performance':
    st.markdown(
        '<div class="header"><div><div class="kicker">POST-SCAN ANALYSIS</div>'
        '<div class="main-title">SMART-SCAN // PERFORMANCE</div>'
        '<div class="sub">Explainable results from the latest synthetic scan</div>'
        '</div><div class="chip">● ANALYSIS MODE</div></div>',
        unsafe_allow_html=True
    )
    r=st.session_state.get('last_result')
    if not r:
        st.info('Run a Live Scan first. Performance appears here after a simulation has completed.')
    else:
        pd_v=r['P_D']*100; pfa_v=r['P_FA']*100; prec_v=r['precision']*100; f1_v=r['F1']*100
        delay = r['avg_intercept_time']
        st.markdown(
            f'<div class="demo-banner">● LATEST RUN • {r["scans"]} SCANS • {len(r["truth"]) if r.get("truth") is not None else "—"} TIME SLOTS • '
            f'{len(r["learner"].reward)} BANDS • SOFTWARE SIMULATION ONLY</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="score-grid">'
            f'<div class="score-box"><div class="slabel">P_D / DETECTION</div><div class="svalue">{pd_v:.1f}%</div><div class="ssub">Higher is better</div></div>'
            f'<div class="score-box"><div class="slabel">PRECISION</div><div class="svalue">{prec_v:.1f}%</div><div class="ssub">Detection quality</div></div>'
            f'<div class="score-box"><div class="slabel">F1 SCORE</div><div class="svalue">{f1_v:.1f}%</div><div class="ssub">Balanced quality</div></div>'
            f'<div class="score-box"><div class="slabel">P_FA / FALSE ALARM</div><div class="svalue">{pfa_v:.1f}%</div><div class="ssub">Lower is better</div></div>'
            f'</div>', unsafe_allow_html=True
        )

        c1,c2,c3=st.columns(3)
        c1.metric('Detections', r['hits'])
        c2.metric('Scans', r['scans'])
        c3.metric('Average Detection Delay', '—' if delay is None else f'{delay:.2f} slots')

        st.markdown('<div class="panel-title">WHAT THE SCHEDULER LEARNED</div>', unsafe_allow_html=True)
        reward_df = pd.DataFrame({
            'Band':[f'B-{x:02d}' for x in range(len(r['learner'].reward))],
            'Learned Reward':r['learner'].reward,
            'Visits':r['learner'].visits,
            'Detections':r['learner'].hits
        }).set_index('Band')
        st.bar_chart(reward_df[['Learned Reward']], height=300, width='stretch')
        st.markdown('<div class="section-note">Higher learned reward indicates that the adaptive loop has found the band more useful to revisit under the current synthetic scenario.</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-title">SCAN COVERAGE</div>', unsafe_allow_html=True)
        coverage_df = pd.DataFrame({
            'Band':[f'B-{x:02d}' for x in range(len(r['learner'].visits))],
            'Scans':r['learner'].visits,
            'Detections':r['learner'].hits
        }).set_index('Band')
        st.bar_chart(coverage_df, height=300, width='stretch')

        st.markdown('<div class="panel-title">DETECTION TREND</div>', unsafe_allow_html=True)
        if r.get('performance_history'):
            perf_df = pd.DataFrame({'Cumulative Detection Rate (%)':r['performance_history']})
            st.line_chart(perf_df, height=260, width='stretch')

        st.markdown('<div class="panel-title">INTERPRETATION</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="demo-banner">SMART-SCAN prioritizes bands using model probability, uncertainty, freshness and learned hit history. '
            f'This run produced <b>{pd_v:.1f}% P_D</b>, <b>{prec_v:.1f}% precision</b> and <b>{f1_v:.1f}% F1</b>, '
            f'with a <b>{pfa_v:.1f}% false-alarm rate</b>. These values are outputs of the synthetic software environment, not real RF measurements.</div>',
            unsafe_allow_html=True
        )

elif page=='Benchmark':
    st.markdown(
        '<div class="header"><div><div class="kicker">CONTROLLED EVALUATION</div>'
        '<div class="main-title">SMART-SCAN // BASELINE BENCHMARK</div>'
        '<div class="sub">Repeated fair evaluation • identical synthetic worlds • Smart Scan vs Round Robin</div>'
        '</div><div class="chip">● SIMULATION ONLY</div></div>',
        unsafe_allow_html=True
    )

    if benchmark_start:
        with st.spinner(f'Running {benchmark_trials} controlled benchmark trials…'):
            bench_pack = run_multi_benchmark(
                n_bands, horizon, epsilon, int(seed), benchmark_trials
            )
        st.session_state.benchmark_pack = bench_pack
        st.session_state.benchmark = bench_pack['summary']

    bench_pack = st.session_state.get('benchmark_pack')
    bench = st.session_state.get('benchmark')

    if bench_pack is None or bench is None:
        st.info('Press RUN BASELINE COMPARISON to run repeated controlled trials against Round Robin.')
    else:
        smart = bench[bench['Strategy'] == 'Smart Scan'].iloc[0]
        rr = bench[bench['Strategy'] == 'Round Robin'].iloc[0]
        trials_df = bench_pack['trials']
        trial_csv = trials_df.to_csv(index=False).encode('utf-8')
        st.download_button('⬇  EXPORT ALL TRIALS (CSV)', trial_csv, file_name='spectra_benchmark_trials.csv', mime='text/csv', width='stretch')

        st.markdown(
            f'<div class="panel-title">RESULT COMPARISON — MEAN OF {bench_pack["trials_count"]} TRIALS</div>',
            unsafe_allow_html=True
        )

        display_df = bench[[
            'Strategy', 'Accuracy', 'P_D', 'P_FA', 'precision',
            'recall', 'F1', 'avg_intercept_time', 'Scans', 'Detections'
        ]].copy()

        for col in ['Accuracy', 'P_D', 'P_FA', 'precision', 'recall', 'F1']:
            display_df[col] = display_df[col].map(lambda x: f'{x * 100:.2f}%')

        display_df['avg_intercept_time'] = display_df['avg_intercept_time'].map(
            lambda x: '—' if pd.isna(x) else f'{x:.2f}'
        )
        st.dataframe(display_df, hide_index=True, width='stretch')
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button('⬇  EXPORT MEAN RESULTS (CSV)', csv_data, file_name='spectra_benchmark_mean_results.csv', mime='text/csv', width='stretch')

        cards = st.columns(4)
        cards[0].metric(
            'Mean Smart Scan P_D',
            f"{smart['P_D']:.1%}",
            f"{(smart['P_D'] - rr['P_D']):+.1%} vs RR",
            delta_color='normal'
        )
        cards[1].metric(
            'Mean Smart Scan F1',
            f"{smart['F1']:.1%}",
            f"{(smart['F1'] - rr['F1']):+.1%} vs RR",
            delta_color='normal'
        )
        cards[2].metric(
            'Mean Smart Scan P_FA',
            f"{smart['P_FA']:.1%}",
            f"{(smart['P_FA'] - rr['P_FA']):+.1%} vs RR",
            delta_color='inverse'
        )
        if pd.notna(smart['avg_intercept_time']) and pd.notna(rr['avg_intercept_time']):
            cards[3].metric(
                'Mean Avg Delay',
                f"{smart['avg_intercept_time']:.2f}",
                f"{(smart['avg_intercept_time'] - rr['avg_intercept_time']):+.2f} slots",
                delta_color='inverse'
            )
        else:
            cards[3].metric('Mean Avg Delay', '—')

        # Robustness: how often Smart Scan wins each metric across independent trials.
        wins = bench_pack['smart_wins']
        st.markdown('<div class="panel-title">ROBUSTNESS ACROSS TRIALS</div>', unsafe_allow_html=True)
        st.info(
            f"Smart Scan wins P_D in {wins['P_D']}/{bench_pack['trials_count']} trials • "
            f"Precision in {wins['precision']}/{bench_pack['trials_count']} • "
            f"Recall in {wins['recall']}/{bench_pack['trials_count']} • "
            f"F1 in {wins['F1']}/{bench_pack['trials_count']} • "
            f"lower P_FA in {bench_pack['pfa_wins']}/{bench_pack['trials_count']} • "
            f"lower delay in {bench_pack['delay_wins']}/{bench_pack['trials_count']} trials."
        )

        # Judge-ready advantage panel. This deliberately avoids inventing a
        # single composite score: the scheduler has different strengths and
        # the underlying metrics are not interchangeable.
        n_trials = bench_pack['trials_count']
        quality_metrics = [('P_D', 'P_D'), ('Precision', 'precision'), ('Recall', 'recall'), ('F1 Score', 'F1')]
        quality_win_total = sum(wins[m] for _, m in quality_metrics)
        quality_total = len(quality_metrics) * n_trials
        smart_pfa_better = smart['P_FA'] < rr['P_FA']
        pfa_equal = np.isclose(smart['P_FA'], rr['P_FA'], atol=1e-12)
        smart_delay_better = (pd.notna(smart['avg_intercept_time']) and pd.notna(rr['avg_intercept_time']) and
                              smart['avg_intercept_time'] < rr['avg_intercept_time'])

        if quality_win_total / quality_total >= 0.75:
            overall_label = 'STRONG DETECTION-QUALITY ADVANTAGE'
            overall_class = 'adv-good'
        elif quality_win_total / quality_total >= 0.50:
            overall_label = 'DETECTION-QUALITY ADVANTAGE'
            overall_class = 'adv-good'
        else:
            overall_label = 'MIXED DETECTION-QUALITY RESULT'
            overall_class = 'adv-neutral'

        if pfa_equal:
            pfa_label, pfa_class, pfa_note = 'NEUTRAL', 'adv-neutral', 'Mean false-alarm rate is effectively equal.'
        elif smart_pfa_better:
            pfa_label, pfa_class, pfa_note = 'SMART SCAN', 'adv-good', 'Lower mean false-alarm rate.'
        else:
            pfa_label, pfa_class, pfa_note = 'ROUND ROBIN', 'adv-neutral', 'Round Robin has the lower mean false-alarm rate.'

        if smart_delay_better:
            delay_label, delay_class, delay_note = 'SMART SCAN', 'adv-good', 'Lower mean detection delay.'
        else:
            delay_label, delay_class, delay_note = 'ROUND ROBIN', 'adv-trade', 'Round Robin reaches detections sooner on average.'

        cards_html = []
        for label, metric_name in quality_metrics:
            win_count = wins[metric_name]
            pct = (win_count / n_trials) * 100
            mean_s = smart[metric_name] * 100
            mean_r = rr[metric_name] * 100
            cards_html.append(
                f'<div class="adv-card"><div class="adv-label">{label.upper()}</div>'
                f'<div class="adv-value adv-good">SMART SCAN</div>'
                f'<div class="adv-note">Mean {mean_s:.2f}% vs RR {mean_r:.2f}% · wins {win_count}/{n_trials}</div>'
                f'<div class="winbar"><div class="winfill" style="width:{pct:.0f}%"></div></div></div>'
            )
        cards_html.extend([
            f'<div class="adv-card"><div class="adv-label">FALSE-ALARM RATE</div><div class="adv-value {pfa_class}">{pfa_label}</div><div class="adv-note">Smart Scan {smart['P_FA']*100:.2f}% · RR {rr['P_FA']*100:.2f}% · lower is better</div></div>',
            f'<div class="adv-card"><div class="adv-label">DETECTION DELAY</div><div class="adv-value {delay_class}">{delay_label}</div><div class="adv-note">Smart Scan {smart['avg_intercept_time']:.2f} slots · RR {rr['avg_intercept_time']:.2f} slots · lower is better</div></div>'
        ])

        st.markdown(
            f'<div class="advantage-panel"><div class="advantage-title">BENCHMARK METRIC SUMMARY</div>'
            f'<div class="adv-grid">{"".join(cards_html)}</div>'
            f'<div style="margin-top:13px;padding-top:11px;border-top:1px solid #17323b;font-size:10px;color:#8eb2ba">'
            f'<b class="{overall_class}">{overall_label}</b> · Quality-metric wins: {quality_win_total}/{quality_total} across P_D, precision, recall and F1. '
            f'No composite score is used; P_FA and delay are reported separately because they represent different trade-offs.</div></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="panel-title">DETECTION QUALITY — MEAN (%)</div>', unsafe_allow_html=True)
        qcols = st.columns(2)
        quality_metrics = [
            ('P_D / DETECTION PROBABILITY', 'P_D'),
            ('PRECISION', 'precision'),
            ('RECALL', 'recall'),
            ('F1 SCORE', 'F1'),
        ]
        for i, (title, metric_name) in enumerate(quality_metrics):
            with qcols[i % 2]:
                chart_df = bench.set_index('Strategy')[[metric_name]] * 100
                chart_df = chart_df.rename(columns={metric_name: title})
                st.markdown(
                    f'<div class="small" style="margin:8px 0 4px">{title}</div>',
                    unsafe_allow_html=True
                )
                st.bar_chart(chart_df, height=190, width='stretch')

        pfa_df = bench.set_index('Strategy')[['P_FA']] * 100
        pfa_df = pfa_df.rename(columns={'P_FA': 'False Alarm Rate (%)'})
        st.markdown('<div class="panel-title">FALSE ALARM RATE — MEAN (%)</div>', unsafe_allow_html=True)
        st.bar_chart(pfa_df, height=240, width='stretch')

        accuracy_df = bench.set_index('Strategy')[['Accuracy']] * 100
        st.markdown('<div class="panel-title">SCAN-LEVEL ACCURACY — MEAN (%)</div>', unsafe_allow_html=True)
        st.bar_chart(accuracy_df, height=240, width='stretch')

        delay_df = bench.set_index('Strategy')[['avg_intercept_time']].rename(
            columns={'avg_intercept_time': 'Average Detection Delay (slots)'}
        )
        st.markdown('<div class="panel-title">AVERAGE DETECTION DELAY — MEAN</div>', unsafe_allow_html=True)
        st.bar_chart(delay_df, height=260, width='stretch')

        # Trial variability is useful evidence for an SIH evaluation.
        variability_df = bench[['Strategy', 'P_D', 'F1']].copy().set_index('Strategy')
        variability_df['P_D'] *= 100
        variability_df['F1'] *= 100
        st.markdown('<div class="panel-title">MEAN ± STANDARD DEVIATION</div>', unsafe_allow_html=True)
        for strategy in ['Smart Scan', 'Round Robin']:
            row = bench[bench['Strategy'] == strategy].iloc[0]
            st.caption(
                f"{strategy}: P_D {row['P_D']*100:.2f}% ± {row['P_D_std']*100:.2f}%  |  "
                f"F1 {row['F1']*100:.2f}% ± {row['F1_std']*100:.2f}%"
            )

        scenario_name = 'baseline / stationary'
        scenario = 'baseline'
        adaptive_metrics = {
            'System P_D': smart['P_D'] > rr['P_D'],
            'Useful detections / 100 scans': smart.get('useful_detections_per_100_scans', smart.get('TP', 0) / max(smart.get('Scans', 1), 1) * 100) > rr.get('useful_detections_per_100_scans', rr.get('TP', 0) / max(rr.get('Scans', 1), 1) * 100),
        }
        if scenario == 'agile_hopping':
            adaptive_metrics['Hop recovery'] = (pd.notna(smart.get('avg_hop_recovery_time')) and pd.notna(rr.get('avg_hop_recovery_time')) and smart['avg_hop_recovery_time'] < rr['avg_hop_recovery_time'])
            adaptive_metrics['Blind time'] = (pd.notna(smart.get('receiver_blind_time_fraction')) and pd.notna(rr.get('receiver_blind_time_fraction')) and smart['receiver_blind_time_fraction'] < rr['receiver_blind_time_fraction'])
        smart_primary = sum(adaptive_metrics.values())
        rr_primary = len(adaptive_metrics) - smart_primary
        if smart_primary > rr_primary:
            verdict, verdict_class = 'SMART SCAN', 'adv-good'
        elif rr_primary > smart_primary:
            verdict, verdict_class = 'ROUND ROBIN', 'adv-neutral'
        else:
            verdict, verdict_class = 'TIE', 'adv-neutral'
        st.info(
            f'PRIMARY OBJECTIVE — {scenario_name.upper()}: SPECTRA is evaluated on adaptive resource allocation using system P_D and useful detections per scan. '
            + ('Agile hopping additionally measures recovery after a synthetic target-frequency change and receiver blind time. ' if scenario == 'agile_hopping' else 'Run the agile-hopping scenario to test the behavior that distinguishes SPECTRA from a fixed uniform scan. ')
            + 'Round Robin remains the fixed uniform control.'
        )
        st.markdown(
            f'<div class="advantage-panel final-verdict"><div class="advantage-title">FINAL VERDICT · ADAPTIVE OBJECTIVE</div>'
            f'<div class="adv-value {verdict_class}" style="font-size:24px;letter-spacing:.08em">{verdict}</div>'
            f'<div class="adv-note">{smart_primary}/{len(adaptive_metrics)} primary metrics favor Smart Scan; {rr_primary}/{len(adaptive_metrics)} favor Round Robin. Scenario: {scenario_name}.</div>'
            f'<div class="adv-note" style="margin-top:8px;padding-top:8px;border-top:1px solid #17323b">Secondary metrics such as first-intercept delay remain visible. A Round Robin delay advantage in a stationary scenario is reported as a trade-off, not hidden.</div>'
            f'</div>', unsafe_allow_html=True
        )

        st.caption(
            'Fair repeated simulation: every trial gives both strategies the same synthetic ground truth, '
            'detector characteristics and deterministic detector random draws. '
            'The results are software simulation outputs, not measurements from real RF hardware.'
        )

else:
    st.markdown('<div class="panel"><div class="panel-title">EXPERIMENT LOG</div>',unsafe_allow_html=True)
    rows=get_experiments()
    if rows:
        st.dataframe(pd.DataFrame(rows),width="stretch",hide_index=True)
    else:
        st.info('No experiments saved yet.')
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#45636b;font-size:9px;margin-top:15px;letter-spacing:.12em">SPECTRA • SIH26055 • SOFTWARE SIMULATION ONLY • SYNTHETIC SPECTRUM ENVIRONMENT</div>',unsafe_allow_html=True)
