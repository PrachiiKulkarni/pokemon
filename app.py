"""
Pokemon Battle Predictor - Flask app for the trained XGBoost model.

Folder layout (same as the notebook):
    app.py
    xgboost_model.json
    Data/Pokemon.csv

Run:
    pip install flask pandas numpy scikit-learn xgboost
    python app.py
    open http://127.0.0.1:5000
"""
import os
import threading
import webbrowser

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template_string, request
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # running inside a Jupyter notebook: use the notebook's folder
    BASE = os.getcwd()
MODEL_PATH = os.path.join(BASE, "xgboost_model.json")


def find_data():
    """Use POKEMON_CSV if set, otherwise the first of these that exists."""
    if os.environ.get("POKEMON_CSV"):
        return os.environ["POKEMON_CSV"]
    for cand in (os.path.join(BASE, "Data", "Pokemon.csv"), os.path.join(BASE, "Data", "pokemon.csv"),
                 os.path.join(BASE, "Pokemon.csv"), os.path.join(BASE, "pokemon.csv")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("Put your Pokemon CSV at Data/Pokemon.csv or next to app.py (or set POKEMON_CSV)")


DATA_PATH = find_data()

# ---------------------------------------------------------------------------
# Feature engineering - identical to main.ipynb
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

stat_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
if 'Total' not in df.columns:  # the model uses Total; compute it if the CSV doesn't have it
    df.insert(df.columns.get_loc('HP'), 'Total', df[stat_columns].sum(axis=1))
scaler = StandardScaler()
stats_scaled = scaler.fit_transform(df[stat_columns])

pca = PCA(n_components=4)
pca_features = pca.fit_transform(stats_scaled)
df['PC1_Power'] = pca_features[:, 0]
df['PC2_Bulk_vs_Speed'] = pca_features[:, 1]
df['PC3_Phys_vs_Spec'] = pca_features[:, 2]
df['PC4_HP_Sponges'] = pca_features[:, 3]

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Combat_Role'] = kmeans.fit_predict(pca_features)

df['Dataset_ID'] = df.index + 1

type_chart = {
    'Normal':   {'Rock': 0.5, 'Ghost': 0.0, 'Steel': 0.5},
    'Fire':     {'Fire': 0.5, 'Water': 0.5, 'Grass': 2.0, 'Ice': 2.0, 'Bug': 2.0, 'Rock': 0.5, 'Dragon': 0.5, 'Steel': 2.0},
    'Water':    {'Fire': 2.0, 'Water': 0.5, 'Grass': 0.5, 'Ground': 2.0, 'Rock': 2.0, 'Dragon': 0.5},
    'Electric': {'Water': 2.0, 'Electric': 0.5, 'Grass': 0.5, 'Ground': 0.0, 'Flying': 2.0, 'Dragon': 0.5},
    'Grass':    {'Fire': 0.5, 'Water': 0.5, 'Grass': 0.5, 'Poison': 0.5, 'Ground': 2.0, 'Flying': 0.5, 'Bug': 0.5, 'Rock': 2.0, 'Dragon': 0.5, 'Steel': 0.5},
    'Ice':      {'Fire': 0.5, 'Water': 0.5, 'Grass': 2.0, 'Ice': 0.5, 'Ground': 2.0, 'Flying': 2.0, 'Dragon': 2.0, 'Steel': 0.5},
    'Fighting': {'Normal': 2.0, 'Ice': 2.0, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2.0, 'Ghost': 0.0, 'Dark': 2.0, 'Steel': 2.0, 'Fairy': 0.5},
    'Poison':   {'Grass': 2.0, 'Poison': 0.5, 'Ground': 0.5, 'Rock': 0.5, 'Ghost': 0.5, 'Steel': 0.0, 'Fairy': 2.0},
    'Ground':   {'Fire': 2.0, 'Electric': 2.0, 'Grass': 0.5, 'Poison': 2.0, 'Flying': 0.0, 'Bug': 0.5, 'Rock': 2.0, 'Steel': 2.0},
    'Flying':   {'Electric': 0.5, 'Grass': 2.0, 'Fighting': 2.0, 'Bug': 2.0, 'Rock': 0.5, 'Steel': 0.5},
    'Psychic':  {'Fighting': 2.0, 'Poison': 2.0, 'Psychic': 0.5, 'Dark': 0.0, 'Steel': 0.5},
    'Bug':      {'Fire': 0.5, 'Grass': 2.0, 'Fighting': 0.5, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 2.0, 'Ghost': 0.5, 'Dark': 2.0, 'Steel': 0.5, 'Fairy': 0.5},
    'Rock':     {'Fire': 2.0, 'Ice': 2.0, 'Fighting': 0.5, 'Ground': 0.5, 'Flying': 2.0, 'Bug': 2.0, 'Steel': 0.5},
    'Ghost':    {'Normal': 0.0, 'Psychic': 2.0, 'Ghost': 2.0, 'Dark': 0.5},
    'Dragon':   {'Dragon': 2.0, 'Steel': 0.5, 'Fairy': 0.0},
    'Dark':     {'Fighting': 0.5, 'Psychic': 2.0, 'Ghost': 2.0, 'Dark': 0.5, 'Fairy': 0.5},
    'Steel':    {'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Ice': 2.0, 'Rock': 2.0, 'Steel': 0.5, 'Fairy': 2.0},
    'Fairy':    {'Fire': 0.5, 'Fighting': 2.0, 'Poison': 0.5, 'Dragon': 2.0, 'Dark': 2.0, 'Steel': 0.5},
}


def get_best_mult(atk_t1, atk_t2, def_t1, def_t2):
    def calc_mult(attack_type):
        if pd.isna(attack_type):
            return 0.0
        mult = type_chart.get(attack_type, {}).get(def_t1, 1.0)
        if pd.notna(def_t2):
            mult *= type_chart.get(attack_type, {}).get(def_t2, 1.0)
        return mult
    return max(calc_mult(atk_t1), calc_mult(atk_t2))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(MODEL_PATH)
FEATURES = xgb_model.get_booster().feature_names


def predict_battle(p1_name, p2_name):
    """Same logic as predict_battle() in the notebook; returns (winner, confidence)."""
    p1 = df[df['Name'] == p1_name].copy()
    p2 = df[df['Name'] == p2_name].copy()

    # Faster Pokemon becomes Player 1 (speed tie: first pick attacks first)
    if p2['Speed'].values[0] > p1['Speed'].values[0]:
        p1, p2 = p2, p1
        p1_name, p2_name = p2_name, p1_name

    p1 = p1.iloc[[0]].add_suffix('_P1').reset_index(drop=True)
    p2 = p2.iloc[[0]].add_suffix('_P2').reset_index(drop=True)
    sim_df = pd.concat([p1, p2], axis=1)

    sim_df['Speed_Diff'] = sim_df['Speed_P1'] - sim_df['Speed_P2']
    sim_df['Physical_Advantage_P1'] = sim_df['Attack_P1'] - sim_df['Defense_P2']
    sim_df['Special_Advantage_P1'] = sim_df['Sp. Atk_P1'] - sim_df['Sp. Def_P2']
    sim_df['Physical_Advantage_P2'] = sim_df['Attack_P2'] - sim_df['Defense_P1']
    sim_df['Special_Advantage_P2'] = sim_df['Sp. Atk_P2'] - sim_df['Sp. Def_P1']
    sim_df['P1_Best_Multiplier'] = sim_df.apply(
        lambda r: get_best_mult(r['Type 1_P1'], r['Type 2_P1'], r['Type 1_P2'], r['Type 2_P2']), axis=1)
    sim_df['P2_Best_Multiplier'] = sim_df.apply(
        lambda r: get_best_mult(r['Type 1_P2'], r['Type 2_P2'], r['Type 1_P1'], r['Type 2_P1']), axis=1)

    X_sim = sim_df[FEATURES]
    prob = xgb_model.predict_proba(X_sim)[0]
    win_prob = {p1_name: float(prob[1]), p2_name: float(prob[0])}  # P(win) for each Pokemon
    if prob[1] >= 0.5:
        return p1_name, float(prob[1]), win_prob
    return p2_name, float(prob[0]), win_prob


def verdict(confidence):
    if confidence >= 0.80:
        return "Decisive win"
    if confidence >= 0.65:
        return "Clear win"
    return "Close call"


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = Flask(__name__)
NAMES = set(df['Name'])

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Battle Predictor</title>
<style>
  :root { --bg:#f5f4f0; --card:#fff; --ink:#1d1d1b; --muted:#6b6a66; --line:#e2e0da; --accent:#d64533; --hl:#f0eee8; --p1:#2f6fd6; --p2:#d64533; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#141413; --card:#1e1e1c; --ink:#eceae4; --muted:#9a988f; --line:#2f2e2b; --accent:#e8604f; --hl:#2a2926; --p1:#5b92ee; --p2:#e8604f; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width:760px; margin:0 auto; padding:32px 16px; }
  h1 { margin:0 0 4px; font-size:28px; } .sub { margin:0 0 20px; color:var(--muted); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; }
  .pick { display:grid; grid-template-columns:1fr auto 1fr; gap:14px; align-items:end; }
  .field label { display:block; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:4px; }
  .combo { position:relative; }
  .combo input { width:100%; padding:10px 12px; font:inherit; font-weight:600; color:var(--ink); background:var(--bg);
                 border:1px solid var(--line); border-radius:8px; }
  .combo input:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  .combo ul { position:absolute; z-index:10; left:0; right:0; top:calc(100% + 4px); margin:0; padding:4px; list-style:none;
              max-height:260px; overflow:auto; background:var(--card); border:1px solid var(--line); border-radius:8px;
              box-shadow:0 8px 24px rgba(0,0,0,.12); }
  .combo li { padding:7px 10px; border-radius:6px; cursor:pointer; display:flex; justify-content:space-between; gap:8px; }
  .combo li.active { background:var(--hl); }
  .combo li span { color:var(--muted); font-size:12.5px; white-space:nowrap; }
  .combo li.none { color:var(--muted); cursor:default; }
  .vs { font-weight:800; color:var(--muted); letter-spacing:.1em; padding-bottom:10px; }
  button { display:block; margin:18px auto 0; font:inherit; font-weight:700; font-size:16px; color:#fff;
           background:var(--accent); border:0; border-radius:10px; padding:12px 30px; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .results { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }
  .label { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
  .value { font-size:26px; font-weight:700; }
  .value small { display:block; font-size:14px; font-weight:500; color:var(--muted); }
  .err { color:var(--accent); }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; vertical-align:1px; }
  .dot.b { background:var(--p1); } .dot.r { background:var(--p2); }
  .section { margin-top:16px; }
  .section h3 { margin:0 0 12px; font-size:15px; }
  .split { display:flex; height:34px; border-radius:8px; overflow:hidden; background:var(--hl); }
  .split div { display:flex; align-items:center; min-width:0; transition:width .7s ease; }
  .split span { padding:0 10px; font-size:13px; font-weight:650; color:#fff; white-space:nowrap; overflow:hidden; }
  .split.empty div { background:var(--line); }
  .hint { margin:0; color:var(--muted); }
  .split .a { background:var(--p1); } .split .b { background:var(--p2); justify-content:flex-end; }
  .legend { display:flex; justify-content:space-between; margin-top:8px; font-size:13px; color:var(--muted); }
  .statrow { display:grid; grid-template-columns:1fr 72px 1fr; align-items:center; gap:8px; margin:7px 0; }
  .statrow .lbl { text-align:center; font-size:12.5px; color:var(--muted); }
  .bar { position:relative; height:18px; background:var(--hl); border-radius:4px; overflow:hidden; }
  .bar i { position:absolute; top:0; bottom:0; transition:width .5s; }
  .bar.l i { right:0; background:var(--p1); } .bar.r i { left:0; background:var(--p2); }
  .bar:not(.lead) i { opacity:.45; }
  .bar b { position:absolute; top:0; bottom:0; display:flex; align-items:center; padding:0 6px; font-size:11.5px; }
  .bar.l b { left:0; } .bar.r b { right:0; }
  .stat-head { display:grid; grid-template-columns:1fr 72px 1fr; font-weight:650; margin-bottom:4px; }
  .stat-head div:last-child { text-align:right; }
  @media (max-width:600px) { .pick, .results { grid-template-columns:1fr; } .vs { text-align:center; padding:0; } }
</style></head>
<body><main>
  <h1>Battle Predictor</h1>
  <p class="sub">Search and pick two Pokémon. The XGBoost model decides who wins.</p>
  <div class="card">
    <div class="pick">
      <div class="field"><label for="p1"><span class="dot b"></span>Pokémon 1</label>
        <div class="combo"><input id="p1" placeholder="Type to search…" autocomplete="off"><ul hidden></ul></div></div>
      <span class="vs">VS</span>
      <div class="field"><label for="p2"><span class="dot r"></span>Pokémon 2</label>
        <div class="combo"><input id="p2" placeholder="Type to search…" autocomplete="off"><ul hidden></ul></div></div>
    </div>
    <button id="go" disabled>Battle!</button>
  </div>
  <div class="results">
    <div class="card"><div class="label">Who won</div><div class="value" id="winner">—</div></div>
    <div class="card"><div class="label">Model verdict</div><div class="value" id="verdict">—</div></div>
  </div>
  <div class="card section" id="probCard">
    <h3>Win probability</h3>
    <div class="split" id="split"><div class="a" id="barA" style="width:50%"><span></span></div><div class="b" id="barB" style="width:50%"><span></span></div></div>
    <div class="legend"><span><span class="dot b"></span><span id="legA"></span></span><span><span id="legB"></span><span class="dot r" style="margin:0 0 0 6px"></span></span></div>
  </div>
  <div class="card section" id="statCard">
    <h3>Stats</h3>
    <div class="stat-head"><div id="sA"></div><div></div><div id="sB"></div></div>
    <div id="stats"><p class="hint">Pick two Pokémon to compare their stats.</p></div>
  </div>
  <p class="err" id="err"></p>
</main>
<script>
const $ = (id) => document.getElementById(id);
let POKEMON = [];
const picked = { p1: null, p2: null };

function esc(s) { return s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

function setupCombo(id) {
  const input = $(id), list = input.nextElementSibling;
  let items = [], active = 0;

  function render() {
    const q = input.value.trim().toLowerCase();
    items = POKEMON.filter(p => p.name.toLowerCase().includes(q))
                   .sort((a, b) => a.name.toLowerCase().indexOf(q) - b.name.toLowerCase().indexOf(q))
                   .slice(0, 100);
    active = 0;
    list.innerHTML = items.length
      ? items.map((p, i) => `<li data-i="${i}" class="${i === 0 ? "active" : ""}">${esc(p.name)}<span>${p.types.join(" / ")}</span></li>`).join("")
      : `<li class="none">No match</li>`;
    list.hidden = false;
  }
  function choose(p) {
    picked[id] = p.name; input.value = p.name; list.hidden = true; update();
  }
  function highlight() {
    [...list.children].forEach((li, i) => li.classList.toggle("active", i === active));
    list.children[active]?.scrollIntoView({ block: "nearest" });
  }
  input.addEventListener("focus", () => { input.select(); render(); });
  input.addEventListener("input", () => { picked[id] = null; update(); render(); });
  input.addEventListener("keydown", (e) => {
    if (list.hidden) return;
    if (e.key === "ArrowDown") { active = Math.min(active + 1, items.length - 1); highlight(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); highlight(); e.preventDefault(); }
    else if (e.key === "Enter" && items[active]) { choose(items[active]); e.preventDefault(); }
    else if (e.key === "Escape") { list.hidden = true; }
  });
  list.addEventListener("mousedown", (e) => {
    const li = e.target.closest("li[data-i]"); if (li) { choose(items[+li.dataset.i]); e.preventDefault(); }
  });
  input.addEventListener("blur", () => {
    list.hidden = true;
    const exact = POKEMON.find(p => p.name.toLowerCase() === input.value.trim().toLowerCase());
    if (exact) choose(exact);
  });
}

const STATS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Total"];

function update() {
  $("go").disabled = !(picked.p1 && picked.p2);
  $("winner").textContent = "—"; $("verdict").textContent = "—"; $("err").textContent = "";
  resetProb();
  renderStats();
}

function renderStats() {
  const a = POKEMON.find(p => p.name === picked.p1), b = POKEMON.find(p => p.name === picked.p2);
  $("sA").textContent = a ? a.name : ""; $("sB").textContent = b ? b.name : "";
  if (!a || !b) { $("stats").innerHTML = '<p class="hint">Pick two Pokémon to compare their stats.</p>'; return; }
  const max = {};
  STATS.forEach(k => max[k] = Math.max(...POKEMON.map(p => p.stats[k])));  // scale to the dataset's max
  $("stats").innerHTML = STATS.map(k => {
    const va = a.stats[k], vb = b.stats[k];
    return `<div class="statrow">
      <div class="bar l ${va >= vb ? "lead" : ""}"><i style="width:${va / max[k] * 100}%"></i><b>${va}</b></div>
      <div class="lbl">${k}</div>
      <div class="bar r ${vb >= va ? "lead" : ""}"><i style="width:${vb / max[k] * 100}%"></i><b>${vb}</b></div>
    </div>`;
  }).join("");
}

function resetProb() {
  $("split").classList.add("empty");
  $("barA").style.width = "50%"; $("barB").style.width = "50%";
  $("barA").firstChild.textContent = ""; $("barB").firstChild.textContent = "";
  $("legA").textContent = picked.p1 || "Pokémon 1";
  $("legB").textContent = (picked.p2 || "Pokémon 2") + " · press Battle!";
}

function showProb(j) {
  const a = j.p1_win_prob * 100, b = j.p2_win_prob * 100;
  $("split").classList.remove("empty");
  const A = $("barA"), B = $("barB");
  A.style.width = "50%"; B.style.width = "50%";
  requestAnimationFrame(() => requestAnimationFrame(() => { A.style.width = a + "%"; B.style.width = b + "%"; }));
  A.firstChild.textContent = a >= 10 ? a.toFixed(1) + "%" : "";
  B.firstChild.textContent = b >= 10 ? b.toFixed(1) + "%" : "";
  $("legA").textContent = `${picked.p1} ${a.toFixed(1)}%`;
  $("legB").textContent = `${picked.p2} ${b.toFixed(1)}%`;
}

$("go").onclick = async () => {
  if (picked.p1 === picked.p2) { $("err").textContent = "Pick two different Pokémon."; return; }
  $("err").textContent = ""; $("winner").textContent = "…"; $("verdict").textContent = "…";
  $("go").disabled = true;
  try {
    const r = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ p1: picked.p1, p2: picked.p2 }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    $("winner").textContent = j.winner;
    $("verdict").innerHTML = j.verdict + "<small>" + (j.confidence * 100).toFixed(1) + "% confidence</small>";
    showProb(j);
  } catch (e) {
    $("winner").textContent = "—"; $("verdict").textContent = "—"; $("err").textContent = e.message;
  } finally { $("go").disabled = false; }
};

fetch("/pokemon").then(r => r.json()).then(list => {
  POKEMON = list;
  setupCombo("p1"); setupCombo("p2"); resetProb();
  $("p1").placeholder = $("p2").placeholder = `Search ${list.length} Pokémon…`;
});
</script>
</body></html>"""


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE)


@app.route("/pokemon", methods=["GET"])
def pokemon_list():
    """All Pokemon in the CSV, for the searchable dropdowns."""
    rows = df.drop_duplicates('Name')
    return jsonify([
        {"name": r['Name'], "types": [t for t in (r['Type 1'], r['Type 2']) if pd.notna(t)],
         "stats": {c: int(r[c]) for c in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']}}
        for _, r in rows.sort_values('Name').iterrows()
    ])


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True) or {}
    p1, p2 = body.get("p1"), body.get("p2")
    if p1 not in NAMES or p2 not in NAMES:
        return jsonify(error="Could not find one or both Pokémon in the dataset."), 400
    if p1 == p2:
        return jsonify(error="Pick two different Pokémon."), 400
    winner, confidence, win_prob = predict_battle(p1, p2)
    return jsonify(winner=winner, verdict=verdict(confidence), confidence=round(confidence, 4),
                   p1_win_prob=round(win_prob[p1], 4), p2_win_prob=round(win_prob[p2], 4))


# ---------------------------------------------------------------------------
# Start the server (works both in Jupyter and as `python app.py`)
# ---------------------------------------------------------------------------
_server = globals().get("_server")  # survives re-running the notebook cell


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def run(port=5000, open_browser=True):
    global _server
    url = "http://127.0.0.1:%d" % port
    if in_notebook():
        # No reloader in Jupyter (that is what caused SystemExit: 1). Run in a background
        # thread so the notebook stays usable, and stop any server from a previous run first.
        from werkzeug.serving import make_server
        if _server is not None:
            _server.shutdown()
            _server.server_close()
        _server = make_server("127.0.0.1", port, app, threaded=True)
        threading.Thread(target=_server.serve_forever, daemon=True).start()
        print("Battle Predictor (with stats + win-probability bar) running at", url)
        if open_browser:
            webbrowser.open(url)
        from IPython.display import IFrame, display
        display(IFrame(url, width="100%", height=1150))
    elif os.environ.get("PORT"):  # on a cloud host: listen on the port it gives us
        app.run(host="0.0.0.0", port=int(os.environ["PORT"]), debug=False, use_reloader=False)
    else:
        if open_browser:
            threading.Timer(1.0, webbrowser.open, [url]).start()
        app.run(port=port, debug=False, use_reloader=False)


def stop():
    """In Jupyter: stop the server started by run()."""
    global _server
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _server = None
        print("Server stopped")


if __name__ == "__main__":
    run()
