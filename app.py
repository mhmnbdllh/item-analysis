import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, norm
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import io
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# ACADEMIC ITEM ANALYSIS — CTT + IRT SUITE (2026)
# CTT + IRT 1PL/2PL/3PL + Auto-Interpretation
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro by Dr. Muhaimin Abdullah, S.Pd., M.Pd.", page_icon="📊", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background-color: #0d1117; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #58a6ff !important;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }
.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.25rem 0;
}
.flag-retain { background:#1a4731; color:#3fb950; border:1px solid #3fb950; border-radius:4px; padding:2px 8px; font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; }
.flag-revise { background:#3d2b00; color:#d29922; border:1px solid #d29922; border-radius:4px; padding:2px 8px; font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; }
.flag-reject { background:#3d1212; color:#f85149; border:1px solid #f85149; border-radius:4px; padding:2px 8px; font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; }
div[data-testid="stExpander"] { border: 1px solid #30363d; border-radius:6px; }
.stAlert { border-radius:6px !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: #161b22; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: #8b949e; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
.stTabs [aria-selected="true"] { background:#21262d !important; color:#58a6ff !important; }
a[href*="github.com"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
.stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────────────────
st.markdown("# ITEM ANALYSIS by Dr. Muhaimin Abdullah, S.Pd., M.Pd.")
st.markdown("**Classical Test Theory (CTT) + Item Response Theory (IRT 1PL / 2PL / 3PL)** · *Methodologically Validated · 2026 Edition*")
st.divider()

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.markdown("**CTT Parameters**")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27,
        help="Percentage of top/bottom students for discrimination index d. Kelley (1939) recommends 27%.")
    validity_limit = st.number_input("r_pbis Validity Threshold", 0.0, 1.0, 0.25, step=0.05,
        help="Minimum corrected point-biserial correlation. Common threshold: 0.20–0.30 (Ebel & Frisbie, 1991).")

    st.markdown("---")
    st.markdown("**IRT Parameters**")
    irt_model = st.selectbox("IRT Model", ["1PL (Rasch)", "2PL", "3PL"],
        help="1PL: difficulty only. 2PL: difficulty + discrimination. 3PL: difficulty + discrimination + pseudo-guessing.")
    irt_max_iter = st.slider("Max EM Iterations", 50, 500, 200, step=50)

    st.markdown("---")
    st.markdown("### 📖 Legend")
    with st.expander("Difficulty Index (p)", expanded=False):
        st.markdown("""
| Range | Label | Symbol |
|---|---|---|
| > 0.70 | Easy | 🟢 |
| 0.30–0.70 | Moderate | 🟡 |
| < 0.30 | Difficult | 🔴 |

*Optimal range: 0.30–0.70 — maximises score variance.*
*p = (number correct) / (total students)*
        """)
    with st.expander("Discrimination (d & r_pbis)", expanded=False):
        st.markdown("""
**Discrimination Index (d):**

| d-value | Label |
|---|---|
| ≥ 0.40 | 🟢 Excellent |
| 0.30–0.39 | 🔵 Good |
| 0.20–0.29 | 🟡 Fair |
| < 0.20 | 🔴 Poor |

*d = p(Upper) − p(Lower)*
*Upper/Lower = top & bottom % by total score (Kelley)*

**r_pbis:** Corrected item-total correlation.
Minimum ≥ 0.20–0.30 for validity.

**DDI:** prop(Lower selecting option) − prop(Upper selecting option).
Positive DDI = distractor functioning correctly.
        """)
    with st.expander("IRT Parameters", expanded=False):
        st.markdown("""
**b (difficulty):** θ where P(correct) = 0.50 (1PL/2PL) or (1+c)/2 (3PL).
**a (discrimination):** ICC slope. Higher = better discriminating.
**c (pseudo-guessing):** Lower asymptote — P(correct) for very low-ability students.
**INFIT MNSQ:** Variance-weighted fit. Ideal: 0.70–1.30.
**OUTFIT MNSQ:** Unweighted fit (outlier-sensitive). Ideal: 0.70–1.30.
        """)
    with st.expander("Decision Criteria", expanded=False):
        st.markdown("""
**RETAIN:** p 0.20–0.90 AND d ≥ 0.30 AND r_pbis ≥ threshold AND all DDI ≥ 0
**REJECT:** p < 0.05 or > 0.95, OR d < 0, OR (r_pbis < threshold AND d < 0.20), OR DDI < −0.10
**REVISE:** Borderline — at least one criterion not met, no severe failure

*Ebel & Frisbie (1991); Crocker & Algina (1986)*
        """)
    with st.expander("Reliability Benchmarks", expanded=False):
        st.markdown("""
| KR-20 | Interpretation |
|---|---|
| ≥ 0.90 | Excellent (high-stakes) |
| 0.80–0.89 | High |
| 0.70–0.79 | Acceptable |
| < 0.70 | Low — revision needed |

*KR-20: Kuder & Richardson (1937) — for dichotomous items only.*
        """)

# ══════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════
st.markdown("### 📁 Data Input")
student_file = st.file_uploader("Upload Data CSV", type=['csv'],
    help="Row 1 = Answer Key (starts with 'ANSWER'). Row 2 = Header (StudentID, Q1, Q2, ...). Row 3 onwards = Student responses.")

st.markdown("""
<details><summary>📋 <b>CSV Format Guide</b></summary>

**Format CSV (1 file):**
```
ANSWER,A,C,A,D,B,...
StudentID,Q1,Q2,Q3,Q4,Q5,...
S001,A,C,B,D,A,...
S002,B,C,A,A,B,...
S003,A,A,B,C,D,...
```
*Baris 1 = kunci jawaban (kolom pertama diisi ANSWER). Baris 2 = header. Baris 3 dst = jawaban siswa.*
</details>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# IRT CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def irt_prob(theta, a, b, c):
    """3PL ICC: P(theta) = c + (1-c) / (1 + exp(-a*(theta-b)))"""
    return c + (1 - c) * expit(a * (theta - b))

def estimate_irt_em(X, model='2PL', max_iter=200, tol=1e-5):
    """
    EM Algorithm for IRT parameter estimation.
    Uses Gauss-Hermite quadrature with normal prior on theta.
    """
    n, k = X.shape
    n_quad  = 21
    theta_q = np.linspace(-4, 4, n_quad)
    w_q     = norm.pdf(theta_q, 0, 1)
    w_q    /= w_q.sum()

    p_mean  = X.mean(axis=0).clip(0.05, 0.95)
    params  = {
        'a': np.ones(k),
        'b': -norm.ppf(p_mean),
        'c': np.zeros(k)
    }
    prev_ll = -np.inf

    for _ in range(max_iter):
        a_arr, b_arr, c_arr = params['a'], params['b'], params['c']

        P_qj = np.array([[irt_prob(th, a_arr[j], b_arr[j], c_arr[j])
                          for j in range(k)] for th in theta_q])

        L_qi = np.ones((n_quad, n))
        for q in range(n_quad):
            p_q    = P_qj[q]
            L_qi[q] = np.prod(np.where(X == 1, p_q, 1 - p_q), axis=1)

        f_qi = L_qi * w_q[:, None]
        marg = np.maximum(f_qi.sum(axis=0), 1e-300)
        post = f_qi / marg[None, :]

        r_q  = post.sum(axis=1)
        rj_q = post @ X
        log_lik = np.sum(np.log(marg))

        for j in range(k):
            def neg_loglik_item(pars, j=j):
                if model == '1PL':
                    a_j, b_j, c_j = 1.0, pars[0], 0.0
                elif model == '2PL':
                    a_j, b_j, c_j = pars[0], pars[1], 0.0
                else:
                    a_j, b_j, c_j = pars[0], pars[1], pars[2]
                p_j = np.clip(
                    np.array([irt_prob(th, a_j, b_j, c_j) for th in theta_q]),
                    1e-9, 1 - 1e-9)
                return -np.sum(rj_q[:, j] * np.log(p_j) +
                               (r_q - rj_q[:, j]) * np.log(1 - p_j))

            if model == '1PL':
                x0, bounds = [b_arr[j]], [(-4, 4)]
            elif model == '2PL':
                x0, bounds = [a_arr[j], b_arr[j]], [(0.3, 3.0), (-4, 4)]
            else:
                x0     = [a_arr[j], b_arr[j], c_arr[j]]
                bounds = [(0.3, 3.0), (-4, 4), (0.0, 0.40)]
            try:
                res = minimize(neg_loglik_item, x0, bounds=bounds,
                               method='L-BFGS-B', options={'maxiter':50,'ftol':1e-8})
                if model == '1PL':
                    params['b'][j] = res.x[0]
                elif model == '2PL':
                    params['a'][j], params['b'][j] = res.x
                else:
                    params['a'][j], params['b'][j], params['c'][j] = res.x
            except Exception:
                pass

        if abs(log_lik - prev_ll) < tol:
            break
        prev_ll = log_lik

    thetas = []
    for i in range(n):
        def neg_ll_theta(th, i=i):
            p_j = np.clip(
                np.array([irt_prob(th[0], params['a'][j], params['b'][j], params['c'][j])
                          for j in range(k)]),
                1e-9, 1 - 1e-9)
            return -np.sum(X[i]*np.log(p_j) + (1-X[i])*np.log(1-p_j))
        try:
            r = minimize(neg_ll_theta, [0.0], bounds=[(-4,4)], method='L-BFGS-B')
            thetas.append(float(r.x[0]))
        except Exception:
            thetas.append(0.0)

    return params, np.array(thetas), log_lik


def compute_item_info(theta_range, a, b, c):
    """
    Item Information Function.
    I(theta) = [dP/dtheta]^2 / [P*Q]
    For 3PL: dP/dtheta = a*(P-c)/(1-c)*Q
    """
    P     = irt_prob(theta_range, a, b, c)
    Q     = 1 - P
    denom = np.where(np.abs(1-c) > 1e-10, 1-c, 1e-10)
    dP    = a * (P - c) / denom * Q
    return np.where(P*Q > 1e-10, dP**2 / (P*Q), 0.0)


def rasch_fit_stats(X, b_arr, theta_arr):
    """
    INFIT and OUTFIT MNSQ — Wright & Masters (1982).
    OUTFIT = mean of squared standardised residuals (unweighted).
    INFIT  = variance-weighted mean of squared standardised residuals.
    Ideal range: 0.70–1.30.
    """
    n, k = X.shape
    infit_list, outfit_list = [], []
    for j in range(k):
        P_ij  = expit(theta_arr - b_arr[j])
        W_ij  = P_ij * (1 - P_ij)
        Z_ij  = (X[:, j] - P_ij) / np.sqrt(np.maximum(W_ij, 1e-10))
        Z2_ij = Z_ij ** 2
        outfit_list.append(Z2_ij.mean())
        infit_list.append(
            np.sum(W_ij * Z2_ij) / np.maximum(W_ij.sum(), 1e-10))
    return np.array(infit_list), np.array(outfit_list)


# ══════════════════════════════════════════════════════════════════════
# INTERPRETATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def interpret_reliability(kr20, sem, n_items):
    if kr20 >= 0.90:
        rel_label  = "Excellent"
        rel_detail = (
            "The instrument demonstrates **excellent internal consistency** (KR-20 ≥ 0.90). "
            "Items cohesively measure the same latent construct. "
            "Appropriate for **high-stakes decisions** such as selection or certification.")
    elif kr20 >= 0.80:
        rel_label  = "High"
        rel_detail = (
            "The instrument shows **high reliability** (KR-20 0.80–0.89). "
            "Suitable for most formal assessments. "
            "A small number of items may benefit from revision to reach the excellent range.")
    elif kr20 >= 0.70:
        rel_label  = "Acceptable"
        rel_detail = (
            "Reliability is **acceptable** (KR-20 0.70–0.79) — minimum standard for classroom assessment. "
            "For high-stakes use, further item refinement is strongly recommended. "
            "Focus revision on items with low r_pbis or poor discrimination.")
    else:
        rel_label  = "Low"
        rel_detail = (
            "Reliability is **low** (KR-20 < 0.70) — substantial measurement error present. "
            "This instrument **should not be used for individual-level decisions** without major revision. "
            "Remove rejected items and consider increasing item count.")

    sem_interp = ("Acceptably small margin relative to test length."
                  if sem < n_items * 0.1 else
                  "Relatively large margin — interpret individual scores with caution.")
    sem_detail = (
        f"SEM = {sem:.3f}: a student's true score lies within "
        f"**±{sem:.2f} points** (~68% CI) or **±{2*sem:.2f} points** (~95% CI) "
        f"of their observed score. {sem_interp}")
    return rel_label, rel_detail, sem_detail


def interpret_item_profile(df_res, n_items, validity_limit):
    n_retain   = (df_res['DECISION']=='RETAIN').sum()
    n_revise   = (df_res['DECISION']=='REVISE').sum()
    n_reject   = (df_res['DECISION']=='REJECT').sum()
    pct_retain = n_retain / n_items * 100
    n_easy     = (df_res['p'] > 0.70).sum()
    n_mod      = ((df_res['p'] >= 0.30) & (df_res['p'] <= 0.70)).sum()
    n_hard     = (df_res['p'] < 0.30).sum()
    n_valid    = (df_res['r_pbis'] >= validity_limit).sum()

    diff_comment = (
        "Difficulty distribution is well-balanced." if n_mod/n_items >= 0.50 else
        f"Only {n_mod}/{n_items} items in the moderate range (0.30–0.70) — "
        "consider revising extreme items to improve score variance.")
    val_comment = (
        "Strong item validity overall." if n_valid/n_items >= 0.75 else
        f"Only {n_valid}/{n_items} items meet r_pbis ≥ {validity_limit} — "
        "this substantially impacts construct validity of the total score.")

    summary = (
        f"Out of **{n_items} items**, **{n_retain} ({pct_retain:.0f}%)** RETAIN, "
        f"**{n_revise}** REVISE, **{n_reject}** REJECT.\n\n"
        f"**Difficulty:** {n_easy} easy · {n_mod} moderate · {n_hard} difficult. {diff_comment}\n\n"
        f"**Validity (r_pbis):** {n_valid}/{n_items} meet threshold ≥ {validity_limit}. {val_comment}"
    )
    return summary, n_retain, n_revise, n_reject, n_easy, n_mod, n_hard


def interpret_irt_params(a_arr, b_arr, c_arr, model):
    lines  = []
    b_mean = b_arr.mean(); b_std = b_arr.std()
    if model in ['2PL','3PL']:
        a_mean = a_arr.mean()
        a_comment = (
            "**highly discriminating** (ā ≥ 1.5)." if a_mean >= 1.5 else
            "**moderate discrimination** (0.8 ≤ ā < 1.5)." if a_mean >= 0.8 else
            "**weak discrimination** (ā < 0.8) — review item construction.")
        lines.append(f"**Mean discrimination (ā = {a_mean:.3f}):** Items are {a_comment}")
    b_comment = (
        "Items **well-centered** around mean ability (b̄ ≈ 0)." if abs(b_mean) < 0.3 else
        f"Items tend to be **{'easier' if b_mean<0 else 'harder'}** than average test-taker ability.")
    lines.append(
        f"**Mean difficulty (b̄ = {b_mean:.3f}, SD = {b_std:.3f}):** {b_comment} "
        f"Wider SD = items span a broader ability range.")
    if model == '3PL':
        c_mean    = c_arr.mean()
        c_comment = (
            "**Negligible guessing** (c̄ < 0.10)." if c_mean < 0.10 else
            "**Moderate guessing** — improve distractors or lengthen stems." if c_mean < 0.25 else
            "**High guessing** (c̄ ≥ 0.25) — items may have too few distractors or poor construction.")
        lines.append(f"**Mean pseudo-guessing (c̄ = {c_mean:.3f}):** {c_comment}")
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# MATPLOTLIB HELPERS
# ══════════════════════════════════════════════════════════════════════

def dark_fig(figsize=(10,5)):
    return plt.figure(figsize=figsize, facecolor='#0d1117')

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#e6edf3')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    if title:  ax.set_title(title,  fontsize=10, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color='#21262d', linewidth=0.5, alpha=0.7)

COLORS = {
    'blue':'#58a6ff','green':'#3fb950','red':'#f85149',
    'yellow':'#d29922','purple':'#bc8cff','cyan':'#39d353',
    'orange':'#ffa657','grey':'#8b949e'
}

# ══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS BLOCK
# ══════════════════════════════════════════════════════════════════════

if student_file:
    # ── Load Data ─────────────────────────────────────────────────────
    raw        = pd.read_csv(student_file, header=None)
    answer_key = raw.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    headers    = raw.iloc[1, :].astype(str).str.strip().tolist()
    df         = raw.iloc[2:, :].copy()
    df.columns = headers
    df         = df.reset_index(drop=True).fillna("N/A")

    item_cols   = df.columns[1:]
    id_col_name = df.columns[0]

    # ── Score Matrix ──────────────────────────────────────────────────
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (
            df[col].astype(str).str.upper().str.strip() == answer_key[i]
        ).astype(int)

    total_scores        = df_scores.sum(axis=1)
    df['Total_Score']   = total_scores
    n_students, n_items = len(df), len(item_cols)

    # ── Kelley Grouping ───────────────────────────────────────────────
    # FIX: round() for unbiased group size (avoids systematic truncation)
    n_group   = max(1, round(n_students * group_percent / 100))
    df_sorted = df.sort_values('Total_Score', ascending=False).reset_index(drop=True)
    df_sorted['Rank']  = range(1, n_students+1)
    df_sorted['Group'] = 'Middle'
    df_sorted.iloc[:n_group,  df_sorted.columns.get_loc('Group')] = 'Upper'
    df_sorted.iloc[-n_group:, df_sorted.columns.get_loc('Group')] = 'Lower'
    df_ranking = df_sorted[[id_col_name,'Total_Score','Rank','Group']].copy()
    up_idx     = df_sorted.head(n_group).index
    lo_idx     = df_sorted.tail(n_group).index

    # ══════════════════════════════════════════════════════════════════
    # CTT ITEM ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    with st.spinner("⚙️ Running CTT analysis..."):
        results = []
        for i, item in enumerate(item_cols):
            p        = df_scores[item].mean()
            q        = 1 - p
            pq       = p * q
            item_var = df_scores[item].var(ddof=0)

            # Discrimination — p_up and p_lo are proportions via .mean()
            p_up  = df_scores.loc[up_idx, item].mean()
            p_lo  = df_scores.loc[lo_idx, item].mean()
            d_val = p_up - p_lo

            # DDI — proportion-based, correctly scaled for unequal group sizes
            distractors = [
                opt.strip()
                for opt in df[item].dropna().astype(str).str.upper().unique()
                if opt.strip() not in ["","N/A",answer_key[i]]
            ]
            ddi_vals = []
            for opt in distractors:
                u_opt = (df.loc[up_idx, item].astype(str).str.upper().str.strip()==opt).mean()
                l_opt = (df.loc[lo_idx, item].astype(str).str.upper().str.strip()==opt).mean()
                ddi_vals.append(l_opt - u_opt)

            ddi_best  = max(ddi_vals) if ddi_vals else 0.0
            ddi_worst = min(ddi_vals) if ddi_vals else 0.0

            # Corrected point-biserial (item removed from total)
            corrected_total = total_scores - df_scores[item]
            if df_scores[item].var() > 0 and corrected_total.var() > 0:
                r_pb, _ = pointbiserialr(df_scores[item], corrected_total)
                r_pb    = 0.0 if np.isnan(r_pb) else float(r_pb)
            else:
                r_pb = 0.0

            # Descriptive labels
            p_desc = "Easy" if p>0.70 else "Difficult" if p<0.30 else "Moderate"
            d_desc = ("Excellent" if d_val>=0.40 else "Good" if d_val>=0.30
                      else "Fair" if d_val>=0.20 else "Poor")
            r_desc = "Valid" if r_pb>=validity_limit else "Invalid"

            # Decision logic — Ebel & Frisbie (1991); Crocker & Algina (1986)
            # RETAIN: ALL criteria met
            retain_ok = (
                (0.20 <= p <= 0.90) and
                (d_val >= 0.30) and
                (r_pb >= validity_limit) and
                (ddi_worst >= 0)
            )
            # REJECT: any severe failure
            reject_ok = (
                (p > 0.95) or (p < 0.05) or
                (d_val < 0) or
                (r_pb < validity_limit and d_val < 0.20) or
                (ddi_worst < -0.10)
            )

            if retain_ok:    decision = "RETAIN"
            elif reject_ok:  decision = "REJECT"
            else:            decision = "REVISE"

            reasons = []
            if p > 0.90:              reasons.append("Too easy (p > 0.90)")
            if p < 0.10:              reasons.append("Too difficult (p < 0.10)")
            if d_val < 0:             reasons.append("Negative discrimination (d < 0)")
            elif d_val < 0.20:        reasons.append("Poor discrimination (d < 0.20)")
            if r_pb < validity_limit: reasons.append(f"Low r_pbis (< {validity_limit})")
            if ddi_worst < -0.10:     reasons.append("Severe DDI < −0.10")
            elif ddi_worst < 0:       reasons.append("Malfunctioning distractor (DDI < 0)")
            reason_text = ", ".join(reasons) if reasons else "All criteria satisfied"

            results.append({
                "Item":item,"p":p,"p_Eval":p_desc,"q":q,"pq":pq,"Var":item_var,
                "p_Upper":p_up,"p_Lower":p_lo,"d":d_val,"d_Eval":d_desc,
                "Best_DDI":ddi_best,"Worst_DDI":ddi_worst,
                "r_pbis":r_pb,"r_Eval":r_desc,"DECISION":decision,"REASON":reason_text
            })

    df_res = pd.DataFrame(results)

    # ── CTT Reliability ───────────────────────────────────────────────
    mean_score = total_scores.mean()
    var_total  = total_scores.var(ddof=0)
    std_score  = np.sqrt(var_total)

    # KR-20 exact formula (Kuder & Richardson, 1937)
    kr20 = ((n_items/(n_items-1)) * (1 - df_res["pq"].sum()/var_total)
            if var_total > 0 else 0.0)
    # Cronbach's alpha (identical to KR-20 for binary items)
    alpha = ((n_items/(n_items-1)) * (1 - df_res["Var"].sum()/var_total)
             if var_total > 0 else 0.0)
    # SEM = SD × √(1 − KR-20)
    sem = std_score * np.sqrt(max(1-kr20, 0))

    # Split-half (Spearman-Brown prophecy formula)
    odd_scores  = df_scores.iloc[:,0::2].sum(axis=1)
    even_scores = df_scores.iloc[:,1::2].sum(axis=1)
    if odd_scores.var() > 0 and even_scores.var() > 0:
        r_half     = float(np.corrcoef(odd_scores, even_scores)[0,1])
        split_half = (2*r_half/(1+r_half)) if (1+r_half) > 0 else 0.0
    else:
        r_half = split_half = 0.0

    # ══════════════════════════════════════════════════════════════════
    # IRT ESTIMATION
    # ══════════════════════════════════════════════════════════════════
    X_binary  = df_scores[list(item_cols)].values.astype(float)
    model_key = irt_model.split()[0]

    with st.spinner(f"🔬 Fitting IRT {model_key} via EM ({irt_max_iter} iterations)..."):
        irt_params, theta_hat, log_lik = estimate_irt_em(
            X_binary, model=model_key, max_iter=irt_max_iter)

    a_arr = irt_params['a']
    b_arr = irt_params['b']
    c_arr = irt_params['c']

    item_info_at_theta = np.array([
        compute_item_info(theta_hat, a_arr[j], b_arr[j], c_arr[j])
        for j in range(n_items)
    ])
    test_info_at_theta = item_info_at_theta.sum(axis=0)
    avg_info           = test_info_at_theta.mean()
    theta_var          = theta_hat.var()

    # Marginal IRT reliability — Green (1984) approximation
    irt_rel = (avg_info*theta_var/(1+avg_info*theta_var) if avg_info > 0 else 0.0)

    if model_key == '1PL':
        infit_arr, outfit_arr = rasch_fit_stats(X_binary, b_arr, theta_hat)
    else:
        infit_arr = outfit_arr = np.full(n_items, np.nan)

    df_res['IRT_b']          = b_arr
    df_res['IRT_a']          = a_arr
    df_res['IRT_c']          = c_arr
    df_res['IRT_INFIT']      = infit_arr
    df_res['IRT_OUTFIT']     = outfit_arr
    df_res['Item_Info_Peak'] = [
        compute_item_info(np.array([b_arr[j]]), a_arr[j], b_arr[j], c_arr[j])[0]
        for j in range(n_items)
    ]

    # ══════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ══════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("## 📈 Test-Level Summary")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Students (N)",   f"{n_students:,}")
    m2.metric("Items (k)",      f"{n_items:,}")
    m3.metric("Mean Score",     f"{mean_score:.2f}",
              delta=f"/{n_items} ({mean_score/n_items:.0%})")
    m4.metric("Std. Deviation", f"{std_score:.2f}")

    rel_label, rel_detail, sem_detail = interpret_reliability(kr20, sem, n_items)
    r1,r2,r3,r4,r5 = st.columns(5)
    r1.metric("KR-20",           f"{kr20:.4f}",     delta=rel_label)
    r2.metric("Cronbach's α",    f"{alpha:.4f}")
    r3.metric("Split-Half (SB)", f"{split_half:.4f}",
              help="Spearman-Brown corrected odd/even split-half.")
    r4.metric("SEM",             f"{sem:.3f}",
              help="Standard Error of Measurement = SD × √(1 − KR-20)")
    r5.metric(f"IRT Rel. ({model_key})", f"{irt_rel:.4f}",
              help="Green (1984) marginal reliability approximation.")

    # ══════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════
    tab_ctt, tab_irt, tab_dist, tab_rank, tab_interp, tab_report = st.tabs([
        "📋 CTT Item Matrix","🔬 IRT Analysis","🎯 Distractor Analysis",
        "🏆 Student Ranking","📝 Interpretive Report","📥 Download"
    ])

    # ──────────────────────────────────────────────────────────────────
    # TAB 1 — CTT ITEM MATRIX
    # ──────────────────────────────────────────────────────────────────
    with tab_ctt:
        st.markdown("### CTT Item Statistics Matrix")
        st.caption("RETAIN: p 0.20–0.90, d ≥ 0.30, r_pbis ≥ threshold, DDI ≥ 0 · "
                   "REJECT: extreme p, d < 0, or severe DDI · REVISE: borderline")

        min_ddi = df_res["Worst_DDI"].min()
        if min_ddi < -0.10:
            st.error(f"⚠️ **Severe Distractor Alert:** Worst DDI = {min_ddi:.4f}. "
                     "Possible keying error or misleading stem — immediate review required.")
        elif min_ddi < 0:
            st.warning(f"⚠️ **Malfunctioning Distractor:** Worst DDI = {min_ddi:.4f}. "
                       "Some distractors selected more by upper-group — review wording.")
        else:
            st.success("✅ **All distractors functional.** DDI ≥ 0 for all options.")

        display_cols = ["Item","p","p_Eval","q","pq","Var",
                        "p_Upper","p_Lower","d","d_Eval",
                        "Best_DDI","Worst_DDI","r_pbis","r_Eval","DECISION","REASON"]

        def apply_item_styling(row):
            styles = [''] * len(row)
            base   = 'color:black;'
            # p: green=moderate, orange=borderline, red=extreme
            dif_color = (
                '#ccffcc' if 0.30<=row['p']<=0.70 else
                '#ffe0cc' if (0.20<=row['p']<0.30 or 0.70<row['p']<=0.90) else
                '#ffcccc')
            styles[1] = styles[2] = f'background-color:{dif_color};{base}'
            # d
            if   row['d']>=0.40: dc,tc='#2ecc71','white'
            elif row['d']>=0.30: dc,tc='#3498db','white'
            elif row['d']>=0.20: dc,tc='#f1c40f','black'
            else:                 dc,tc='#e74c3c','white'
            for idx in [8,9,10,11]:
                styles[idx] = f'background-color:{dc};color:{tc}'
            # r_pbis
            val_bg = '#ccffcc' if row['r_pbis']>=validity_limit else '#ffcccc'
            styles[12] = f'background-color:{val_bg};{base}font-weight:bold'
            styles[13] = f'background-color:{val_bg};{base}'
            # decision
            if   row['DECISION']=="RETAIN": styles[14]='background-color:#27ae60;color:white;font-weight:bold'
            elif row['DECISION']=="REVISE": styles[14]='background-color:#f39c12;color:white;font-weight:bold'
            else:                            styles[14]='background-color:#c0392b;color:white;font-weight:bold'
            return styles

        st.dataframe(
            df_res[display_cols].style
                .apply(apply_item_styling, axis=1)
                .format("{:.4f}", subset=["p","q","pq","Var","p_Upper","p_Lower",
                                          "d","Best_DDI","Worst_DDI","r_pbis"]),
            use_container_width=True, height=500
        )

        st.markdown("---")
        c1,c2 = st.columns(2)
        with c1:
            fig=dark_fig((7,4)); ax=fig.add_subplot(111)
            colors_p=[COLORS['green'] if 0.30<=p<=0.70
                      else COLORS['yellow'] if (0.20<=p<0.30 or 0.70<p<=0.90)
                      else COLORS['red'] for p in df_res['p']]
            ax.bar(df_res['Item'], df_res['p'], color=colors_p, edgecolor='#30363d', linewidth=0.5)
            for y,lbl,col in [(0.90,'Reject >0.90',COLORS['red']),(0.70,'Easy 0.70',COLORS['yellow']),
                               (0.30,'Diff 0.30',COLORS['yellow']),(0.10,'Reject <0.10',COLORS['red'])]:
                ax.axhline(y, color=col, linestyle='--', lw=1, alpha=0.6, label=lbl)
            ax.set_ylim(0,1.1)
            ax.legend(fontsize=6, facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
            ax.tick_params(axis='x', rotation=45)
            style_ax(ax,'Item Difficulty (p)','Item','Proportion Correct')
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

        with c2:
            fig=dark_fig((7,4)); ax=fig.add_subplot(111)
            colors_d=[COLORS['green'] if d>=0.40 else COLORS['blue'] if d>=0.30
                      else COLORS['yellow'] if d>=0.20 else COLORS['red'] for d in df_res['d']]
            ax.bar(df_res['Item'], df_res['d'], color=colors_d, edgecolor='#30363d', linewidth=0.5)
            for y,lbl,col in [(0.40,'Excellent',COLORS['green']),(0.30,'Good',COLORS['blue']),
                               (0.20,'Fair',COLORS['yellow']),(0,'d=0',COLORS['red'])]:
                ax.axhline(y, color=col, linestyle='--' if y>0 else '-', lw=1 if y>0 else 0.8, alpha=0.6, label=lbl)
            ax.legend(fontsize=6, facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
            ax.tick_params(axis='x', rotation=45)
            style_ax(ax,'Discrimination Index (d)','Item','d = p(Upper) − p(Lower)')
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

        c3,c4 = st.columns(2)
        with c3:
            fig=dark_fig((7,4)); ax=fig.add_subplot(111)
            colors_r=[COLORS['green'] if r>=validity_limit else COLORS['red'] for r in df_res['r_pbis']]
            ax.bar(df_res['Item'], df_res['r_pbis'], color=colors_r, edgecolor='#30363d', linewidth=0.5)
            ax.axhline(validity_limit, color=COLORS['yellow'], linestyle='--', lw=1,
                       label=f'Threshold ({validity_limit})')
            ax.axhline(0, color='#555', lw=0.8)
            ax.legend(fontsize=7, facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
            ax.tick_params(axis='x', rotation=45)
            style_ax(ax,'Corrected r_pbis','Item','r_pbis')
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

        with c4:
            fig=dark_fig((5,4)); ax=fig.add_subplot(111)
            dec_counts = df_res['DECISION'].value_counts()
            pie_colors = {'RETAIN':'#27ae60','REVISE':'#f39c12','REJECT':'#c0392b'}
            clrs       = [pie_colors.get(l,'#888') for l in dec_counts.index]
            _,texts,autotexts = ax.pie(
                dec_counts.values, labels=dec_counts.index, autopct='%1.0f%%',
                colors=clrs, textprops={'color':'#c9d1d9','fontsize':9},
                wedgeprops={'edgecolor':'#0d1117','linewidth':2})
            for at in autotexts: at.set_fontsize(9); at.set_fontweight('bold')
            ax.set_facecolor('#0d1117')
            style_ax(ax,'Decision Distribution')
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

        st.markdown("#### 📊 Score Distribution")
        fig=dark_fig((10,3.5)); ax=fig.add_subplot(111)
        ax.hist(total_scores, bins=min(n_items,20), color=COLORS['blue'],
                edgecolor='#0d1117', linewidth=0.5, alpha=0.85)
        ax.axvline(mean_score,       color=COLORS['orange'], linestyle='--', lw=1.5,
                   label=f'Mean={mean_score:.2f}')
        ax.axvline(mean_score-sem,   color=COLORS['grey'],   linestyle=':',  lw=1,
                   label=f'±SEM=±{sem:.2f}')
        ax.axvline(mean_score+sem,   color=COLORS['grey'],   linestyle=':',  lw=1)
        ax.legend(fontsize=8, facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
        style_ax(ax,'Raw Score Distribution','Total Score','Frequency')
        fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # TAB 2 — IRT ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    with tab_irt:
        st.markdown(f"### IRT Analysis — {irt_model}")
        i1,i2,i3,i4 = st.columns(4)
        i1.metric("Log-Likelihood",      f"{log_lik:.2f}")
        i2.metric("IRT Rel. (approx)",   f"{irt_rel:.4f}",
                  help="Green (1984) marginal reliability — approximation.")
        i3.metric("Mean b (Difficulty)", f"{b_arr.mean():.3f}")
        i4.metric("Mean θ (Ability)",    f"{theta_hat.mean():.3f}",
                  delta=f"SD={theta_hat.std():.3f}")

        if model_key in ['2PL','3PL']:
            ia1,ia2,ia3 = st.columns(3)
            ia1.metric("Mean a (Discrimination)", f"{a_arr.mean():.3f}")
            ia2.metric("Mean c (Pseudo-guess)",   f"{c_arr.mean():.3f}")
            n_par = {'1PL':1,'2PL':2,'3PL':3}[model_key]
            ia3.metric("AIC (approx)", f"{-2*log_lik + 2*(n_items*n_par):.0f}")

        st.markdown("---")
        st.markdown("#### Item Parameter Estimates")
        irt_display = df_res[['Item','IRT_b','IRT_a','IRT_c',
                               'IRT_INFIT','IRT_OUTFIT','Item_Info_Peak']].copy()
        irt_display.columns = ['Item','b (Difficulty)','a (Discrimination)',
                                'c (Pseudo-guess)','INFIT MNSQ','OUTFIT MNSQ','Peak Info']

        def style_irt(row):
            styles = [''] * len(row)
            b = row['b (Difficulty)']
            styles[1] = ('background-color:#ccffcc;color:black' if b<-1 else
                         'background-color:#ffcccc;color:black' if b>1 else
                         'background-color:#fff2cc;color:black')
            for idx,col in [(4,'INFIT MNSQ'),(5,'OUTFIT MNSQ')]:
                v = row[col]
                if np.isnan(v): continue
                styles[idx] = ('background-color:#ccffcc;color:black' if 0.70<=v<=1.30 else
                                'background-color:#ffcccc;color:black')
            return styles

        st.dataframe(
            irt_display.style.apply(style_irt, axis=1)
                .format("{:.4f}", subset=['b (Difficulty)','a (Discrimination)',
                                          'c (Pseudo-guess)','Peak Info'])
                .format(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A",
                        subset=['INFIT MNSQ','OUTFIT MNSQ']),
            use_container_width=True
        )
        if model_key=='1PL':
            st.caption("**INFIT/OUTFIT ideal: 0.70–1.30** (Wright & Masters, 1982). "
                       ">1.30 = misfit; <0.70 = overfitting/redundancy.")

        st.markdown("---")
        st.markdown("#### Item Characteristic Curves (ICC)")
        theta_range = np.linspace(-4,4,300)
        n_cols_icc  = min(4,n_items)
        n_rows_icc  = int(np.ceil(n_items/n_cols_icc))
        fig = dark_fig((n_cols_icc*3.5, n_rows_icc*2.8))
        gs  = GridSpec(n_rows_icc, n_cols_icc, figure=fig, hspace=0.45, wspace=0.35)
        for j,item in enumerate(item_cols):
            ri,ci = divmod(j,n_cols_icc)
            ax    = fig.add_subplot(gs[ri,ci])
            P     = irt_prob(theta_range, a_arr[j], b_arr[j], c_arr[j])
            ax.plot(theta_range, P, color=COLORS['blue'], lw=1.8)
            ax.axvline(b_arr[j], color=COLORS['orange'], linestyle='--', lw=1, alpha=0.8)
            ax.axhline(0.5,      color=COLORS['grey'],   linestyle=':',  lw=0.7, alpha=0.5)
            if model_key=='3PL':
                ax.axhline(c_arr[j], color=COLORS['purple'], linestyle=':', lw=1, alpha=0.6)
            ax.set_ylim(-0.05,1.10); ax.set_xlim(-4,4)
            ax.set_title(f"{item}  b={b_arr[j]:.2f} a={a_arr[j]:.2f}",
                         color='#e6edf3', fontsize=8.5, fontweight='bold', pad=4)
            style_ax(ax, xlabel='θ', ylabel='P(correct)')
        fig.suptitle(f"ICC — {irt_model}", color='#e6edf3', fontsize=11, fontweight='bold', y=1.01)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

        st.markdown("#### Item & Test Information Functions (IIF / TIF)")
        fig2    = dark_fig((12,4.5))
        ax_left = fig2.add_subplot(121)
        ax_right= fig2.add_subplot(122)
        tif_total = np.zeros(len(theta_range))
        palette   = plt.cm.plasma(np.linspace(0.1,0.9,n_items))
        for j,item in enumerate(item_cols):
            iif = compute_item_info(theta_range, a_arr[j], b_arr[j], c_arr[j])
            ax_left.plot(theta_range, iif, color=palette[j], lw=1.2, alpha=0.8, label=item)
            tif_total += iif
        style_ax(ax_left,'Item Information Functions (IIF)','θ','I(θ)')
        if n_items<=15:
            ax_left.legend(fontsize=6, facecolor='#21262d', labelcolor='#c9d1d9',
                           framealpha=0.7, loc='upper right', ncol=2)
        ax_right.plot(theta_range, tif_total, color=COLORS['cyan'], lw=2.2)
        ax_right.fill_between(theta_range, tif_total, alpha=0.15, color=COLORS['cyan'])
        sem_irt = np.where(tif_total>0, 1/np.sqrt(tif_total), np.nan)
        ax_r2   = ax_right.twinx()
        ax_r2.plot(theta_range, sem_irt, color=COLORS['orange'], lw=1.5, linestyle='--', alpha=0.8)
        ax_r2.set_ylabel('SEM(θ)=1/√I(θ)', color=COLORS['orange'], fontsize=8)
        ax_r2.tick_params(colors=COLORS['orange'], labelsize=7)
        for sp in ax_r2.spines.values(): sp.set_color('#30363d')
        style_ax(ax_right,'Test Information Function (TIF)','θ','Total I(θ)')
        fig2.tight_layout(); st.pyplot(fig2,use_container_width=True); plt.close(fig2)

        st.markdown("#### Wright Map (Person–Item Map)")
        st.caption("Students (θ) and items (b) on the same logit scale.")
        fig3 = dark_fig((8,max(5,n_items*0.5)))
        ax3  = fig3.add_subplot(111)
        bins_theta   = np.linspace(-4,4,25)
        hist_vals,be = np.histogram(theta_hat, bins=bins_theta)
        bin_centers  = (be[:-1]+be[1:])/2
        max_hist     = max(hist_vals.max(),1)
        ax3.barh(bin_centers, -hist_vals/max_hist*1.5, height=0.25,
                 color=COLORS['blue'], alpha=0.6)
        for j,item in enumerate(item_cols):
            ax3.scatter(0.05, b_arr[j], color=COLORS['orange'], s=70, zorder=5)
            ax3.text(0.12, b_arr[j], f"  {item} (b={b_arr[j]:.2f})",
                     va='center', ha='left', fontsize=7.5, color='#c9d1d9')
        ax3.axvline(0, color='#30363d', lw=0.8)
        ax3.set_xlim(-2,2); ax3.set_ylim(-4.2,4.2)
        ax3.set_xlabel('← Persons (count)   |   Items (b) →', color='#c9d1d9', fontsize=8)
        ax3.set_ylabel('Logit Scale (θ / b)', color='#c9d1d9', fontsize=8)
        p_patch = mpatches.Patch(color=COLORS['blue'],   alpha=0.6, label='Students (θ)')
        i_patch = mpatches.Patch(color=COLORS['orange'],             label='Items (b)')
        ax3.legend(handles=[p_patch,i_patch], fontsize=8,
                   facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
        style_ax(ax3,f'Wright Map — {irt_model}')
        fig3.tight_layout(); st.pyplot(fig3,use_container_width=True); plt.close(fig3)

        st.markdown("#### Student Ability Estimates (θ)")
        df_theta = pd.DataFrame({
            id_col_name:       df[id_col_name].values,
            'Total_Score':     total_scores.values,
            'θ (IRT Ability)': theta_hat.round(4),
            'SEM_θ':           [1/np.sqrt(max(ti,1e-9)) for ti in test_info_at_theta]
        }).sort_values('θ (IRT Ability)', ascending=False).reset_index(drop=True)
        df_theta['Rank'] = range(1, n_students+1)
        st.dataframe(
            df_theta.style.background_gradient(subset=['θ (IRT Ability)'], cmap='Blues'),
            use_container_width=True, height=300
        )

    # ──────────────────────────────────────────────────────────────────
    # TAB 3 — DISTRACTOR ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    with tab_dist:
        st.markdown("### Distractor Effectiveness Analysis")
        st.caption("DDI = prop(Lower) − prop(Upper) selecting each option. "
                   "Positive = functional distractor. Effective: chosen ≥ 5% AND DDI ≥ 0.")

        dist_data = []
        for item in item_cols:
            counts = df[item].astype(str).str.upper().str.strip().value_counts(normalize=True)
            row    = counts.to_dict(); row['Item'] = item
            dist_data.append(row)

        df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
        options_sorted = (sorted([c for c in df_dist.columns if len(str(c))==1]) +
                          sorted([c for c in df_dist.columns if len(str(c))>1]))
        df_dist = df_dist[[c for c in options_sorted if c in df_dist.columns]]

        df_dist_pct = df_dist.copy()
        for col in df_dist_pct.columns:
            df_dist_pct[col] = df_dist_pct[col].apply(lambda x: f"{x:.3f} ({x:.1%})")

        def tag_effectiveness(row, item_idx):
            ak = answer_key[item_idx]
            tags = []
            for opt,val in row.items():
                if opt==ak: continue
                if val<0.05: tags.append(f"⚠️ {opt} nonfunctional (<5%)")
                else:         tags.append(f"✅ {opt} effective")
            return " · ".join(tags) if tags else "—"

        df_dist_pct['Distractor Effectiveness'] = [
            tag_effectiveness(df_dist.iloc[i],i) for i in range(len(df_dist))
        ]
        st.dataframe(
            df_dist[df_dist.columns].style
                .format(lambda x: f"{x:.3f} ({x:.1%})")
                .background_gradient(cmap='YlGn', axis=1),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("#### Option Response Heatmap")
        fig = dark_fig((max(8,len(df_dist.columns)*1.5), max(5,n_items*0.55)))
        ax  = fig.add_subplot(111)
        data_arr = df_dist.values
        im = ax.imshow(data_arr, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(df_dist.columns)))
        ax.set_xticklabels(df_dist.columns, color='#c9d1d9', fontsize=9)
        ax.set_yticks(range(len(item_cols)))
        ax.set_yticklabels(list(item_cols), color='#c9d1d9', fontsize=8)
        for i in range(n_items):
            for j2,col in enumerate(df_dist.columns):
                val = data_arr[i,j2]
                ax.text(j2,i,f"{val:.2f}",ha='center',va='center',
                        fontsize=7, color='black' if val>0.4 else '#c9d1d9')
        plt.colorbar(im, ax=ax, label='Proportion Selected')
        style_ax(ax,'Option Selection Heatmap')
        fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # TAB 4 — STUDENT RANKING
    # ──────────────────────────────────────────────────────────────────
    with tab_rank:
        st.markdown("### Student Score Ranking & Group Assignment")
        st.caption(f"Top {group_percent}% = **Upper** · Bottom {group_percent}% = **Lower** "
                   f"(Kelley's criterion, n = {n_group} per group, using round())")

        col_r1,col_r2,col_r3 = st.columns(3)
        col_r1.metric("Upper Group (n)", n_group)
        col_r2.metric("Middle Group (n)", n_students-2*n_group)
        col_r3.metric("Lower Group (n)", n_group)

        def apply_rank_styling(row):
            bg = '#1a4731' if row['Group']=='Upper' else '#3d1212' if row['Group']=='Lower' else '#161b22'
            return [f'background-color:{bg};color:#e6edf3']*len(row)

        theta_dict = dict(zip(df[id_col_name].values, theta_hat))
        df_ranking_display = df_ranking.copy()
        df_ranking_display['θ (IRT)'] = df_ranking_display[id_col_name].map(theta_dict).round(4)
        st.dataframe(
            df_ranking_display.style.apply(apply_rank_styling, axis=1),
            use_container_width=True, height=500
        )

    # ──────────────────────────────────────────────────────────────────
    # TAB 5 — INTERPRETIVE REPORT
    # ──────────────────────────────────────────────────────────────────
    with tab_interp:
        st.markdown("### 📝 Automated Interpretive Report")
        st.caption("Narrative interpretations generated from actual data — suitable for research reports and theses.")

        st.markdown("#### 1. Test Administration Overview")
        st.info(
            f"This item analysis covers a test administered to **{n_students} students** "
            f"across **{n_items} items**. "
            f"Mean raw score = **{mean_score:.2f}** (SD = {std_score:.2f}), "
            f"representing **{mean_score/n_items:.1%}** of the maximum score. "
            f"Scores ranged from {int(total_scores.min())} to {int(total_scores.max())}."
        )

        st.markdown("#### 2. Reliability Analysis")
        st.markdown(rel_detail)
        st.markdown(sem_detail)
        st.markdown(
            f"**Cross-validation:** Cronbach's α = {alpha:.4f} (identical to KR-20 = {kr20:.4f} "
            f"for binary items). "
            f"Spearman-Brown split-half = {split_half:.4f} "
            + ("— consistent with KR-20, confirming internal consistency." if abs(split_half-kr20)<0.10 else
               "— diverges from KR-20; consider reviewing item placement order.")
        )

        st.markdown("#### 3. Item Difficulty & Discrimination Profile")
        item_summary,n_retain,n_revise,n_reject,n_easy,n_mod,n_hard = interpret_item_profile(
            df_res, n_items, validity_limit)
        st.markdown(item_summary)

        reject_items = df_res[df_res['DECISION']=='REJECT']['Item'].tolist()
        revise_items = df_res[df_res['DECISION']=='REVISE']['Item'].tolist()
        if reject_items:
            st.error(
                f"**Rejected items ({len(reject_items)}):** {', '.join(reject_items)}\n\n"
                "Failed critical psychometric criteria. Remove from test and rewrite completely before reuse.")
        if revise_items:
            st.warning(
                f"**Items for revision ({len(revise_items)}):** {', '.join(revise_items)}\n\n"
                "Borderline performance. Review stems, distractors, and answer keys.")

        st.markdown(f"#### 4. IRT Analysis ({irt_model})")
        st.markdown(interpret_irt_params(a_arr, b_arr, c_arr, model_key))
        st.markdown(
            f"IRT marginal reliability (Green, 1984 approximation) = **{irt_rel:.4f}**. "
            + ("Consistent with KR-20 — convergent evidence of internal consistency." if abs(irt_rel-kr20)<0.10 else
               f"Diverges from KR-20 ({kr20:.4f}) — may reflect non-normal latent distribution "
               f"or estimation instability; interpret with caution.")
        )
        r_theta_raw = float(np.corrcoef(theta_hat, total_scores)[0,1])
        st.markdown(
            f"Correlation between θ and raw scores: r = **{r_theta_raw:.4f}** "
            + ("— strong CTT/IRT alignment." if r_theta_raw>0.95 else
               "— moderate re-ranking by IRT; common when item discrimination varies substantially.")
        )
        if model_key=='1PL':
            poor_fit = df_res[
                (~df_res['IRT_INFIT'].isna()) &
                ((df_res['IRT_INFIT']>1.30)|(df_res['IRT_INFIT']<0.70))
            ]
            if len(poor_fit)>0:
                st.warning(
                    f"**Rasch Misfit:** {len(poor_fit)} item(s) outside 0.70–1.30 INFIT range: "
                    f"{', '.join(poor_fit['Item'].tolist())}. "
                    "These items violate the equal-discrimination assumption. "
                    "Consider switching to 2PL or revising these items.")

        st.markdown("#### 5. Distractor Functionality")
        n_neg_ddi = (df_res['Worst_DDI']<0).sum()
        n_severe  = (df_res['Worst_DDI']<-0.10).sum()
        if n_severe>0:
            st.error(
                f"**{n_severe} item(s)** with severely malfunctioning distractors (DDI < −0.10). "
                "These options attract more high-ability than low-ability students — "
                "possible keying errors or ambiguous wording. Immediate review required.")
        elif n_neg_ddi>0:
            st.warning(
                f"**{n_neg_ddi} item(s)** with at least one negative DDI distractor. "
                "Review for clarity and plausibility.")
        else:
            st.success("All distractors functioning correctly — "
                       "lower-group students select wrong options at higher rates than upper-group.")

        st.markdown("#### 6. Recommendations")
        recommendations = []
        if kr20<0.70:
            recommendations.append(
                "🔴 **Increase reliability:** KR-20 < 0.70 insufficient for individual assessment. "
                "Remove rejected items, revise borderline items, consider adding more items.")
        if n_reject>0:
            recommendations.append(
                f"🔴 **Remove/rewrite {n_reject} rejected item(s)** before any future use.")
        if n_revise>0:
            recommendations.append(
                f"🟡 **Revise {n_revise} item(s):** Review stems, improve distractors, verify keys.")
        if n_hard>n_items*0.30:
            recommendations.append(
                "🟡 **Too many difficult items** (p < 0.30): Scaffold or adjust cognitive level.")
        if n_easy>n_items*0.40:
            recommendations.append(
                "🟡 **Too many easy items** (p > 0.70): Add more challenging items to improve discrimination.")
        if model_key=='1PL' and 'poor_fit' in dir() and len(poor_fit)>0:
            recommendations.append(
                "🟡 **Rasch misfit detected:** Consider 2PL model or revise misfitting items.")
        if model_key=='3PL' and c_arr.mean()>0.20:
            recommendations.append(
                "🟡 **High pseudo-guessing (c̄ > 0.20):** Improve distractor quality.")
        if not recommendations:
            recommendations.append(
                "🟢 **No critical issues detected.** "
                "Test meets standard psychometric criteria for its intended purpose.")
        for rec in recommendations:
            st.markdown(rec)

    # ──────────────────────────────────────────────────────────────────
    # TAB 6 — DOWNLOAD
    # ──────────────────────────────────────────────────────────────────
    with tab_report:
        st.markdown("### 📥 Export Full Report")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            wb = writer.book
            hdr_fmt    = wb.add_format({'bold':True,'bg_color':'#1f3864','font_color':'white',
                                        'border':1,'align':'center','valign':'vcenter'})
            retain_fmt = wb.add_format({'bg_color':'#c6efce','font_color':'#276221','bold':True,'border':1})
            revise_fmt = wb.add_format({'bg_color':'#ffeb9c','font_color':'#9c6500','bold':True,'border':1})
            reject_fmt = wb.add_format({'bg_color':'#ffc7ce','font_color':'#9c0006','bold':True,'border':1})
            num_fmt    = wb.add_format({'num_format':'0.0000','border':1,'align':'center'})
            txt_fmt    = wb.add_format({'border':1})
            upper_fmt  = wb.add_format({'bg_color':'#c6efce','border':1})
            lower_fmt  = wb.add_format({'bg_color':'#ffc7ce','border':1})
            mid_fmt    = wb.add_format({'border':1})
            title_fmt  = wb.add_format({'bold':True,'font_size':14,'font_color':'#1f3864'})

            # Sheet 1 — CTT
            ctt_cols = ["Item","p","p_Eval","q","pq","Var",
                        "p_Upper","p_Lower","d","d_Eval",
                        "Best_DDI","Worst_DDI","r_pbis","r_Eval","DECISION","REASON"]
            df_res[ctt_cols].to_excel(writer, index=False, sheet_name='CTT_Item_Analysis', startrow=1)
            ws1 = writer.sheets['CTT_Item_Analysis']
            ws1.write(0,0,'CTT Item Analysis Report',title_fmt)
            for ci,cn in enumerate(ctt_cols): ws1.write(1,ci,cn,hdr_fmt)
            for ri,row in df_res[ctt_cols].iterrows():
                for ci,cn in enumerate(ctt_cols):
                    val = row[cn]
                    if cn=='DECISION':
                        fmt = retain_fmt if val=='RETAIN' else revise_fmt if val=='REVISE' else reject_fmt
                        ws1.write(ri+2,ci,val,fmt)
                    elif isinstance(val,float): ws1.write(ri+2,ci,val,num_fmt)
                    else: ws1.write(ri+2,ci,val,txt_fmt)
            ws1.set_column('A:A',12); ws1.set_column('B:N',12)
            ws1.set_column('O:O',10); ws1.set_column('P:P',50)

            # Sheet 2 — IRT
            irt_exp = df_res[['Item','IRT_b','IRT_a','IRT_c',
                               'IRT_INFIT','IRT_OUTFIT','Item_Info_Peak']].copy()
            irt_exp.columns = ['Item','b (Difficulty)','a (Discrimination)','c (Pseudo-guess)',
                                'INFIT MNSQ','OUTFIT MNSQ','Peak Information']
            irt_exp.to_excel(writer, index=False, sheet_name='IRT_Parameters', startrow=1)
            ws2 = writer.sheets['IRT_Parameters']
            ws2.write(0,0,f'IRT Parameter Estimates — {irt_model}',title_fmt)
            for ci,cn in enumerate(irt_exp.columns): ws2.write(1,ci,cn,hdr_fmt)
            ws2.set_column('A:G',18)

            # Sheet 3 — Rankings
            df_theta_exp = df_ranking.copy()
            df_theta_exp['θ (IRT Ability)'] = df_ranking[id_col_name].map(
                dict(zip(df[id_col_name].values, theta_hat))).round(4)
            df_theta_exp.to_excel(writer, index=False, sheet_name='Student_Ranking', startrow=1)
            ws3 = writer.sheets['Student_Ranking']
            ws3.write(0,0,'Student Score Ranking & IRT Ability',title_fmt)
            for ci,cn in enumerate(df_theta_exp.columns): ws3.write(1,ci,cn,hdr_fmt)
            for ri,row in df_theta_exp.iterrows():
                grp = row.get('Group','')
                fmt = upper_fmt if grp=='Upper' else lower_fmt if grp=='Lower' else mid_fmt
                for ci,val in enumerate(row): ws3.write(ri+2,ci,val,fmt)
            ws3.set_column('A:E',18)

            # Sheet 4 — Distractor
            df_dist_pct.to_excel(writer, index=True, sheet_name='Distractor_Analysis', startrow=1)
            ws4 = writer.sheets['Distractor_Analysis']
            ws4.write(0,0,'Distractor Effectiveness (Proportion Selected)',title_fmt)
            ws4.set_column('A:Z',16)

            # Sheet 5 — Reliability & Notes
            rel_label_val,_,_ = interpret_reliability(kr20, sem, n_items)
            rel_df = pd.DataFrame({
                "Metric": ["N (Students)","k (Items)","Mean","SD","KR-20","Cronbach's α",
                           "Split-Half (SB)","SEM","IRT Rel. (approx)","Log-Likelihood",
                           "n_group (Kelley)","KR-20 Interpretation","SEM Interpretation"],
                "Value":  [n_students, n_items,
                           f"{mean_score:.2f}", f"{std_score:.2f}",
                           f"{kr20:.4f}", f"{alpha:.4f}", f"{split_half:.4f}",
                           f"{sem:.4f}", f"{irt_rel:.4f}", f"{log_lik:.2f}",
                           f"{n_group} (round({n_students}×{group_percent}%))",
                           f"{rel_label_val} reliability",
                           f"±{sem:.4f} (68%) or ±{2*sem:.4f} (95%)"],
                "Methodological Notes": [
                    "Total test takers",
                    "Total items in the test",
                    "Average raw score",
                    "Standard deviation of total scores",
                    "Kuder & Richardson (1937). Exact formula for dichotomous items. "
                    "Equivalent to Cronbach's alpha for binary scoring.",
                    "Identical to KR-20 for binary items (Cronbach, 1951).",
                    "Spearman-Brown odd/even split-half. Cross-validates KR-20.",
                    "SEM = SD × √(1−KR-20). Lower = more precise measurement.",
                    "Green (1984) marginal IRT reliability approximation. "
                    "Not equivalent to KR-20; use for IRT-framework comparison only.",
                    "EM marginal log-likelihood. Higher (less negative) = better model fit.",
                    "Group size uses round() not int() to avoid systematic truncation bias.",
                    "Excellent ≥0.90 | High 0.80–0.89 | Acceptable 0.70–0.79 | Low <0.70 "
                    "(Ebel & Frisbie, 1991; Crocker & Algina, 1986)",
                    "True score CI based on normal distribution of measurement errors."
                ]
            })
            rel_df.to_excel(writer, index=False, sheet_name='Reliability_Report', startrow=1)
            ws5 = writer.sheets['Reliability_Report']
            ws5.write(0,0,'Reliability Summary & Methodological Notes',title_fmt)
            for ci,cn in enumerate(rel_df.columns): ws5.write(1,ci,cn,hdr_fmt)
            ws5.set_column('A:A',25); ws5.set_column('B:B',25); ws5.set_column('C:C',75)

        st.download_button(
            label="📥 Download Complete Item Analysis Report (Excel)",
            data=buf.getvalue(),
            file_name="Item_Analysis_Pro_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.success("✅ Report includes: CTT Matrix · IRT Parameters · Student Rankings + θ · "
                   "Distractor Analysis · Reliability Report with methodological references")

else:
    st.markdown("### Getting Started")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("""
**Step 1: Prepare your CSV file**
```
ANSWER,A,C,A,D,B,...
StudentID,Q1,Q2,Q3,Q4,Q5,...
S001,A,C,B,D,A,...
S002,B,C,A,A,B,...
```
*Row 1 = answer key. Row 2 = headers. Row 3+ = responses.*
        """)
    with col2:
        st.markdown("""
**Step 2: Configure Settings**
- Kelley's grouping % (default 27%)
- r_pbis threshold (default 0.25)
- IRT model (1PL / 2PL / 3PL)
        """)
    with col3:
        st.markdown("""
**Step 3: Upload & Analyze**
- Upload your single CSV file
- Results appear instantly
- Download full Excel report
        """)
    st.info(
        "📌 **Methodological note:** IRT estimation uses EM algorithm with Gauss-Hermite quadrature "
        "and normal prior on θ. For N < 100, use 1PL (Rasch) for parameter stability. "
        "2PL/3PL recommended for N ≥ 200 (Baker & Kim, 2004).")
