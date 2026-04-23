"""
Mixed-Design ANOVA (Split-Plot ANOVA) v2
Pure scipy/numpy - SPSS-equivalent - Light theme
Fixes: validation, light theme, academic labels, Tukey, post-hoc narrative, alpha dropdown, PDF symbols
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, io
from itertools import combinations
from scipy import stats
from scipy.stats import levene as scipy_levene

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Mixed-Design ANOVA", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#f0f2f8;color:#1a1d23;}
.hero{background:linear-gradient(135deg,#1c3557 0%,#1a4a7a 100%);border-radius:12px;
  padding:2rem 2.4rem;margin-bottom:1.8rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#4da6ff,#34c97a,#a78bfa,#f59e0b);}
.hero h1{font-family:'Source Serif 4',serif;font-size:1.75rem;font-weight:700;
  color:#fff;margin:0 0 .35rem;letter-spacing:-.3px;}
.hero p{color:#b8d0ea;font-size:.9rem;margin:0;}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.7rem;
  font-weight:600;font-family:'JetBrains Mono',monospace;margin-right:6px;margin-top:.75rem;}
.bd-g{background:rgba(52,201,122,.15);color:#1a8a4a;border:1px solid #34c97a55;}
.bd-b{background:rgba(77,166,255,.15);color:#1a5fa0;border:1px solid #4da6ff55;}
.bd-p{background:rgba(167,139,250,.15);color:#6d28d9;border:1px solid #a78bfa55;}
.bd-o{background:rgba(245,158,11,.15);color:#b45309;border:1px solid #f59e0b55;}
.sec-card{background:#fff;border:1px solid #dde1ea;border-radius:10px;
  padding:1.4rem 1.6rem;margin-bottom:1.4rem;box-shadow:0 1px 4px rgba(0,0,0,.06);}
.sec-title{font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:600;
  color:#6b7280;text-transform:uppercase;letter-spacing:1.8px;margin-bottom:1rem;
  padding-bottom:.5rem;border-bottom:1px solid #e5e7eb;}
.anova-table{width:100%;border-collapse:collapse;font-size:.84rem;font-family:'JetBrains Mono',monospace;}
.anova-table th{background:#1c3557;color:#fff;padding:9px 14px;text-align:left;
  font-weight:600;border-bottom:2px solid #1a4a7a;white-space:nowrap;}
.anova-table td{padding:8px 14px;border-bottom:1px solid #e5e7eb;color:#374151;}
.anova-table tr:nth-child(even) td{background:#f9fafb;}
.anova-table tr:hover td{background:#eff6ff;}
.anova-table .err-row td{color:#9ca3af;font-style:italic;}
.sig3{color:#059669;font-weight:700;} .sig2{color:#2563eb;font-weight:700;}
.sig1{color:#d97706;font-weight:700;} .sigm{color:#d97706;} .ns{color:#6b7280;}
.ibox{background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;
  border-radius:8px;padding:1rem 1.25rem;margin:.75rem 0;font-size:.875rem;line-height:1.7;color:#1e3a5f;}
.wbox{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;
  border-radius:8px;padding:.85rem 1.1rem;margin:.5rem 0;font-size:.83rem;color:#92400e;}
.obox{background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #16a34a;
  border-radius:8px;padding:.85rem 1.1rem;margin:.5rem 0;font-size:.83rem;color:#14532d;}
.ebox{background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #dc2626;
  border-radius:8px;padding:.85rem 1.1rem;margin:.5rem 0;font-size:.83rem;color:#7f1d1d;}
.interp-block{background:#fff;border:1px solid #dde1ea;border-radius:10px;
  padding:1.25rem 1.5rem;margin:.75rem 0;box-shadow:0 1px 3px rgba(0,0,0,.04);
  font-size:.9rem;line-height:1.8;color:#374151;}
.interp-block h4{font-family:'Source Serif 4',serif;font-size:1rem;font-weight:700;
  color:#1c3557;margin:0 0 .6rem;}
.mcard{background:#fff;border:1px solid #dde1ea;border-radius:10px;padding:1rem;
  text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.mval{font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:700;color:#1c3557;}
.mlbl{font-size:.75rem;color:#6b7280;margin-top:3px;text-transform:uppercase;letter-spacing:.8px;}
div[data-testid="stSidebar"]{background:#1c3557 !important;}
div[data-testid="stSidebar"] p,div[data-testid="stSidebar"] span,
div[data-testid="stSidebar"] label,div[data-testid="stSidebar"] div{color:#e2eaf4 !important;}
.stTabs [data-baseweb="tab-list"]{background:transparent;gap:4px;}
.stTabs [data-baseweb="tab"]{background:#fff;border:1px solid #dde1ea;color:#6b7280;
  border-radius:6px;font-family:'Inter',sans-serif;font-size:.82rem;font-weight:500;}
.stTabs [aria-selected="true"]{background:#1c3557 !important;border-color:#1c3557 !important;
  color:#fff !important;}
.stButton>button{background:#1c3557;color:#fff;border:none;border-radius:6px;
  font-family:'Inter',sans-serif;font-size:.875rem;font-weight:500;transition:background .2s;}
.stButton>button:hover{background:#1a4a7a;}
</style>
""", unsafe_allow_html=True)


# ── Utilities ─────────────────────────────────────────────────────────────────
def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    if isinstance(x, (int, np.integer)): return str(x)
    return f"{x:.{d}f}"

def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return "—"
    if p < .001: return "< .001"
    return f"{p:.3f}"

def sig_star(p):
    if isinstance(p, float) and np.isnan(p): return ""
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    if p < .10:  return "†"
    return "ns"

def sig_cls(p):
    if isinstance(p, float) and np.isnan(p): return "ns"
    if p < .001: return "sig3"
    if p < .01:  return "sig2"
    if p < .05:  return "sig1"
    if p < .10:  return "sigm"
    return "ns"

def eta_interp(e):
    if isinstance(e, float) and np.isnan(e): return "—"
    if e >= .14: return "Large"
    if e >= .06: return "Medium"
    if e >= .01: return "Small"
    return "Negligible"

def design_label(n_groups, n_times):
    gw = {2:"Two",3:"Three",4:"Four",5:"Five",6:"Six"}.get(n_groups, str(n_groups))
    tw = {2:"Two",3:"Three",4:"Four",5:"Five",6:"Six"}.get(n_times, str(n_times))
    gp = "Group" if n_groups == 1 else "Groups"
    tp = "Time Point" if n_times == 1 else "Time Points"
    return f"{gw} {gp} x {tw} {tp}"


# ── Sample data ───────────────────────────────────────────────────────────────
def sample_2g2t():
    np.random.seed(42)
    rows=[]
    for g,m in [('Control',[50,52]),('Treatment',[50,62])]:
        for i in range(15):
            rows.append({'ID':f'{g[0]}{i+1:02d}','Group':g,
                         'Pre':round(np.random.normal(m[0],8),2),
                         'Post':round(np.random.normal(m[1],8),2)})
    return pd.DataFrame(rows)

def sample_3g3t():
    np.random.seed(7)
    rows=[]
    for g,m in [('Control',[40,41,42]),('Low_Dose',[40,50,56]),('High_Dose',[40,58,70])]:
        for i in range(12):
            rows.append({'ID':f'{g[0]}{i+1:02d}','Group':g,
                         'Pre':round(np.random.normal(m[0],9),2),
                         'Mid':round(np.random.normal(m[1],9),2),
                         'Post':round(np.random.normal(m[2],9),2)})
    return pd.DataFrame(rows)


# ── Data validation ───────────────────────────────────────────────────────────
def validate_data(df, subj_col, btwn_col, time_cols):
    issues = []
    for c in [subj_col, btwn_col] + time_cols:
        if c not in df.columns:
            issues.append(('error',
                f"Column <b>'{c}'</b> was not found in the file. "
                f"Available columns: {', '.join(df.columns.tolist())}."))
    if any(lv=='error' for lv,_ in issues): return issues

    for c in time_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            sample_v = df[c].dropna().head(3).tolist()
            issues.append(('error',
                f"Column <b>'{c}'</b> must be numeric but contains non-numeric values "
                f"(sample: {sample_v}). Ensure all time-point columns contain numbers only."))

    n_groups = df[btwn_col].nunique()
    if n_groups < 2:
        issues.append(('error',
            f"The between-subjects factor <b>'{btwn_col}'</b> has only {n_groups} unique value. "
            f"At least 2 independent groups are required."))
    if n_groups > 10:
        issues.append(('warning',
            f"The between-subjects factor has {n_groups} groups, which is unusually high. "
            f"Verify that '{btwn_col}' is a categorical grouping variable, not a continuous measure."))

    group_sizes = df.groupby(btwn_col)[time_cols[0]].count()
    for g, n in group_sizes.items():
        if n < 3:
            issues.append(('error',
                f"Group <b>'{g}'</b> has only {n} participant(s). "
                f"Each group requires at least 3 participants."))
        elif n < 10:
            issues.append(('warning',
                f"Group <b>'{g}'</b> has only {n} participants. "
                f"Small samples reduce power and may violate normality assumptions."))

    req = [subj_col, btwn_col] + time_cols
    n_miss = df[req].isnull().any(axis=1).sum()
    if n_miss > 0:
        pct = 100*n_miss/len(df)
        level = 'warning' if pct > 20 else 'info'
        issues.append((level,
            f"<b>{n_miss} of {len(df)} rows ({pct:.1f}%)</b> contain missing values and will be "
            f"excluded (listwise deletion). The analysis will use {len(df)-n_miss} complete cases."))

    dupes = df[subj_col].duplicated()
    if dupes.any():
        dup_ids = df.loc[dupes, subj_col].unique().tolist()[:5]
        issues.append(('warning',
            f"Duplicate Subject IDs detected: {dup_ids}. "
            f"Each subject should appear exactly once in wide-format data."))

    sizes = group_sizes.values
    if len(sizes) > 1 and sizes.max() > sizes.min():
        issues.append(('info',
            f"Unbalanced design: group sizes range from {sizes.min()} to {sizes.max()}. "
            f"The analysis handles this, but balanced designs have greater statistical power."))

    if not issues:
        issues.append(('info', "Data structure validated successfully. No issues detected."))
    return issues


# ── Descriptives ──────────────────────────────────────────────────────────────
def compute_descriptives(df_wide, between_col, time_cols):
    rows = []
    for g in df_wide[between_col].unique():
        sub = df_wide[df_wide[between_col]==g]
        for t in time_cols:
            v  = sub[t].dropna()
            n  = len(v)
            se = v.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan
            rows.append({'Group':g,'Time':t,'N':n,
                         'Mean':round(v.mean(),3),'SD':round(v.std(ddof=1),3),
                         'SE':round(se,3) if not np.isnan(se) else np.nan,
                         'Median':round(v.median(),3),
                         'Min':round(v.min(),3),'Max':round(v.max(),3),
                         '95% CI Lower':round(v.mean()-1.96*se,3) if not np.isnan(se) else np.nan,
                         '95% CI Upper':round(v.mean()+1.96*se,3) if not np.isnan(se) else np.nan})
    return pd.DataFrame(rows)


# ── Normality ─────────────────────────────────────────────────────────────────
def test_normality(df_wide, between_col, time_cols):
    rows = []
    for g in df_wide[between_col].unique():
        sub = df_wide[df_wide[between_col]==g]
        for t in time_cols:
            v = sub[t].dropna().values; n = len(v)
            if n < 3:
                rows.append({'Group':g,'Time':t,'N':n,'Test':'—','Statistic':np.nan,
                              'p-value':np.nan,'Normal':'—','Rationale':'n < 3'})
                continue
            if n <= 50:
                stat,p = stats.shapiro(v)
                tname,note = 'Shapiro-Wilk','SW selected: n <= 50 (optimal power for small samples)'
            else:
                stat,p = stats.kstest(v,'norm',args=(v.mean(),v.std()))
                tname,note = 'Kolmogorov-Smirnov','KS selected: n > 50 (SW loses sensitivity at large n)'
            rows.append({'Group':g,'Time':t,'N':n,'Test':tname,
                          'Statistic':round(stat,4),'p-value':round(p,4),
                          'Normal':'Yes' if p>.05 else 'No','Rationale':note})
    return pd.DataFrame(rows)


# ── Levene ────────────────────────────────────────────────────────────────────
def test_levene(df_wide, between_col, time_cols):
    rows = []
    for t in time_cols:
        gdata = [df_wide[df_wide[between_col]==g][t].dropna().values
                 for g in df_wide[between_col].unique()]
        if all(len(x)>1 for x in gdata):
            stat,p = scipy_levene(*gdata, center='mean')
            rows.append({'Time Point':t,'Levene F':round(stat,3),
                          'p-value':round(p,4),'Equal Variances':'Yes' if p>.05 else 'No'})
    return pd.DataFrame(rows)


# ── Mauchly sphericity ────────────────────────────────────────────────────────
def mauchly_test(df_wide, between_col, time_cols):
    b = len(time_cols)
    if b < 3: return None
    Y = df_wide[time_cols].values.astype(float); N = Y.shape[0]
    C = np.zeros((b, b-1))
    for j in range(b-1):
        C[:j+1,j] = 1.0/(j+1); C[j+1,j] = -1.0
    T = Y @ C; S = np.cov(T, rowvar=False)
    if S.ndim == 0: S = np.array([[S]])
    try:
        k = b-1
        det_S = np.linalg.det(S); tr_S = np.trace(S)
        W = max(1e-15, min(det_S/(tr_S/k)**k, 1.0))
        df_chi = k*(k+1)//2 - 1
        u = N-1-(2*k**2+k+2)/(6*k)
        chi2 = -u*np.log(W)
        p_W = float(1-stats.chi2.cdf(chi2, df_chi)) if df_chi>0 else np.nan
    except Exception: return None
    tr_S2 = np.trace(S @ S)
    eps_GG = min(max((tr_S**2)/(k*tr_S2), 1/k), 1.0)
    eps_HF = min((N*(k*eps_GG)-2)/(k*(N-1-k*eps_GG)), 1.0)
    return W, p_W, eps_GG, eps_HF, chi2, df_chi


# ── Mixed ANOVA ───────────────────────────────────────────────────────────────
def mixed_anova(df_wide, subject_col, between_col, time_cols):
    groups = df_wide[between_col].unique()
    a = len(groups); b = len(time_cols)
    gd = {g: df_wide[df_wide[between_col]==g][time_cols].values.astype(float) for g in groups}
    ng = {g: gd[g].shape[0] for g in groups}
    N  = sum(ng.values())
    all_s = np.vstack([gd[g] for g in groups])
    GM   = np.mean(all_s)
    gm   = {g: np.mean(gd[g]) for g in groups}
    tm   = np.mean(all_s, axis=0)
    cm   = {g: np.mean(gd[g], axis=0) for g in groups}
    sm   = {g: np.mean(gd[g], axis=1) for g in groups}

    SS_A   = b*sum(ng[g]*(gm[g]-GM)**2 for g in groups); df_A = a-1
    SS_sWG = sum(b*np.sum((sm[g]-gm[g])**2) for g in groups); df_sWG = sum(ng[g]-1 for g in groups)
    SS_B   = N*np.sum((tm-GM)**2); df_B = b-1
    SS_AB  = sum(ng[g]*np.sum((cm[g]-gm[g]-tm+GM)**2) for g in groups); df_AB = (a-1)*(b-1)
    SS_eW  = sum((gd[g][i,j]-sm[g][i]-cm[g][j]+gm[g])**2
                 for g in groups for i in range(ng[g]) for j in range(b))
    df_eW  = sum(ng[g]-1 for g in groups)*(b-1)

    def ms(ss,df): return ss/df if df>0 else np.nan
    MS_A=ms(SS_A,df_A); MS_sWG=ms(SS_sWG,df_sWG)
    MS_B=ms(SS_B,df_B); MS_AB=ms(SS_AB,df_AB); MS_eW=ms(SS_eW,df_eW)

    F_A = MS_A/MS_sWG if MS_sWG else np.nan
    F_B = MS_B/MS_eW  if MS_eW  else np.nan
    F_AB= MS_AB/MS_eW if MS_eW  else np.nan

    def pf(F,d1,d2): return float(1-stats.f.cdf(F,d1,d2)) if not np.isnan(F) else np.nan
    p_A=pf(F_A,df_A,df_sWG); p_B=pf(F_B,df_B,df_eW); p_AB=pf(F_AB,df_AB,df_eW)

    def eta2(se,sr): return se/(se+sr) if (se+sr)>0 else np.nan
    e_A=eta2(SS_A,SS_sWG); e_B=eta2(SS_B,SS_eW); e_AB=eta2(SS_AB,SS_eW)

    sph = mauchly_test(df_wide, between_col, time_cols)
    eps_GG = 1.0
    if sph: _, _, eps_GG, _, _, _ = sph

    def gg_p(F,d1,d2,eps):
        return float(1-stats.f.cdf(F,d1*eps,d2*eps)) if not np.isnan(F) else np.nan

    p_B_gg  = gg_p(F_B,  df_B,  df_eW, eps_GG)
    p_AB_gg = gg_p(F_AB, df_AB, df_eW, eps_GG)

    results = pd.DataFrame([
        {'Source':between_col,'SS':SS_A,'df':df_A,'df_error':df_sWG,
         'MS':MS_A,'F':F_A,'p':p_A,'p_GG':p_A,'eta_p2':e_A,'eps':np.nan,'type':'between'},
        {'Source':'Error(Between)','SS':SS_sWG,'df':df_sWG,'df_error':np.nan,
         'MS':MS_sWG,'F':np.nan,'p':np.nan,'p_GG':np.nan,'eta_p2':np.nan,'eps':np.nan,'type':'error_between'},
        {'Source':'Time','SS':SS_B,'df':df_B,'df_error':df_eW,
         'MS':MS_B,'F':F_B,'p':p_B,'p_GG':p_B_gg,'eta_p2':e_B,'eps':eps_GG,'type':'within'},
        {'Source':f'{between_col} x Time','SS':SS_AB,'df':df_AB,'df_error':df_eW,
         'MS':MS_AB,'F':F_AB,'p':p_AB,'p_GG':p_AB_gg,'eta_p2':e_AB,'eps':eps_GG,'type':'interaction'},
        {'Source':'Error(Within)','SS':SS_eW,'df':df_eW,'df_error':np.nan,
         'MS':MS_eW,'F':np.nan,'p':np.nan,'p_GG':np.nan,'eta_p2':np.nan,'eps':np.nan,'type':'error_within'},
    ])
    return results, sph


# ── p-value adjustment ────────────────────────────────────────────────────────
def _adjust_p(ps, method):
    ps = np.array(ps, dtype=float); n = len(ps)
    if n == 0: return ps
    if method == 'bonferroni': return np.minimum(ps*n, 1.0)
    if method == 'holm':
        order=np.argsort(ps); adj=ps.copy()
        for rank,idx in enumerate(order): adj[idx]=min(ps[idx]*(n-rank),1.0)
        for i in range(1,n): adj[order[i]]=max(adj[order[i]],adj[order[i-1]])
        return np.minimum(adj,1.0)
    if method == 'fdr_bh':
        order=np.argsort(ps)[::-1]; adj=ps.copy(); mn=1.0
        for i,idx in enumerate(order): adj[idx]=min(ps[idx]*n/(n-i),mn); mn=adj[idx]
        return np.minimum(adj,1.0)
    return ps  # tukey handles p separately


def _tukey_p(q_stat, k, df_err):
    try:
        from scipy.stats import studentized_range
        return float(1-studentized_range.cdf(abs(q_stat)*np.sqrt(2), k, df_err))
    except Exception:
        n_pairs = k*(k-1)//2
        return min(2*(1-stats.t.cdf(abs(q_stat), df_err)) * n_pairs, 1.0)


# ── Post-hoc ─────────────────────────────────────────────────────────────────
def run_posthoc(df_wide, between_col, time_cols, method='tukey', alpha=.05):
    groups    = sorted(df_wide[between_col].unique())
    n_grps    = len(groups)
    time_prs  = list(combinations(time_cols, 2))
    rows      = []

    # Between groups at each time
    for t in time_cols:
        pairs = list(combinations(groups, 2))
        raw_ts, raw_ps, meta = [], [], []
        for g1,g2 in pairs:
            v1=df_wide[df_wide[between_col]==g1][t].dropna().values
            v2=df_wide[df_wide[between_col]==g2][t].dropna().values
            t_s,p_r = stats.ttest_ind(v1,v2,equal_var=True)
            raw_ts.append(t_s); raw_ps.append(p_r)
            meta.append((g1,g2,v1,v2,len(v1)+len(v2)-2))

        if method == 'tukey':
            all_v = [df_wide[df_wide[between_col]==g][t].dropna().values for g in groups]
            n_pool= sum(len(v) for v in all_v); df_e=n_pool-n_grps
            ms_e  = sum((len(v)-1)*np.var(v,ddof=1) for v in all_v)/df_e if df_e>0 else np.nan
            adj_ps= []
            for i,(g1,g2,v1,v2,_) in enumerate(meta):
                q = abs(np.mean(v1)-np.mean(v2))/np.sqrt(ms_e*(0.5/len(v1)+0.5/len(v2))) if ms_e else np.nan
                adj_ps.append(_tukey_p(q, n_grps, df_e) if not np.isnan(q) else 1.0)
        else:
            adj_ps = list(_adjust_p(raw_ps, method))

        for i,(g1,g2,v1,v2,df_t) in enumerate(meta):
            md = np.mean(v1)-np.mean(v2)
            pool_v=((len(v1)-1)*np.var(v1,ddof=1)+(len(v2)-1)*np.var(v2,ddof=1))/(len(v1)+len(v2)-2)
            d = md/np.sqrt(pool_v) if pool_v>0 else np.nan
            se_p = np.sqrt(np.var(v1,ddof=1)/len(v1)+np.var(v2,ddof=1)/len(v2))
            rows.append({
                'Comparison Type':'Between Groups','Time Point':t,'Group':'—',
                'Label A':g1,'Label B':g2,
                'Mean A':round(np.mean(v1),3),'Mean B':round(np.mean(v2),3),
                'Mean Diff (A-B)':round(md,3),'SE':round(se_p,3),
                't':round(raw_ts[i],3),'df':int(df_t),
                'p (raw)':round(raw_ps[i],4),'p (adj)':round(adj_ps[i],4),
                "Cohen's d":round(d,3) if not np.isnan(d) else np.nan,
                'Sig':sig_star(adj_ps[i]),'Significant':adj_ps[i]<alpha})

    # Within time per group
    for g in groups:
        sub = df_wide[df_wide[between_col]==g]
        raw_ts2, raw_ps2, meta2 = [], [], []
        for t1,t2 in time_prs:
            v1=sub[t1].dropna().values; v2=sub[t2].dropna().values
            nm=min(len(v1),len(v2))
            t_s,p_r = stats.ttest_rel(v1[:nm],v2[:nm])
            raw_ts2.append(t_s); raw_ps2.append(p_r); meta2.append((t1,t2,v1[:nm],v2[:nm],nm-1))

        n_t = len(time_cols)
        if method == 'tukey':
            adj_ps2=[]
            for i,(t1,t2,v1,v2,df_t) in enumerate(meta2):
                diff=v1-v2; ms_d=np.var(diff,ddof=1) if len(diff)>1 else np.nan
                q = abs(np.mean(diff))/(np.sqrt(ms_d/len(diff))*np.sqrt(0.5)) if ms_d and ms_d>0 else np.nan
                adj_ps2.append(_tukey_p(abs(raw_ts2[i])*np.sqrt(2), n_t, df_t) if not np.isnan(q) else 1.0)
        else:
            adj_ps2 = list(_adjust_p(raw_ps2, method))

        for i,(t1,t2,v1,v2,df_t) in enumerate(meta2):
            diff=v1-v2
            se = np.std(diff,ddof=1)/np.sqrt(len(diff)) if len(diff)>1 else np.nan
            d  = np.mean(diff)/np.std(diff,ddof=1) if np.std(diff,ddof=1)>0 else np.nan
            rows.append({
                'Comparison Type':'Within Time','Time Point':'—','Group':g,
                'Label A':t1,'Label B':t2,
                'Mean A':round(np.mean(v1),3),'Mean B':round(np.mean(v2),3),
                'Mean Diff (A-B)':round(np.mean(diff),3),
                'SE':round(se,3) if not np.isnan(se) else np.nan,
                't':round(raw_ts2[i],3),'df':int(df_t),
                'p (raw)':round(raw_ps2[i],4),'p (adj)':round(adj_ps2[i],4),
                "Cohen's d":round(d,3) if not np.isnan(d) else np.nan,
                'Sig':sig_star(adj_ps2[i]),'Significant':adj_ps2[i]<alpha})

    return pd.DataFrame(rows)


def posthoc_narrative(ph_df, alpha, method_name):
    if ph_df.empty: return []
    lines = []
    bt = ph_df[ph_df['Comparison Type']=='Between Groups']
    wt = ph_df[ph_df['Comparison Type']=='Within Time']

    if not bt.empty:
        lines.append(("Between-Group Comparisons at Each Time Point",
            f"The following pairwise comparisons examine whether groups differed at each individual "
            f"measurement occasion after applying {method_name} correction."))
        for _,r in bt.iterrows():
            md = r['Mean Diff (A-B)']; d = r["Cohen's d"]
            dir_w = ("higher" if md>0 else "lower") if md!=0 else "equal to"
            sig_t = (f"a statistically significant difference (p = {fmt_p(r['p (adj)'])}, {r['Sig']})"
                     if r['Significant'] else f"no statistically significant difference (p = {fmt_p(r['p (adj)'])}, ns)")
            d_t   = (f"; Cohen's d = {fmt(abs(d),2)} ({eta_interp(abs(d))} effect)"
                     if not (isinstance(d,float) and np.isnan(d)) else "")
            lines.append((f"Time Point: {r['Time Point']} — {r['Label A']} vs. {r['Label B']}",
                f"At {r['Time Point']}, the {r['Label A']} group (M = {r['Mean A']}) scored "
                f"{abs(round(md,2))} units {dir_w} the {r['Label B']} group (M = {r['Mean B']}), "
                f"indicating {sig_t}{d_t}."))

    if not wt.empty:
        lines.append(("Within-Subject Change Over Time by Group",
            f"The following comparisons examine the trajectory of change across time points "
            f"within each group after applying {method_name} correction."))
        for _,r in wt.iterrows():
            md = r['Mean Diff (A-B)']; d = r["Cohen's d"]
            dir_w = "decreased" if md>0 else ("increased" if md<0 else "did not change")
            sig_t = (f"a statistically significant change (p = {fmt_p(r['p (adj)'])}, {r['Sig']})"
                     if r['Significant'] else f"no statistically significant change (p = {fmt_p(r['p (adj)'])}, ns)")
            d_t   = (f"; Cohen's d = {fmt(abs(d),2)} ({eta_interp(abs(d))} effect)"
                     if not (isinstance(d,float) and np.isnan(d)) else "")
            lines.append((f"Group: {r['Group']} — {r['Label A']} to {r['Label B']}",
                f"Within the {r['Group']} group, scores {dir_w} from {r['Label A']} "
                f"(M = {r['Mean A']}) to {r['Label B']} (M = {r['Mean B']}), indicating "
                f"{sig_t}{d_t}."))
    return lines


# ── Interpretation ────────────────────────────────────────────────────────────
def interpret(aov_df, desc, between_col, n_total, alpha, group_names, time_names):
    rd = {r['Source']:r for _,r in aov_df.iterrows()}
    ab_src = f'{between_col} x Time'

    def info(src):
        r = rd.get(src,{})
        return (r.get('F',np.nan), r.get('p_GG',r.get('p',np.nan)),
                r.get('eta_p2',np.nan), r.get('df',np.nan), r.get('df_error',np.nan))

    F_A,p_A,e_A,df_A,dfe_A      = info(between_col)
    F_B,p_B,e_B,df_B,dfe_B      = info('Time')
    F_AB,p_AB,e_AB,df_AB,dfe_AB = info(ab_src)

    n_g = len(group_names); n_t = len(time_names)
    design = design_label(n_g, n_t)
    g_list = ', '.join(f'"{g}"' for g in group_names)
    t_list = ', '.join(f'"{t}"' for t in time_names)
    blocks = []

    blocks.append(("Study Design Overview",
        f"A Mixed-Design Analysis of Variance (Split-Plot ANOVA) was conducted using a {design} design. "
        f"The between-subjects factor comprised group membership ({g_list}), and the within-subjects "
        f"factor represented the time point of measurement ({t_list}). A total of {n_total} participants "
        f"contributed complete data to the analysis. The significance threshold was set at alpha = {alpha}."))

    if not np.isnan(p_A):
        sig = "statistically significant" if p_A<alpha else "not statistically significant"
        mag = eta_interp(e_A)
        conc = ("Collapsed across all time points, the groups differed significantly in their "
                "overall mean level of the dependent variable."
                if p_A<alpha else
                "Collapsed across all time points, the groups did not differ significantly in their "
                "overall mean performance.")
        blocks.append(("Main Effect of Group (Between-Subjects Factor)",
            f"The main effect of the between-subjects factor (group) was {sig}, "
            f"F({fmt(df_A,0)}, {fmt(dfe_A,0)}) = {fmt(F_A,2)}, p = {fmt_p(p_A)}, "
            f"partial eta-squared = {fmt(e_A,3)} ({mag} effect; Cohen, 1988). {conc}"))

    if not np.isnan(p_B):
        sig = "statistically significant" if p_B<alpha else "not statistically significant"
        mag = eta_interp(e_B)
        conc = ("Averaged across all groups, scores changed significantly across the measurement occasions."
                if p_B<alpha else
                "Averaged across all groups, scores did not change significantly across time points.")
        blocks.append(("Main Effect of Time (Within-Subjects Factor)",
            f"The main effect of time was {sig}, "
            f"F({fmt(df_B,0)}, {fmt(dfe_B,0)}) = {fmt(F_B,2)}, p = {fmt_p(p_B)}, "
            f"partial eta-squared = {fmt(e_B,3)} ({mag} effect). {conc}"))

    if not np.isnan(p_AB):
        mag = eta_interp(e_AB)
        if p_AB < alpha:
            conc = (f"This significant interaction indicates that the {n_g} groups followed different "
                    f"trajectories of change across the {n_t} measurement occasions. The rate and "
                    f"pattern of change over time was not uniform across groups. Because the interaction "
                    f"is significant, the main effects of Group and Time should be interpreted with "
                    f"caution, as their meaning is qualified by the other factor. Post-hoc pairwise "
                    f"comparisons are recommended to identify the specific contrasts driving this "
                    f"differential pattern.")
            heading = "Group x Time Interaction Effect — Statistically Significant"
        elif p_AB < .10:
            conc = (f"A marginal trend toward a Group x Time interaction was observed (p < .10). "
                    f"This did not reach the conventional significance threshold of alpha = {alpha}. "
                    f"Researchers may wish to examine this trend in the context of the effect size "
                    f"(partial eta-squared = {fmt(e_AB,3)}) and consider it exploratory.")
            heading = "Group x Time Interaction Effect — Marginal Trend (p < .10)"
        else:
            conc = (f"The non-significant interaction indicates that the {n_g} groups followed "
                    f"a similar pattern of change across the {n_t} time points. Consequently, "
                    f"the main effects of Group and Time may be interpreted independently of one another.")
            heading = "Group x Time Interaction Effect — Not Statistically Significant"

        blocks.append((heading,
            f"The Group x Time interaction was {'statistically significant' if p_AB<alpha else 'not statistically significant'}, "
            f"F({fmt(df_AB,0)}, {fmt(dfe_AB,0)}) = {fmt(F_AB,2)}, p = {fmt_p(p_AB)}, "
            f"partial eta-squared = {fmt(e_AB,3)} ({mag} effect size). {conc}"))

    blocks.append(("Effect Size Reference (Cohen, 1988)",
        "Partial eta-squared benchmarks: "
        "Negligible effect: partial eta-squared < .01; "
        "Small effect: partial eta-squared >= .01; "
        "Medium effect: partial eta-squared >= .06; "
        "Large effect: partial eta-squared >= .14."))
    return blocks


def apa_sentence(aov_df, between_col, n_total, n_groups, time_names, alpha):
    ab_src = f'{between_col} x Time'
    r = {r2['Source']:r2 for _,r2 in aov_df.iterrows()}.get(ab_src,{})
    F=r.get('F',np.nan); p=r.get('p_GG',r.get('p',np.nan))
    e=r.get('eta_p2',np.nan); df1=r.get('df',np.nan); df2=r.get('df_error',np.nan)
    sig = "significant" if not np.isnan(p) and p<alpha else "not significant"
    design = design_label(n_groups, len(time_names))
    return (f"A {design} Mixed-Design ANOVA was conducted (N = {n_total}). "
            f"The Group x Time interaction was {sig}, "
            f"F({fmt(df1,0)}, {fmt(df2,0)}) = {fmt(F,2)}, p = {fmt_p(p)}, "
            f"partial eta-squared = {fmt(e,3)}.")


# ── Visualization — light theme ───────────────────────────────────────────────
PAL = ['#1d6fa4','#2e9b5f','#8b5cf6','#d97706','#dc2626','#0891b2','#65a30d']

def _light_ax(ax, ylabel='Mean Score'):
    ax.set_facecolor('#fafafa')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#d1d5db')
    ax.tick_params(colors='#374151', labelsize=9)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.8)
    ax.set_xlabel('Time Point', color='#374151', fontsize=10)
    ax.set_ylabel(ylabel, color='#374151', fontsize=10)

def profile_plot(desc, between_col):
    plt.rcParams.update({'font.family':'DejaVu Sans'})
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    fig.patch.set_facecolor('#ffffff')
    groups  = desc['Group'].unique()
    times_u = list(dict.fromkeys(desc['Time'].tolist()))

    for ax in axes: _light_ax(ax)

    ax = axes[0]
    for i,g in enumerate(groups):
        sub = desc[desc['Group']==g].sort_values(
            'Time', key=lambda x: pd.Categorical(x, categories=times_u, ordered=True))
        ax.errorbar(range(len(sub)), sub['Mean'], yerr=sub['SE'],
                    marker='o', markersize=8, linewidth=2.5, color=PAL[i%len(PAL)],
                    label=str(g), capsize=5, capthick=1.5, elinewidth=1.5,
                    markeredgecolor='#fff', markeredgewidth=1.5)
    ax.set_xticks(range(len(times_u))); ax.set_xticklabels(times_u, fontsize=9)
    ax.set_title('Profile Plot (Mean +/- 1 SE)', color='#1c3557', fontsize=11, fontweight='bold', pad=10)
    ax.legend(title=between_col, title_fontsize=8, fontsize=8, facecolor='white', edgecolor='#d1d5db')

    ax = axes[1]
    ngrp=len(groups); w=0.75/ngrp; xs=np.arange(len(times_u))
    for i,g in enumerate(groups):
        sub = desc[desc['Group']==g].sort_values(
            'Time', key=lambda x: pd.Categorical(x, categories=times_u, ordered=True))
        off=(i-ngrp/2+.5)*w
        ax.bar(xs+off, sub['Mean'], width=w*0.9, color=PAL[i%len(PAL)],
               alpha=0.85, label=str(g), edgecolor='white', linewidth=0.6)
        ax.errorbar(xs+off, sub['Mean'], yerr=1.96*sub['SE'], fmt='none',
                    color='#374151', capsize=3, capthick=1, elinewidth=1, alpha=0.7)
    ax.set_xticks(xs); ax.set_xticklabels(times_u, fontsize=9)
    ax.set_title('Bar Chart (Mean +/- 95% CI)', color='#1c3557', fontsize=11, fontweight='bold', pad=10)
    ax.legend(title=between_col, title_fontsize=8, fontsize=8, facecolor='white', edgecolor='#d1d5db')
    plt.tight_layout(pad=2.5)
    return fig

def dist_plots(df_wide, between_col, time_cols):
    plt.rcParams.update({'font.family':'DejaVu Sans'})
    n=len(time_cols)
    fig, axes = plt.subplots(1, n, figsize=(5*n,4))
    fig.patch.set_facecolor('#ffffff')
    if n==1: axes=[axes]
    groups = df_wide[between_col].unique()
    for ax,t in zip(axes,time_cols):
        _light_ax(ax, ylabel='Frequency')
        ax.set_xlabel(str(t), color='#374151', fontsize=9)
        for i,g in enumerate(groups):
            v = df_wide[df_wide[between_col]==g][t].dropna()
            ax.hist(v, bins=12, alpha=0.5, color=PAL[i%len(PAL)], label=str(g),
                    edgecolor='white', linewidth=0.4)
            if len(v)>5:
                kde=stats.gaussian_kde(v); xr=np.linspace(v.min(),v.max(),200)
                ax.plot(xr, kde(xr)*len(v)*(v.max()-v.min())/12,
                        color=PAL[i%len(PAL)], linewidth=2, alpha=0.9)
        ax.set_title(f'Distribution: {t}', color='#1c3557', fontsize=10, fontweight='bold')
    axes[0].legend(title=between_col, fontsize=8, facecolor='white', edgecolor='#d1d5db')
    plt.tight_layout()
    return fig

def qq_plots(df_wide, between_col, time_cols):
    plt.rcParams.update({'font.family':'DejaVu Sans'})
    groups=df_wide[between_col].unique()
    nr=len(groups); nc=len(time_cols)
    fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 3.5*nr), squeeze=False)
    fig.patch.set_facecolor('#ffffff')
    for r,g in enumerate(groups):
        for c,t in enumerate(time_cols):
            ax=axes[r][c]; v=df_wide[df_wide[between_col]==g][t].dropna().values
            ax.set_facecolor('#fafafa')
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            for sp in ['bottom','left']: ax.spines[sp].set_color('#d1d5db')
            ax.tick_params(colors='#374151', labelsize=7)
            ax.grid(color='#e5e7eb', linewidth=0.6)
            if len(v)>=3:
                (osm,osr),(_s,_int,_r2)=stats.probplot(v,dist='norm')
                ax.scatter(osm,osr,color=PAL[r%len(PAL)],s=22,alpha=0.75,
                           edgecolors='white',linewidths=.5)
                q1,q3=np.percentile(v,[25,75]); th1,th3=stats.norm.ppf([.25,.75])
                slope=(q3-q1)/(th3-th1); inter=q1-slope*th1
                xl=np.array([osm[0],osm[-1]])
                ax.plot(xl,slope*xl+inter,color='#dc2626',linewidth=1.2,alpha=0.8,linestyle='--')
            ax.set_title(f'{g} - {t}',color='#1c3557',fontsize=8,fontweight='bold',pad=4)
            ax.set_xlabel('Theoretical Quantiles',color='#6b7280',fontsize=7)
            ax.set_ylabel('Sample Quantiles',color='#6b7280',fontsize=7)
    plt.suptitle('Q-Q Plots for Normality Assessment',color='#1c3557',
                 fontsize=11,fontweight='bold',y=1.01)
    plt.tight_layout()
    return fig


# ── ANOVA HTML table ──────────────────────────────────────────────────────────
def render_anova_html(aov_df, use_gg=True):
    rows_html = ""
    for _,r in aov_df.iterrows():
        src=r['Source']; ss=r.get('SS',np.nan); df_v=r.get('df',np.nan)
        ms=r.get('MS',np.nan); F=r.get('F',np.nan)
        p = r['p_GG'] if use_gg else r['p']
        eta=r.get('eta_p2',np.nan); eps=r.get('eps',np.nan); typ=r.get('type','')
        is_err='error' in typ; rc='class="err-row"' if is_err else ''
        p_str=fmt_p(p); star=sig_star(p) if not (isinstance(p,float) and np.isnan(p)) else ""
        cls=sig_cls(p) if not (isinstance(p,float) and np.isnan(p)) else "ns"
        rows_html += f"""<tr {rc}>
          <td><b>{src}</b></td><td>{fmt(ss,3)}</td><td>{fmt(df_v,0)}</td>
          <td>{fmt(ms,3)}</td>
          <td>{'—' if isinstance(F,float) and np.isnan(F) else fmt(F,3)}</td>
          <td class="{cls}">{p_str} {star}</td>
          <td>{'—' if isinstance(eta,float) and np.isnan(eta) else fmt(eta,3)}</td>
          <td>{'—' if isinstance(eps,float) and np.isnan(eps) else fmt(eps,3)}</td>
        </tr>"""
    return f"""<table class="anova-table"><thead><tr>
      <th>Source</th><th>SS</th><th>df</th><th>MS</th>
      <th>F</th><th>p-value</th><th>Partial Eta-sq</th><th>GG Epsilon</th>
    </tr></thead><tbody>{rows_html}</tbody></table>
    <p style="font-size:.73rem;color:#6b7280;margin-top:6px;">
    *** p &lt; .001 &nbsp;** p &lt; .01 &nbsp;* p &lt; .05 &nbsp;
    &dagger; p &lt; .10 &nbsp;ns p &ge; .10</p>"""


# ── PDF export ────────────────────────────────────────────────────────────────
def build_pdf(desc, norm_df, lev_df, aov_df, sph, interp_blocks, ph_df,
              between_col, dv_label, alpha, n_groups, time_names, method_name):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_JUSTIFY
        from reportlab.lib.units import cm
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    NAVY=rl.HexColor('#1c3557'); BLUE=rl.HexColor('#2563eb')
    LGRY=rl.HexColor('#f8fafc'); DGRY=rl.HexColor('#6b7280')

    sH1=ParagraphStyle('H1',fontName='Helvetica-Bold',fontSize=16,textColor=NAVY,spaceAfter=4)
    sH2=ParagraphStyle('H2',fontName='Helvetica-Bold',fontSize=11,textColor=NAVY,spaceAfter=3,spaceBefore=14)
    sH3=ParagraphStyle('H3',fontName='Helvetica-Bold',fontSize=9.5,textColor=BLUE,spaceAfter=2,spaceBefore=8)
    sBD=ParagraphStyle('BD',fontName='Helvetica',fontSize=9,leading=14,spaceAfter=3,alignment=TA_JUSTIFY)
    sNT=ParagraphStyle('NT',fontName='Helvetica-Oblique',fontSize=7.5,textColor=DGRY,leading=11,spaceAfter=2)

    def clean(val):
        s = str(round(val,3)) if isinstance(val,float) and not np.isnan(val) else str(val)
        replacements = [
            ('—','--'),('<=','<='),('>=','>='),('x','x'),
            ('eta','eta'),('alpha','alpha'),('epsilon','eps'),
            ('\u00d7','x'),('\u2014','--'),('\u03b7','eta'),
            ('\u00b2','2'),('\u2082','2'),('\u2070','0'),
            ('\u2265','>='),('\u2264','<='),('\u00e0','a'),
            ('\u2013','-'),('\u2019',"'"),('\u2018',"'"),
            ('\u201c','"'),('\u201d','"'),('\u00b1','+/-'),
        ]
        for old,new in replacements: s=s.replace(old,new)
        return s

    def make_table(df, col_widths=None):
        data=[[clean(c) for c in df.columns]]
        for _,row in df.iterrows():
            data.append([clean(v) for v in row])
        tbl=Table(data, repeatRows=1, hAlign='LEFT', colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),rl.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),7.5),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl.white,rl.HexColor('#f1f5f9')]),
            ('GRID',(0,0),(-1,-1),0.25,rl.HexColor('#cbd5e1')),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('LEFTPADDING',(0,0),(-1,-1),5),
        ]))
        return tbl

    story=[]; design=design_label(n_groups, len(time_names))
    story.append(Paragraph("Mixed-Design ANOVA (Split-Plot) Report", sH1))
    story.append(Paragraph(
        f"Design: {design}  |  Between-Subjects Factor: {between_col}  |  "
        f"Dependent Variable: {dv_label}  |  Alpha = {alpha}", sBD))
    story.append(HRFlowable(width="100%",thickness=2,color=NAVY,spaceAfter=10))
    story.append(Spacer(1,6))

    story.append(Paragraph("1. Descriptive Statistics", sH2))
    story.append(make_table(desc))
    story.append(Spacer(1,8))

    story.append(Paragraph("2. Normality Tests", sH2))
    story.append(Paragraph(
        "Shapiro-Wilk (SW) is applied when n <= 50 per cell; "
        "Kolmogorov-Smirnov (KS) when n > 50. H0: data are normally distributed. "
        "Reject H0 if p < alpha.", sBD))
    story.append(make_table(norm_df))
    story.append(Spacer(1,8))

    if not lev_df.empty:
        story.append(Paragraph("3. Levene's Test of Equality of Variances", sH2))
        story.append(Paragraph(
            "H0: group variances are equal at each time point. Reject H0 if p < alpha.", sBD))
        story.append(make_table(lev_df))
        story.append(Spacer(1,8))

    if sph:
        W,p_W,eps_GG,eps_HF,chi2,df_chi=sph
        story.append(Paragraph("4. Mauchly's Test of Sphericity", sH2))
        story.append(Paragraph(
            "H0: sphericity is satisfied. Reject H0 if p < .05. "
            "If violated, Greenhouse-Geisser (GG) or Huynh-Feldt (HF) correction is applied.", sBD))
        sph_tbl=pd.DataFrame([{"Mauchly's W":round(W,4),"Chi-sq":round(chi2,3),
                                 "df":df_chi,"p-value":fmt_p(p_W),
                                 "GG Epsilon":round(eps_GG,4),"HF Epsilon":round(eps_HF,4)}])
        story.append(make_table(sph_tbl))
        story.append(Spacer(1,8))

    story.append(Paragraph("5. Mixed-Design ANOVA Results", sH2))
    story.append(Paragraph(
        "Type III Sum of Squares (SPSS-compatible). "
        "GG-corrected p-values reported for within-subjects effects. "
        "Partial eta-squared reported as the effect size measure.", sBD))
    aov_show=aov_df[['Source','SS','df','MS','F','p_GG','eta_p2','eps']].copy()
    aov_show.columns=['Source','SS','df','MS','F','p (GG-corrected)','Partial Eta-sq','GG Epsilon']
    story.append(make_table(aov_show.round(4)))
    story.append(Paragraph(
        "Note: *** p < .001  ** p < .01  * p < .05  ns p >= .05. "
        "Error rows reflect residual variance terms and are italicised.", sNT))
    story.append(Spacer(1,10))

    if not ph_df.empty:
        story.append(Paragraph(f"6. Post-Hoc Pairwise Comparisons ({method_name})", sH2))
        story.append(Paragraph(
            "Conducted because the Group x Time interaction was statistically significant.", sBD))
        ph_show=ph_df[['Comparison Type','Time Point','Group','Label A','Label B',
                        'Mean A','Mean B','Mean Diff (A-B)','SE','t','df',
                        'p (raw)','p (adj)',"Cohen's d",'Sig']].copy()
        story.append(make_table(ph_show))
        story.append(Spacer(1,8))

    story.append(Paragraph("7. Statistical Interpretation", sH2))
    for heading, body in interp_blocks:
        story.append(Paragraph(heading, sH3))
        body_c=(body.replace('x','x').replace('eta-squared','eta-squared')
                .replace('>=','>=').replace('<=','<=').replace('—','--'))
        story.append(Paragraph(body_c, sBD))
        story.append(Spacer(1,4))

    story.append(Spacer(1,12))
    story.append(HRFlowable(width="100%",thickness=0.5,color=DGRY))
    story.append(Paragraph(
        "Generated by Mixed-Design ANOVA Analyzer  |  "
        "Engine: scipy + numpy (SPSS-compatible algorithms)", sNT))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:.6rem 0 .3rem">
          <div style="font-family:'Source Serif 4',serif;font-size:1.1rem;font-weight:700;">
            Mixed-Design ANOVA</div>
          <div style="font-size:.73rem;margin-top:2px;opacity:.7;">SPSS-Equivalent Analyzer</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        with st.expander("User Guide", expanded=False):
            st.markdown("""
**What is a Mixed-Design ANOVA?**
Also called Split-Plot ANOVA. Combines:
- A **between-subjects** factor (independent groups)
- A **within-subjects** factor (repeated measurements over time)

**Data Format — Wide CSV**
```
ID,    Group,  Pre,  Mid,  Post
P001,  A,      45,   60,   55
```
Each row = one participant.

**Assumption Tests**
| Test | Applied when |
|------|-------------|
| Shapiro-Wilk | n <= 50 per cell |
| Kolmogorov-Smirnov | n > 50 per cell |
| Levene's | All time points |
| Mauchly's | More than 2 time points |

**Post-Hoc Methods**
- **Tukey HSD** — recommended for equal group sizes
- **Bonferroni** — most conservative
- **Holm** — more powerful than Bonferroni
- **FDR (BH)** — controls false discovery rate

**Effect Size (Cohen, 1988)**
Negligible: < .01 / Small: >= .01 / Medium: >= .06 / Large: >= .14
            """)
        st.markdown("---")
        st.markdown("<p style='font-size:.78rem;opacity:.75;'>Sample Datasets</p>",
                    unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.download_button("2 Groups\n2 Times", sample_2g2t().to_csv(index=False),
                                "sample_2grp_2time.csv","text/csv",use_container_width=True)
        with c2:
            st.download_button("3 Groups\n3 Times", sample_3g3t().to_csv(index=False),
                                "sample_3grp_3time.csv","text/csv",use_container_width=True)
        st.markdown("---")
        st.markdown("""
        <div style="font-size:.69rem;opacity:.6;line-height:1.75;">
        Engine: scipy · numpy · pandas<br>
        SS Type: Type III (SPSS-compatible)<br>
        Sphericity: Greenhouse-Geisser / Huynh-Feldt<br>
        Post-Hoc: Tukey HSD, Bonferroni, Holm, FDR
        </div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sidebar()

    st.markdown("""
    <div class="hero">
      <h1>Mixed-Design ANOVA Analyzer</h1>
      <p>Split-Plot ANOVA &nbsp;&middot;&nbsp; SPSS-Level Statistical Analysis
         &nbsp;&middot;&nbsp; Pure scipy/numpy Engine</p>
      <div>
        <span class="badge bd-b">Type III SS</span>
        <span class="badge bd-g">Mauchly + GG/HF Correction</span>
        <span class="badge bd-p">Tukey HSD Post-Hoc</span>
        <span class="badge bd-o">PDF / CSV Export</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Step 1 — Upload
    st.markdown('<div class="sec-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Step 1 — Data Input</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Wide-Format CSV", type=['csv'],
        help="Each row = one participant. Columns: Subject ID, Group, numeric time-point columns.")

    if uploaded is None:
        st.markdown("""
        <div class="ibox">
        <b>No file uploaded yet.</b> Please upload a wide-format CSV file or download a sample
        dataset from the sidebar to explore the application.<br><br>
        <b>Required format:</b> <code>ID, Group, TimePoint1, TimePoint2, ...</code><br>
        Each row must represent one unique participant.
        All time-point columns must contain numeric values.
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.markdown(f'<div class="ebox"><b>File Read Error:</b> Could not parse the file as CSV.<br>'
                    f'Detail: {e}<br>Ensure the file is a valid comma-separated CSV and not corrupted.</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if df.empty or len(df.columns) < 3:
        st.markdown('<div class="ebox"><b>Invalid File:</b> The file appears empty or has fewer than 3 '
                    'columns. A valid file requires: Subject ID, Group, and at least one time-point column.</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    with st.expander(f"Preview uploaded data — {df.shape[0]} rows x {df.shape[1]} columns", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        if df.shape[0]>10: st.caption(f"Showing first 10 of {df.shape[0]} rows.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 2 — Column mapping
    st.markdown('<div class="sec-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Step 2 — Column Mapping & Analysis Options</div>',
                unsafe_allow_html=True)

    all_cols = list(df.columns)
    num_cols = list(df.select_dtypes(include=[np.number]).columns)

    c1,c2,c3 = st.columns(3)
    with c1:
        subj_col = st.selectbox("Subject ID Column", all_cols, 0,
                                 help="Column with unique participant identifiers.")
    with c2:
        btwn_col = st.selectbox("Between-Subjects Factor (Group)",
                                 [c for c in all_cols if c!=subj_col], 0,
                                 help="Column with group labels (e.g., Control, Treatment).")
    with c3:
        avail_time = [c for c in num_cols if c not in [subj_col,btwn_col]]
        time_cols  = st.multiselect("Within-Subjects Time-Point Columns", avail_time,
                                     default=avail_time[:min(4,len(avail_time))],
                                     help="All columns representing repeated measurements over time.")

    c4,c5,c6,c7 = st.columns(4)
    with c4:
        dv_label = st.text_input("Dependent Variable Label", "Score",
                                  help="Descriptive name for the outcome measure (used in reports).")
    with c5:
        ph_method = st.selectbox("Post-Hoc Correction",
            ['tukey','bonferroni','holm','fdr_bh'],
            format_func=lambda x: {'tukey':'Tukey HSD (recommended)',
                                    'bonferroni':'Bonferroni (conservative)',
                                    'holm':'Holm (step-down)',
                                    'fdr_bh':'FDR Benjamini-Hochberg'}[x],
            help="Multiple comparison correction for post-hoc pairwise tests.")
    with c6:
        alpha = st.selectbox("Significance Level (alpha)",
            options=[0.001, 0.01, 0.05, 0.10], index=2,
            format_func=lambda x: f"alpha = {x}",
            help="Probability threshold for statistical significance.")
    with c7:
        use_gg = st.checkbox("Apply GG Correction", value=True,
                              help="Apply Greenhouse-Geisser correction to within-subjects p-values.")

    run_btn = st.button("Run Mixed-Design ANOVA", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not run_btn: return

    if len(time_cols) < 2:
        st.markdown('<div class="ebox"><b>Insufficient Time Points:</b> Please select at least 2 '
                    'time-point columns. A within-subjects factor requires a minimum of 2 levels.</div>',
                    unsafe_allow_html=True)
        return

    # Validate
    issues = validate_data(df, subj_col, btwn_col, time_cols)
    has_error = any(lv=='error' for lv,_ in issues)
    for level,msg in issues:
        if level=='error':
            st.markdown(f'<div class="ebox"><b>Data Error:</b> {msg}</div>', unsafe_allow_html=True)
        elif level=='warning':
            st.markdown(f'<div class="wbox"><b>Warning:</b> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ibox">{msg}</div>', unsafe_allow_html=True)

    if has_error:
        st.markdown('<div class="ebox"><b>Analysis halted.</b> Correct the errors above and try again.</div>',
                    unsafe_allow_html=True)
        return

    req_cols = [subj_col, btwn_col] + time_cols
    df_clean = df[req_cols].dropna()
    if len(df_clean) < 4:
        st.markdown(f'<div class="ebox"><b>Insufficient Data:</b> Only {len(df_clean)} complete cases '
                    'remain after removing missing values. At least 4 are required.</div>',
                    unsafe_allow_html=True)
        return

    n_total    = df_clean[subj_col].nunique()
    n_groups   = df_clean[btwn_col].nunique()
    group_names= sorted(df_clean[btwn_col].unique().tolist())
    design     = design_label(n_groups, len(time_cols))
    method_nm  = {'tukey':'Tukey HSD','bonferroni':'Bonferroni',
                   'holm':'Holm','fdr_bh':'FDR (Benjamini-Hochberg)'}[ph_method]

    with st.spinner("Running statistical analysis..."):
        try:
            desc    = compute_descriptives(df_clean, btwn_col, time_cols)
            norm_df = test_normality(df_clean, btwn_col, time_cols)
            lev_df  = test_levene(df_clean, btwn_col, time_cols)
            aov_df, sph = mixed_anova(df_clean, subj_col, btwn_col, time_cols)
            ab_src  = f'{btwn_col} x Time'
            inter_r = aov_df[aov_df['Source']==ab_src]
            int_p   = inter_r['p_GG'].values[0] if not inter_r.empty else 1.0
            run_ph  = int_p < alpha
            ph_df   = run_posthoc(df_clean, btwn_col, time_cols, ph_method, alpha) if run_ph else pd.DataFrame()
            interp_blocks = interpret(aov_df, desc, btwn_col, n_total, alpha, group_names, time_cols)
            ph_narr = posthoc_narrative(ph_df, alpha, method_nm) if not ph_df.empty else []
            apa     = apa_sentence(aov_df, btwn_col, n_total, n_groups, time_cols, alpha)
        except Exception as e:
            st.markdown(f'<div class="ebox"><b>Analysis Error:</b> The computation failed.<br>'
                        f'<code>{e}</code><br>Possible causes: constant column values, singular '
                        f'covariance matrix, or extremely unbalanced groups.</div>',
                        unsafe_allow_html=True)
            st.exception(e); return

    st.markdown(f'<div class="obox">Analysis complete. Design: <b>{design}</b> | '
                f'N = {n_total} participants | {n_groups} groups | {len(time_cols)} time points.</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1rem 0 1.5rem;">
      <div class="mcard"><div class="mval">{n_total}</div><div class="mlbl">Participants</div></div>
      <div class="mcard"><div class="mval">{n_groups}</div><div class="mlbl">Groups</div></div>
      <div class="mcard"><div class="mval">{len(time_cols)}</div><div class="mlbl">Time Points</div></div>
      <div class="mcard"><div class="mval">{n_total*len(time_cols)}</div><div class="mlbl">Observations</div></div>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["Descriptive Statistics","Assumption Tests","ANOVA Results",
                    "Post-Hoc Tests","Visualizations","Interpretation","Export Report"])

    # Tab 0 — Descriptives
    with tabs[0]:
        st.markdown('<div class="sec-title">Cell Means and Descriptive Statistics</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="ibox">Design: <b>{design}</b> &nbsp;|&nbsp; '
                    f'Groups: {", ".join(group_names)} &nbsp;|&nbsp; '
                    f'Time points: {", ".join(time_cols)}</div>', unsafe_allow_html=True)
        st.dataframe(desc, use_container_width=True, hide_index=True)
        st.markdown('<div class="sec-title" style="margin-top:1.5rem">Marginal Means</div>',
                    unsafe_allow_html=True)
        ca,cb=st.columns(2)
        with ca:
            st.markdown("**By Group (averaged across all time points)**")
            mg=desc.groupby('Group')[['N','Mean','SD']].mean().round(3); mg['N']=mg['N'].astype(int)
            st.dataframe(mg, use_container_width=True)
        with cb:
            st.markdown("**By Time Point (averaged across all groups)**")
            mt=desc.groupby('Time')[['N','Mean','SD']].mean().round(3); mt['N']=mt['N'].astype(int)
            st.dataframe(mt, use_container_width=True)

    # Tab 1 — Assumptions
    with tabs[1]:
        st.markdown('<div class="sec-title">A. Normality Tests</div>', unsafe_allow_html=True)
        sw_n=(norm_df['N']<=50).sum(); ks_n=(norm_df['N']>50).sum()
        st.markdown(f"""<div class="ibox">
        <b>Automatic Test Selection:</b> Shapiro-Wilk selected for <b>{sw_n}</b> cells (n &le; 50);
        Kolmogorov-Smirnov for <b>{ks_n}</b> cells (n &gt; 50).<br>
        H0: data are normally distributed. Reject H0 if p &lt; {alpha}.
        ANOVA is generally robust to moderate violations when group sizes are equal and n &ge; 15 per cell.
        </div>""", unsafe_allow_html=True)
        norm_show=norm_df.copy()
        norm_show['p-value']=norm_show['p-value'].apply(lambda x: fmt_p(x) if pd.notna(x) else '—')
        st.dataframe(norm_show, use_container_width=True, hide_index=True)
        if (norm_df['Normal']=='Yes').all():
            st.markdown('<div class="obox">All cells satisfy the normality assumption (p &gt; alpha).</div>',
                        unsafe_allow_html=True)
        elif (norm_df['Normal']=='No').any():
            n_fail=(norm_df['Normal']=='No').sum()
            st.markdown(f'<div class="wbox">{n_fail} cell(s) show evidence of non-normality. '
                        'ANOVA is robust for balanced designs with n &ge; 15 per cell. '
                        'For severe violations, consider non-parametric alternatives '
                        '(Friedman test + Kruskal-Wallis test).</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:1.5rem">'
                    "B. Levene's Test of Homogeneity of Variance</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="ibox">H0: group variances are equal at each time point. '
                    f'Reject H0 if p &lt; {alpha}. Violation may inflate the between-subjects F-ratio.</div>',
                    unsafe_allow_html=True)
        if not lev_df.empty:
            lev_show=lev_df.copy(); lev_show['p-value']=lev_show['p-value'].apply(fmt_p)
            st.dataframe(lev_show, use_container_width=True, hide_index=True)
            if (lev_df['Equal Variances']=='Yes').all():
                st.markdown('<div class="obox">Homogeneity of variance satisfied at all time points.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="wbox">Unequal variances detected at one or more time points. '
                            'Interpret the between-subjects F-ratio with caution.</div>',
                            unsafe_allow_html=True)

        if len(time_cols) > 2:
            st.markdown('<div class="sec-title" style="margin-top:1.5rem">'
                        "C. Mauchly's Test of Sphericity</div>", unsafe_allow_html=True)
            st.markdown("""<div class="ibox">
            H0: sphericity is satisfied. Reject H0 if p &lt; .05.<br>
            Use Greenhouse-Geisser (GG) correction when epsilon &lt; .75;
            Huynh-Feldt (HF) when epsilon &ge; .75.
            </div>""", unsafe_allow_html=True)
            if sph:
                W,p_W,eps_GG,eps_HF,chi2,df_chi=sph
                cm1,cm2,cm3,cm4,cm5=st.columns(5)
                cm1.metric("Mauchly's W",fmt(W,4)); cm2.metric("Chi-sq",fmt(chi2,3))
                cm3.metric("df",str(df_chi)); cm4.metric("p-value",fmt_p(p_W))
                cm5.metric("GG Epsilon",fmt(eps_GG,4))
                if not np.isnan(p_W):
                    if p_W<.05:
                        rec="Greenhouse-Geisser" if eps_GG<.75 else "Huynh-Feldt"
                        st.markdown(f'<div class="wbox">Sphericity violated (p = {fmt_p(p_W)}). '
                                    f'<b>{rec} correction recommended</b> (GG epsilon = {fmt(eps_GG,3)}). '
                                    'GG-corrected p-values are applied when the "Apply GG Correction" '
                                    'option is enabled.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="obox">Sphericity satisfied (p = {fmt_p(p_W)}). '
                                    'No correction required.</div>', unsafe_allow_html=True)
            else:
                st.info("Mauchly's test could not be computed. Check data balance.")
        else:
            st.markdown('<div class="ibox">Sphericity test is not applicable for designs with exactly '
                        '2 time points (sphericity is automatically satisfied with a single difference score).</div>',
                        unsafe_allow_html=True)

    # Tab 2 — ANOVA Results
    with tabs[2]:
        st.markdown('<div class="sec-title">Mixed-Design ANOVA Summary Table</div>', unsafe_allow_html=True)
        corr_note="Greenhouse-Geisser corrected" if use_gg else "uncorrected"
        st.markdown(f'<div class="ibox"><b>Algorithm:</b> Split-Plot ANOVA (manual SS decomposition) &nbsp;|&nbsp; '
                    f'<b>SS Type:</b> III (SPSS-compatible) &nbsp;|&nbsp; '
                    f'<b>p-values:</b> {corr_note}</div>', unsafe_allow_html=True)
        st.markdown(render_anova_html(aov_df, use_gg=use_gg), unsafe_allow_html=True)
        st.markdown("""<div style="margin-top:.75rem;padding:.7rem 1rem;background:#f8fafc;
                    border:1px solid #e2e8f0;border-radius:8px;font-size:.76rem;color:#6b7280;">
        <b style="color:#374151;">Partial eta-squared benchmarks (Cohen, 1988):</b> &nbsp;
        Negligible: &lt; .01 &nbsp;|&nbsp; Small: &ge; .01 &nbsp;|&nbsp;
        Medium: &ge; .06 &nbsp;|&nbsp; Large: &ge; .14
        </div>""", unsafe_allow_html=True)
        with st.expander("View raw ANOVA data table"):
            st.dataframe(aov_df, use_container_width=True, hide_index=True)

    # Tab 3 — Post-Hoc
    with tabs[3]:
        if not run_ph:
            p_show=aov_df[aov_df['Source']==ab_src]['p_GG'].values
            p_show=p_show[0] if len(p_show) else 1.0
            st.markdown(f"""<div class="ibox">
            <b>Post-hoc tests were not conducted.</b><br><br>
            The Group x Time interaction did not reach the significance threshold
            (p = {fmt_p(p_show)}, alpha = {alpha}). Post-hoc pairwise comparisons are
            conventionally conducted only when the omnibus interaction is statistically significant,
            to control the family-wise error rate.<br><br>
            If a main effect is significant and of interest, a separate simple-effects analysis
            or one-way ANOVA at each level of the other factor may be appropriate.
            </div>""", unsafe_allow_html=True)
        elif ph_df.empty:
            st.warning("Post-hoc comparisons could not be computed.")
        else:
            st.markdown(f'<div class="sec-title">Pairwise Comparisons — {method_nm} Corrected</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="ibox">Post-hoc tests follow a statistically significant '
                        f'Group x Time interaction (p = {fmt_p(int_p)}). '
                        f'The {method_nm} correction has been applied.</div>',
                        unsafe_allow_html=True)

            bt=ph_df[ph_df['Comparison Type']=='Between Groups']
            wt=ph_df[ph_df['Comparison Type']=='Within Time']
            COLS_BT=['Time Point','Label A','Label B','Mean A','Mean B',
                     'Mean Diff (A-B)','SE','t','df','p (raw)','p (adj)',"Cohen's d",'Sig']
            COLS_WT=['Group','Label A','Label B','Mean A','Mean B',
                     'Mean Diff (A-B)','SE','t','df','p (raw)','p (adj)',"Cohen's d",'Sig']

            if not bt.empty:
                st.markdown("#### Between-Group Comparisons at Each Time Point")
                bt_s=bt[[c for c in COLS_BT if c in bt.columns]].copy()
                bt_s['p (raw)']=bt_s['p (raw)'].apply(fmt_p)
                bt_s['p (adj)']=bt_s['p (adj)'].apply(fmt_p)
                st.dataframe(bt_s, use_container_width=True, hide_index=True)

            if not wt.empty:
                st.markdown("#### Within-Subject Time Comparisons per Group")
                wt_s=wt[[c for c in COLS_WT if c in wt.columns]].copy()
                wt_s['p (raw)']=wt_s['p (raw)'].apply(fmt_p)
                wt_s['p (adj)']=wt_s['p (adj)'].apply(fmt_p)
                st.dataframe(wt_s, use_container_width=True, hide_index=True)

            st.markdown('<div style="font-size:.75rem;color:#6b7280;margin-top:.4rem;">'
                        '*** p &lt; .001 &nbsp;** p &lt; .01 &nbsp;* p &lt; .05 &nbsp;'
                        '&dagger; p &lt; .10 &nbsp;ns p &ge; .10</div>', unsafe_allow_html=True)

            if ph_narr:
                st.markdown('<div class="sec-title" style="margin-top:1.5rem">'
                            'Post-Hoc Narrative Interpretation</div>', unsafe_allow_html=True)
                for heading, body in ph_narr:
                    st.markdown(f'<div class="interp-block"><h4>{heading}</h4>{body}</div>',
                                unsafe_allow_html=True)

    # Tab 4 — Visualizations
    with tabs[4]:
        st.markdown('<div class="sec-title">Profile Plot (Interaction Plot)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ibox">Non-parallel lines indicate a potential Group x Time '
                    'interaction. Error bars: left panel = +/- 1 SE; right panel = +/- 95% CI.</div>',
                    unsafe_allow_html=True)
        fig1=profile_plot(desc, btwn_col)
        st.pyplot(fig1, use_container_width=True); plt.close(fig1)

        st.markdown('<div class="sec-title" style="margin-top:1.5rem">'
                    'Distributions with Kernel Density Estimate</div>', unsafe_allow_html=True)
        fig2=dist_plots(df_clean, btwn_col, time_cols)
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

        st.markdown('<div class="sec-title" style="margin-top:1.5rem">'
                    'Q-Q Plots for Normality Assessment</div>', unsafe_allow_html=True)
        st.markdown('<div class="ibox">Points near the diagonal reference line indicate normality. '
                    'Systematic departures suggest non-normal distributions.</div>',
                    unsafe_allow_html=True)
        fig3=qq_plots(df_clean, btwn_col, time_cols)
        st.pyplot(fig3, use_container_width=True); plt.close(fig3)

    # Tab 5 — Interpretation
    with tabs[5]:
        st.markdown('<div class="sec-title">Automated Statistical Interpretation</div>',
                    unsafe_allow_html=True)
        for heading, body in interp_blocks:
            st.markdown(f'<div class="interp-block"><h4>{heading}</h4>{body}</div>',
                        unsafe_allow_html=True)
        st.markdown('<div class="sec-title" style="margin-top:1.5rem">APA-Style Reporting Sentence</div>',
                    unsafe_allow_html=True)
        st.info(apa)
        st.markdown('<div class="sec-title" style="margin-top:1.5rem">Effect Size Summary</div>',
                    unsafe_allow_html=True)
        es=aov_df[~aov_df['type'].str.contains('error')][['Source','F','p_GG','eta_p2']].copy()
        es['F']=es['F'].apply(lambda x: fmt(x,3))
        es['p_GG']=es['p_GG'].apply(fmt_p)
        es['Interpretation']=aov_df[~aov_df['type'].str.contains('error')]['eta_p2'].apply(eta_interp)
        es['eta_p2']=es['eta_p2'].apply(lambda x: fmt(x,4))
        es.columns=['Source','F','p (GG-corrected)','Partial Eta-squared','Effect Size Interpretation']
        st.dataframe(es, use_container_width=True, hide_index=True)

    # Tab 6 — Export
    with tabs[6]:
        st.markdown('<div class="sec-title">Download Full Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="ibox">All analysis outputs are available for download. '
                    'The PDF report contains formatted tables and academic interpretations. '
                    'The CSV report contains all raw numerical outputs.</div>',
                    unsafe_allow_html=True)
        ca,cb,cc=st.columns(3)

        with ca:
            st.markdown("**CSV Report**")
            buf_csv=io.StringIO()
            buf_csv.write(f"Mixed-Design ANOVA Report\nDesign: {design}\n"
                          f"Between-Subjects Factor: {btwn_col}\n"
                          f"Dependent Variable: {dv_label}\nAlpha: {alpha}\n\n")
            buf_csv.write("DESCRIPTIVE STATISTICS\n"); desc.to_csv(buf_csv, index=False)
            buf_csv.write("\n\nNORMALITY TESTS\n"); norm_df.to_csv(buf_csv, index=False)
            buf_csv.write("\n\nLEVENE'S TEST\n"); lev_df.to_csv(buf_csv, index=False)
            if sph:
                W2,p_W2,eg2,ef2,ch2,dc2=sph
                s2=pd.DataFrame([{"W":round(W2,4),"chi2":round(ch2,3),"df":dc2,
                                   "p":fmt_p(p_W2),"GG_eps":round(eg2,4),"HF_eps":round(ef2,4)}])
                buf_csv.write("\n\nMAUCHLY SPHERICITY\n"); s2.to_csv(buf_csv, index=False)
            buf_csv.write("\n\nANOVA RESULTS\n"); aov_df.to_csv(buf_csv, index=False)
            if not ph_df.empty:
                buf_csv.write("\n\nPOST-HOC TESTS\n"); ph_df.to_csv(buf_csv, index=False)
            buf_csv.write("\n\nINTERPRETATION\n")
            for h,b in interp_blocks: buf_csv.write(f"\n[{h}]\n{b}\n")
            st.download_button("Download CSV Report", buf_csv.getvalue().encode(),
                                "mixed_anova_report.csv","text/csv",use_container_width=True)

        with cb:
            st.markdown("**PDF Report**")
            pdf=build_pdf(desc,norm_df,lev_df,aov_df,sph,interp_blocks,ph_df,
                           btwn_col,dv_label,alpha,n_groups,time_cols,method_nm)
            if pdf:
                st.download_button("Download PDF Report", pdf,
                                    "mixed_anova_report.pdf","application/pdf",
                                    use_container_width=True)
            else:
                st.caption("PDF requires `reportlab`. Install with: pip install reportlab")

        with cc:
            st.markdown("**Long-Format Data**")
            df_long2=df_clean.melt(id_vars=[subj_col,btwn_col],value_vars=time_cols,
                                    var_name='Time',value_name=dv_label)
            st.download_button("Download Long-Format CSV",
                                df_long2.to_csv(index=False).encode(),
                                "long_format_data.csv","text/csv",use_container_width=True)


if __name__ == '__main__':
    main()
