import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ======================================================================
# FULL ACADEMIC ITEM ANALYSIS - PARAMETER & DECISION EDITION
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📈", layout="wide")

# Brutalist Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border: 2px solid #000; padding: 10px; background: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 FULL-METRIC ITEM ANALYSIS (CTT)")
st.write("Comprehensive evaluation with explicit decision parameters and descriptive interpretations.")

# SIDEBAR: PARAMETERS & LEGEND
with st.sidebar:
    st.header("⚙️ Analysis Parameters")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27)
    validity_limit = st.number_input("Minimum r_pbis", 0.0, 1.0, 0.25)
    
    st.markdown("---")
    st.header("📖 Interpretation Legend")
    st.info("""
    **Difficulty (P):**
    - > 0.70: Easy 🟢
    - 0.30 - 0.70: Moderate 🟡
    - < 0.30: Difficult 🔴
    
    **Discrimination (D):**
    - >= 0.40: Excellent 🟢
    - 0.20 - 0.39: Satisfactory 🟡
    - < 0.20: Poor/Reject 🔴
    
    **Validity (r_pbis):**
    - >= Threshold: Valid 🟢
    - < Threshold: Invalid 🔴
    """)

# FILE UPLOADER
u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Responses (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Answer Key (CSV)", type=['csv'])

if student_file and key_file:
    # 1. LOAD DATA
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
    # 2. SCORING
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students = len(df)
    n_items = len(item_cols)

    # 3. GROUPING
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    up_idx, lo_idx = df_sorted.head(n_group).index, df_sorted.tail(n_group).index

    # 4. CALCULATION & DECISION LOGIC
    results = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        p_up, p_lo = df_scores.loc[up_idx, item].mean(), df_scores.loc[lo_idx, item].mean()
        d = p_up - p_lo
        r_pb, _ = pointbiserialr(df_scores[item], total_scores) if df_scores[item].var() != 0 else (0,0)

        # Interpretations
        p_desc = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_desc = "Excellent" if d >= 0.4 else "Fair" if d >= 0.2 else "Poor"
        v_desc = "Valid" if r_pb >= validity_limit else "Invalid"
        
        # Decision Matrix
        if r_pb >= validity_limit and d >= 0.3:
            final_decision = "RETAIN"
        elif r_pb >= validity_limit and d >= 0.2:
            final_decision = "REVISE"
        else:
            final_decision = "REJECT"

        results.append({
            "Item": item,
            "P (Diff)": p,
            "Diff_Desc": p_desc,
            "Q (Fail)": 1-p,
            "PQ (Var)": p*(1-p),
            "D (Disc)": d,
            "Disc_Desc": d_desc,
            "DDI": (p_up - p_lo) / p_up if p_up > 0 else 0,
            "r_pbis": r_pb,
            "Val_Desc": v_desc,
            "DECISION": final_decision
        })

    df_res = pd.DataFrame(results)

    # 5. RELIABILITY
    sum_pq = df_res["PQ (Var)"].sum()
    var_total = total_scores.var(ddof=1)
    kr20 = (n_items/(n_items-1)) * (1 - (sum_pq/var_total)) if var_total > 0 else 0

    # DASHBOARD
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Students (N)", n_students)
    m2.metric("Items (k)", n_items)
    m3.metric("KR-20 Reliability", f"{kr20:.3f}")

    # 6. TRAFFIC LIGHT STYLING
    def apply_styles(row):
        styles = [''] * len(row)
        # Difficulty (P)
        if row['P (Diff)'] < 0.3: styles[1] = 'background-color: #ffcccc' 
        elif row['P (Diff)'] > 0.7: styles[1] = 'background-color: #ccffcc'
        else: styles[1] = 'background-color: #fff2cc'
        
        # Discrimination (D)
        if row['D (Disc)'] >= 0.4: styles[5] = 'background-color: #2ecc71; color: white'
        elif row['D (Disc)'] < 0.2: styles[5] = 'background-color: #e74c3c; color: white'
        else: styles[5] = 'background-color: #f1c40f'
        
        # Validity (r_pbis)
        if row['r_pbis'] >= validity_limit: styles[8] = 'background-color: #2ecc71; color: white'
        else: styles[8] = 'background-color: #e74c3c; color: white'
        
        return styles

    st.subheader("📋 Item Analysis Matrix & Decisions")
    styled_df = df_res.style.apply(apply_styles, axis=1)\
                            .format("{:.3f}", subset=["P (Diff)", "Q (Fail)", "PQ (Var)", "D (Disc)", "DDI", "r_pbis"])
    st.dataframe(styled_df, use_container_width=True)

    # 7. AUTOMATIC INTERPRETATION
    st.divider()
    st.header("📝 Statistical Report")
    if kr20 >= 0.7:
        st.success(f"**Test Reliability is High ({kr20:.3f}).** The instrument is stable.")
    else:
        st.error(f"**Test Reliability is Low ({kr20:.3f}).** Revision highly recommended.")

    # 8. DISTRACTOR ANALYSIS
    st.subheader("🎯 Distractor Efficiency (A-E)")
    dist_data = [df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict() | {"Item": item} for item in item_cols]
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
    cols = sorted([c for c in df_dist.columns if len(str(c)) == 1]) + sorted([c for c in df_dist.columns if len(str(c)) > 1])
    st.dataframe(df_dist[cols].style.background_gradient(cmap='YlGn').format("{:.2%}"), use_container_width=True)

    # EXPORT
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='ItemAnalysis')
    st.download_button("📥 Export Report", data=buf.getvalue(), file_name="Item_Analysis.xlsx")
