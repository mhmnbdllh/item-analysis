import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ACADEMIC ITEM ANALYSIS TOOL - FINAL STABLE VERSION
# ======================================================================

st.set_page_config(page_title="Item Analysis Tool", page_icon="📊", layout="wide")

st.title("📊 ACADEMIC ITEM ANALYSIS (PRO VERSION)")
st.write("Full metrics with automatic color-coded pedagogical interpretations.")

# SIDEBAR CONFIG
with st.sidebar:
    st.header("⚙️ Threshold Parameters")
    group_percent = st.slider("Grouping % (Kelley)", 10, 50, 27)
    validity_limit = st.number_input("Min r_pbis (Validity)", 0.0, 1.0, 0.25)
    st.markdown("---")
    st.markdown("""
    **Color Markers Legend:**
    - 🟢 **Green:** Excellent / Ideal
    - 🟡 **Yellow:** Marginal / Needs Review
    - 🔴 **Red:** Poor / Reject
    """)

# FILE UPLOADER
u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Responses (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Answer Key (CSV)", type=['csv'])

if student_file and key_file:
    # Data Loading
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
    # 1. SCORING ENGINE
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores

    # 2. GROUPING (UPPER/LOWER 27%)
    n_group = max(1, int(np.ceil(len(df) * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    up_idx, lo_idx = df_sorted.head(n_group).index, df_sorted.tail(n_group).index

    # 3. METRICS CALCULATION
    results = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        p_up = df_scores.loc[up_idx, item].mean()
        p_lo = df_scores.loc[lo_idx, item].mean()
        d = p_up - p_lo
        # DDI Formula: (P_upper - P_lower) / P_upper
        ddi = (p_up - p_lo) / p_up if p_up > 0 else 0
        
        r_pb, _ = pointbiserialr(df_scores[item], total_scores) if df_scores[item].var() != 0 else (0,0)

        results.append({
            "Item": item,
            "P (Difficulty)": p,
            "Q (Failure)": 1-p,
            "PQ (Variance)": p*(1-p),
            "D (Discrim)": d,
            "DDI": ddi,
            "r_pbis (Valid)": r_pb,
            "Status": "RETAIN" if (r_pb >= validity_limit and d >= 0.2) else "REVISE"
        })

    df_res = pd.DataFrame(results)

    # 4. COLOR CODING FUNCTIONS (Using .map for Compatibility)
    def style_p(val):
        if val < 0.3: color = '#ffcccc' # Hard (Pink/Red)
        elif val > 0.7: color = '#ccffcc' # Easy (Green)
        else: color = '#fff2cc' # Moderate (Yellow)
        return f'background-color: {color}'

    def style_d_r(val):
        if val >= 0.4: color = '#2ecc71; color: white' # Excellent (Solid Green)
        elif val >= 0.2: color = '#f1c40f' # Marginal (Yellow)
        else: color = '#e74c3c; color: white' # Poor (Solid Red)
        return f'background-color: {color}'

    # STYLING TABLE
    st.subheader("📋 Item Analysis Matrix")
    styled_df = df_res.style.map(style_p, subset=['P (Difficulty)'])\
                            .map(style_d_r, subset=['D (Discrim)', 'r_pbis (Valid)'])\
                            .format("{:.3f}", subset=['P (Difficulty)', 'Q (Failure)', 'PQ (Variance)', 'D (Discrim)', 'DDI', 'r_pbis (Valid)'])
    
    st.dataframe(styled_df, use_container_width=True)

    # 5. DESCRIPTIVE INTERPRETATIONS
    st.divider()
    st.header("📝 Descriptive Interpretation")
    
    total_var = total_scores.var(ddof=1)
    kr20 = (len(item_cols)/(len(item_cols)-1)) * (1 - (df_res["PQ (Variance)"].sum()/total_var)) if total_var > 0 else 0
    
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Reliability (KR-20): {kr20:.3f}**")
        if kr20 >= 0.8: st.success("Interpretation: Very High Consistency.")
        elif kr20 >= 0.6: st.warning("Interpretation: Moderate Consistency.")
        else: st.error("Interpretation: Low Reliability. Scores may be unstable.")

    with c2:
        good_items = len(df_res[df_res['Status'] == 'RETAIN'])
        st.metric("Accepted Items", f"{good_items}/{len(item_cols)}")
        st.write(f"Test Efficiency: {(good_items/len(item_cols))*100:.1f}%")

    # 6. DISTRACTOR ANALYSIS (ALPHABETICAL)
    st.subheader("🎯 Distractor Analysis (Option Frequency Heatmap)")
    dist_data = [df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict() | {"Item": item} for item in item_cols]
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
    
    # Sort columns A-E and others
    sorted_cols = sorted([c for c in df_dist.columns if len(str(c)) == 1])
    other_cols = sorted([c for c in df_dist.columns if len(str(c)) > 1])
    df_dist = df_dist[sorted_cols + other_cols]
    
    st.dataframe(df_dist.style.background_gradient(cmap='YlGnBu', axis=None).format("{:.2%}"), use_container_width=True)

    # DOWNLOAD BUTTON
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Item_Summary')
        df_dist.to_excel(writer, sheet_name='Distractor_Analysis')
    st.download_button("📥 Download Excel Report", data=buf.getvalue(), file_name="Item_Analysis_Report.xlsx")
