import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ACADEMIC ITEM ANALYSIS TOOL - VISUAL GRADIENT EDITION
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="🎨", layout="wide")

st.title("🎨 ACADEMIC ITEM ANALYSIS (VISUAL GRADIENT VERSION)")
st.write("Full metrics with automatic color-coded pedagogical interpretations.")

# SIDEBAR CONFIG
with st.sidebar:
    st.header("⚙️ Threshold Settings")
    group_percent = st.slider("Grouping %", 10, 50, 27)
    validity_limit = st.number_input("Min r_pbis", 0.0, 1.0, 0.25)
    st.markdown("---")
    st.markdown("""
    **Color Legend:**
    - 🟢 **Green:** Excellent / Standard
    - 🟡 **Yellow:** Marginal / Review
    - 🔴 **Red:** Poor / Reject
    """)

# FILE UPLOADER
u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Key (CSV)", type=['csv'])

if student_file and key_file:
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].str.upper().str.strip().tolist()
    
    # 1. SCORING
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores

    # 2. GROUPING
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
        r_pb, _ = pointbiserialr(df_scores[item], total_scores) if df_scores[item].var() != 0 else (0,0)

        results.append({
            "Item": item,
            "P (Difficulty)": round(p, 3),
            "Q (Failure)": round(1-p, 3),
            "PQ (Var)": round(p*(1-p), 3),
            "D (Discrim)": round(d, 3),
            "r_pbis (Valid)": round(r_pb, 3),
            "Status": "RETAIN" if (r_pb >= validity_limit and d >= 0.2) else "REJECT"
        })

    df_res = pd.DataFrame(results)

    # 4. COLOR CODING FUNCTIONS (GRADIENT & MARKERS)
    def style_p(val):
        if val < 0.3: color = '#ffcccc' # Hard (Red-ish)
        elif val > 0.7: color = '#ccffcc' # Easy (Green-ish)
        else: color = '#fff2cc' # Moderate (Yellow-ish)
        return f'background-color: {color}'

    def style_d_r(val):
        if val >= 0.4: color = '#2ecc71; color: white' # Excellent
        elif val >= 0.2: color = '#f1c40f' # Marginal
        else: color = '#e74c3c; color: white' # Poor
        return f'background-color: {color}'

    # APPLY STYLING
    st.subheader("📋 Item Analysis Matrix with Color Markers")
    styled_df = df_res.style.applymap(style_p, subset=['P (Difficulty)'])\
                            .applymap(style_d_r, subset=['D (Discrim)', 'r_pbis (Valid)'])\
                            .format("{:.3f}", subset=['P (Difficulty)', 'D (Discrim)', 'r_pbis (Valid)'])
    
    st.dataframe(styled_df, use_container_width=True)

    # 5. DESCRIPTIVE INTERPRETATION SUMMARY
    st.divider()
    st.header("📝 Descriptive Interpretation Report")
    
    total_var = total_scores.var(ddof=1)
    kr20 = (len(item_cols)/(len(item_cols)-1)) * (1 - (df_res["PQ (Var)"].sum()/total_var))
    
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Reliability Analysis (KR-20): {kr20:.3f}**")
        if kr20 >= 0.8: st.success("Interpretation: Very High Reliability. The test is consistent for high-stakes assessment.")
        elif kr20 >= 0.6: st.warning("Interpretation: Moderate Reliability. Acceptable for classroom use but needs improvement.")
        else: st.error("Interpretation: Low Reliability. Scores are not stable. Review the whole test structure.")

    with c2:
        good_items = len(df_res[df_res['Status'] == 'RETAIN'])
        st.metric("Healthy Items (Retain)", f"{good_items}/{len(item_cols)}")
        st.write(f"Efficiency Rate: {(good_items/len(item_cols))*100:.1f}%")

    # 6. DISTRACTOR ANALYSIS (ALPHABETICAL)
    st.subheader("🎯 Distractor Efficiency (Option Frequency)")
    dist_data = [df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict() | {"Item": item} for item in item_cols]
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
    df_dist = df_dist[sorted([c for c in df_dist.columns if len(c)==1]) + sorted([c for c in df_dist.columns if len(c)>1])]
    
    # Heatmap styling for distractors
    st.dataframe(df_dist.style.background_gradient(cmap='YlGnBu', axis=None).format("{:.2%}"), use_container_width=True)

    # EXPORT
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Summary')
        df_dist.to_excel(writer, sheet_name='Distractors')
    st.download_button("📥 Download Analysis", data=buf.getvalue(), file_name="Item_Analysis_Report.xlsx")
