import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ACADEMIC ITEM ANALYSIS TOOL - TRAFFIC LIGHT EDITION
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📊", layout="wide")
st.title("📊 ACADEMIC ITEM ANALYSIS (OFFICIAL STANDARDS)")

with st.sidebar:
    st.header("⚙️ Threshold Settings")
    group_percent = st.slider("Grouping % (Kelley)", 10, 50, 27)
    validity_limit = st.number_input("Min r_pbis (Validity)", 0.0, 1.0, 0.25)
    st.markdown("---")
    st.markdown("""
    **Color Legend:**
    - 🟢 **Green:** Excellent (No revision)
    - 🟡 **Yellow:** Moderate (Needs review)
    - 🔴 **Red:** Poor (Discard/Reject)
    """)

u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Responses (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Answer Key (CSV)", type=['csv'])

if student_file and key_file:
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
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

    # 3. METRICS
    results = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        p_up, p_lo = df_scores.loc[up_idx, item].mean(), df_scores.loc[lo_idx, item].mean()
        d = p_up - p_lo
        ddi = (p_up - p_lo) / p_up if p_up > 0 else 0
        r_pb, _ = pointbiserialr(df_scores[item], total_scores) if df_scores[item].var() != 0 else (0,0)

        results.append({
            "Item": item, "P": p, "Q": 1-p, "PQ": p*(1-p), "D": d, "DDI": ddi, "r_pbis": r_pb
        })
    df_res = pd.DataFrame(results)

    # 4. COLOR MARKERS (FIXED TO RED, YELLOW, GREEN)
    def style_p(val):
        if val < 0.3: return 'background-color: #ffcccc' # Red (Hard)
        if val > 0.7: return 'background-color: #ccffcc' # Green (Easy)
        return 'background-color: #fff2cc' # Yellow (Moderate)

    def style_d_r(val):
        if val >= 0.4: return 'background-color: #2ecc71; color: white' # Green (Excellent)
        if val >= 0.2: return 'background-color: #f1c40f' # Yellow (Marginal)
        return 'background-color: #e74c3c; color: white' # Red (Poor)

    st.subheader("📋 Item Analysis Matrix")
    styled_df = df_res.style.map(style_p, subset=['P'])\
                            .map(style_d_r, subset=['D', 'r_pbis'])\
                            .format("{:.3f}", subset=['P', 'Q', 'PQ', 'D', 'DDI', 'r_pbis'])
    st.dataframe(styled_df, use_container_width=True)

    # 5. DESCRIPTIVE INTERPRETATIONS (DETAILED)
    st.divider()
    st.header("📝 Descriptive Interpretation Report")
    
    total_var = total_scores.var(ddof=1)
    kr20 = (len(item_cols)/(len(item_cols)-1)) * (1 - (df_res["PQ"].sum()/total_var)) if total_var > 0 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Test Reliability")
        if kr20 >= 0.8:
            st.success(f"**KR-20: {kr20:.3f} (Very High)**\nThe test is highly consistent and reliable for academic evaluation.")
        elif kr20 >= 0.6:
            st.warning(f"**KR-20: {kr20:.3f} (Moderate)**\nThe test is acceptable but some items need refinement to increase consistency.")
        else:
            st.error(f"**KR-20: {kr20:.3f} (Low)**\nWarning: This test is inconsistent. Results may not accurately reflect student ability.")

    with col2:
        st.subheader("Item Quality Summary")
        retain = len(df_res[(df_res['r_pbis'] >= validity_limit) & (df_res['D'] >= 0.2)])
        st.metric("Recommended to Retain", f"{retain} / {len(item_cols)}")
        st.write(f"Items with low D or negative r_pbis should be discarded or re-keyed.")

    # 6. DISTRACTOR ANALYSIS
    st.subheader("🎯 Distractor Efficiency")
    dist_data = [df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict() | {"Item": item} for item in item_cols]
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
    sorted_cols = sorted([c for c in df_dist.columns if len(str(c)) == 1]) + sorted([c for c in df_dist.columns if len(str(c)) > 1])
    st.dataframe(df_dist[sorted_cols].style.background_gradient(cmap='YlGn').format("{:.2%}"), use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Stats')
        df_dist.to_excel(writer, sheet_name='Distractors')
    st.download_button("📥 Export Report", data=buf.getvalue(), file_name="Item_Analysis.xlsx")
