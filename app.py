import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ITEM ANALYSIS TOOL - FULL ACADEMIC SUITE (CTT)
# ======================================================================

st.set_page_config(page_title="Academic Item Analysis", page_icon="📈", layout="wide")

# Custom CSS for Brutalist Aesthetic
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { border: 3px solid #000; padding: 15px; background: #fff; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 ACADEMIC ITEM ANALYSIS TOOL (CTT)")
st.write("Comprehensive Classical Test Theory Analysis: Difficulty, Discrimination, and Reliability.")

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("⚙️ Analysis Parameters")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27)
    validity_min = st.number_input("Min. r_pbis (Validity)", 0.0, 1.0, 0.25)
    
    st.markdown("---")
    st.write("**Legend:**")
    st.write("- **P:** Difficulty Index")
    st.write("- **Q:** 1 - P")
    st.write("- **PQ:** Item Variance")
    st.write("- **D:** Discrimination Index")
    st.write("- **DDI:** Discrimination Difficulty Index")

# FILE UPLOADER
col_up1, col_up2 = st.columns(2)
with col_up1:
    student_file = st.file_uploader("Upload Student Responses (CSV)", type=['csv'])
with col_up2:
    key_file = st.file_uploader("Upload Answer Key (CSV)", type=['csv'])

if student_file and key_file:
    # Load and clean data
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].str.upper().str.strip().tolist()
    
    # 1. SCORING ENGINE
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        # Comparison with key (Vectorized)
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students = len(df)
    n_items = len(item_cols)

    # 2. GROUPING (UPPER/LOWER)
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    upper_idx = df_sorted.head(n_group).index
    lower_idx = df_sorted.tail(n_group).index

    # 3. CORE STATISTICAL CALCULATION
    item_stats = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        q = 1 - p
        pq = p * q
        
        p_upper = df_scores.loc[upper_idx, item].mean()
        p_lower = df_scores.loc[lower_idx, item].mean()
        
        d_index = p_upper - p_lower
        # DDI Formula: (Pu - Pl) / Pu (Sensitivity to the upper group)
        ddi = (p_upper - p_lower) / p_upper if p_upper > 0 else 0
        
        # Point Biserial (r_pbis)
        if df_scores[item].var() == 0:
            r_pbis = 0
        else:
            r_pbis, _ = pointbiserialr(df_scores[item], total_scores)

        # Interpretation Logic
        status = "RETAIN" if (r_pbis >= validity_min and d_index >= 0.2) else "REVISE"

        item_stats.append({
            "Item": item,
            "P": round(p, 3),
            "Q": round(q, 3),
            "PQ": round(pq, 3),
            "P_Upper": round(p_upper, 3),
            "P_Lower": round(p_lower, 3),
            "D": round(d_index, 3),
            "DDI": round(ddi, 3),
            "r_pbis": round(r_pbis, 3),
            "Decision": status
        })

    df_results = pd.DataFrame(item_stats)

    # 4. RELIABILITY (KR-20)
    sum_pq = df_results["PQ"].sum()
    var_total = total_scores.var(ddof=1)
    kr20 = (n_items/(n_items-1)) * (1 - (sum_pq/var_total)) if var_total > 0 else 0
    sem = total_scores.std(ddof=1) * np.sqrt(1 - kr20)

    # DASHBOARD OUTPUT
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Students (N)", n_students)
    m2.metric("Items (k)", n_items)
    m3.metric("KR-20 Reliability", f"{kr20:.3f}")
    m4.metric("SEM", f"{sem:.3f}")

    # ITEM TABLE
    st.subheader("📋 Item Analysis Matrix")
    st.dataframe(df_results.style.background_gradient(subset=['P', 'D', 'r_pbis'], cmap='Blues'), use_container_width=True)

    # 5. DISTRACTOR ANALYSIS (ALPHABETICAL FIX)
    st.subheader("🎯 Distractor Analysis (Option Frequency)")
    dist_list = []
    for item in item_cols:
        counts = df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict()
        counts['Item'] = item
        dist_list.append(counts)
    
    df_dist = pd.DataFrame(dist_list).set_index('Item').fillna(0)
    # Sort columns: Single letters first (A-Z), then others (N/A, Missing)
    sorted_cols = sorted([c for c in df_dist.columns if len(c) == 1])
    other_cols = sorted([c for c in df_dist.columns if len(c) > 1])
    df_dist = df_dist[sorted_cols + other_cols]
    
    st.dataframe(df_dist.style.format("{:.2%}"), use_container_width=True)

    # DOWNLOAD
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, sheet_name='Item_Stats', index=False)
        df_dist.to_excel(writer, sheet_name='Distractors')
    st.download_button("📥 Download Full Report", data=output.getvalue(), file_name="Item_Analysis.xlsx")

# ======================================================================
# STATISTICAL INTERPRETATION GUIDE
# ======================================================================
st.divider()
st.header("📖 Statistical Interpretation Guide")

with st.expander("1. Difficulty Index (P)"):
    st.write("""
    - **P > 0.70:** Easy item.
    - **0.30 ≤ P ≤ 0.70:** Moderate/Ideal item.
    - **P < 0.30:** Difficult item.
    - *Note: In PQ calculation, a P value of 0.50 yields the maximum variance (0.25), which is best for distinguishing student ability.*
    """)

with st.expander("2. Discrimination Index (D)"):
    st.write("""
    - **D ≥ 0.40:** Excellent discrimination.
    - **0.30 ≤ D < 0.39:** Good.
    - **0.20 ≤ D < 0.29:** Marginal (needs revision).
    - **D < 0.20:** Poor (reject item).
    - *Formula: P(Upper Group) - P(Lower Group).*
    """)

with st.expander("3. Point Biserial Correlation (r_pbis)"):
    st.write("""
    Measures the relationship between performance on a specific item and the total test score. 
    A high positive value indicates that students who did well on the whole test also got this item right. 
    **Negative values** suggest a flawed item or a miskeyed answer.
    """)

with st.expander("4. KR-20 Reliability"):
    st.write("""
    - **> 0.80:** Very high reliability (standard for high-stakes exams).
    - **0.70 - 0.80:** Respectable.
    - **0.60 - 0.70:** Minimum acceptable for classroom tests.
    - **< 0.60:** Low reliability; test results may be inconsistent.
    """)
