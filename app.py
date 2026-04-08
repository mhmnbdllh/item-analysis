import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ACADEMIC ITEM ANALYSIS - RIGOROUS ENGLISH VERSION (2026)
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f4f9; }
    [data-testid="stMetricValue"] { font-family: 'Courier New', Courier, monospace; color: #1a1a1a; }
    .stAlert { border: 2px solid #000; border-radius: 0px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ RIGOROUS ITEM ANALYSIS TOOL (CTT)")

with st.sidebar:
    st.header("📊 Methodological Legend")
    with st.expander("1. Difficulty Index (p/d)", expanded=True):
        st.write("- **Easy (p > 0.70):** 🟢\n- **Moderate (0.30 - 0.70):** 🟡\n- **Difficult (p < 0.30):** 🔴")
    with st.expander("2. Discrimination (ddi)", expanded=True):
        st.write("- **Excellent (ddi ≥ 0.40):** 🟢\n- **Good (0.30 - 0.39):** 🔵\n- **Fair (0.20 - 0.29):** 🟡\n- **Poor (ddi < 0.20):** 🔴")
    st.header("⚙️ Settings")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27)
    validity_limit = st.number_input("r_pbis Threshold", 0.0, 1.0, 0.25)

u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Key Data (CSV)", type=['csv'])

if student_file and key_file:
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students = len(df)

    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    up_idx, lo_idx = df_sorted.head(n_group).index, df_sorted.tail(n_group).index

    results = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        q = 1 - p
        p_up, p_lo = df_scores.loc[up_idx, item].mean(), df_scores.loc[lo_idx, item].mean()
        d_val = p_up - p_lo 
        r_pb, _ = pointbiserialr(df_scores[item], total_scores) if df_scores[item].var() != 0 else (0,0)

        p_desc = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_desc = "Excellent" if d_val >= 0.4 else "Good" if d_val >= 0.3 else "Fair" if d_val >= 0.2 else "Poor"
        r_desc = "Valid" if r_pb >= validity_limit else "Invalid"
        
        if r_pb >= validity_limit and d_val >= 0.3: decision = "RETAIN"
        elif r_pb >= 0.2 and d_val >= 0.2: decision = "REVISE"
        else: decision = "REJECT"

        results.append({
            "Item": item, "p": p, "p_Eval": p_desc, "q": q, "pq": p*q,
            "ddi": d_val, "d": p, "d_Eval": d_desc, "r_pbis": r_pb, 
            "r_Eval": r_desc, "DECISION": decision
        })

    df_res = pd.DataFrame(results)
    kr20 = (len(item_cols)/(len(item_cols)-1)) * (1 - (df_res["pq"].sum()/total_scores.var())) if total_scores.var() > 0 else 0

    st.subheader("📋 Comprehensive Item Statistics Matrix")
    def apply_full_styling(row):
        styles = [''] * len(row)
        dif_color = '#ccffcc' if row['p'] > 0.7 else '#ffcccc' if row['p'] < 0.3 else '#fff2cc'
        styles[1] = styles[2] = styles[6] = f'background-color: {dif_color}'
        dis_color = '#2ecc71' if row['ddi'] >= 0.4 else '#3498db' if row['ddi'] >= 0.3 else '#f1c40f' if row['ddi'] >= 0.2 else '#e74c3c'
        styles[5] = styles[7] = f'background-color: {dis_color}; color: white'
        val_bg = '#ccffcc' if row['r_pbis'] >= validity_limit else '#ffcccc'
        styles[8] = styles[9] = f'background-color: {val_bg}'
        styles[10] = 'background-color: #27ae60; color: white' if row['DECISION'] == "RETAIN" else 'background-color: #f39c12; color: white' if row['DECISION'] == "REVISE" else 'background-color: #c0392b; color: white'
        return styles

    st.dataframe(df_res.style.apply(apply_full_styling, axis=1).format("{:.3f}", subset=["p", "q", "pq", "ddi", "d", "r_pbis"]), use_container_width=True)

    # 8. DISTRACTOR ANALYSIS (FORMAT 1 (10%) + TETAP BERWARNA)
    st.subheader("🎯 Distractor Effectiveness (Option Frequency)")
    
    # Hitung persentase asli untuk warna
    dist_pct = df[item_cols].apply(lambda x: x.astype(str).str.upper().str.strip().value_counts(normalize=True)).fillna(0).T
    # Hitung frekuensi untuk teks
    dist_freq = df[item_cols].apply(lambda x: x.astype(str).str.upper().str.strip().value_counts()).fillna(0).T
    
    # Gabungkan menjadi format 1 (10%)
    df_dist_display = pd.DataFrame(index=item_cols)
    for col in dist_pct.columns:
        df_dist_display[col] = [f"{int(f)} ({p:.0%})" for f, p in zip(dist_freq[col], dist_pct[col])]

    # FUNGSI WARNA BARU (Mewarnai Teks berdasarkan Nilai Persentase Asli)
    def color_distractor(val_display, val_pct):
        color = 'background-color: rgba(46, 204, 113, ' + str(val_pct) + ')' # Warna hijau transparan sesuai %
        return color

    # Terapkan warna ke tabel display menggunakan data persentase asli sebagai referensi
    st.dataframe(df_dist_display.style.apply(lambda x: [color_distractor(v, p) for v, p in zip(x, dist_pct[x.name])], axis=0), use_container_width=True)

    # EXPORT KE EXCEL
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Item_Analysis')
        df_dist_display.to_excel(writer, index=True, sheet_name='Distractor_Analysis')
    
    st.download_button(label="📥 Download Full Report", data=buf.getvalue(), file_name="Item_Analysis_Report.xlsx")
