import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ======================================================
# CTT ITEM ANALYSIS - VALID PSYCHOMETRIC IMPLEMENTATION
# ======================================================

st.set_page_config(page_title="CTT Item Analysis", page_icon="📊", layout="wide")

st.title("Classical Test Theory (CTT) Item Analysis Tool")

# ======================
# INPUT
# ======================
col1, col2 = st.columns(2)

with col1:
    student_file = st.file_uploader("Upload Student Responses (CSV)", type="csv")
with col2:
    key_file = st.file_uploader("Upload Answer Key (CSV)", type="csv")

with st.sidebar:
    st.header("Settings")
    group_percent = st.slider("Upper-Lower Group (%)", 10, 50, 27)
    r_threshold = st.number_input("Point-Biserial Threshold", 0.0, 1.0, 0.25)

if student_file and key_file:

    df = pd.read_csv(student_file).fillna("")
    key_df = pd.read_csv(key_file)

    item_cols = df.columns[1:]
    id_col = df.columns[0]

    key = key_df.iloc[0, 1:].astype(str).str.strip().str.upper().tolist()

    # ======================
    # SCORING MATRIX (0/1)
    # ======================
    score = pd.DataFrame()

    for i, col in enumerate(item_cols):
        score[col] = (
            df[col].astype(str).str.strip().str.upper()
            == key[i]
        ).astype(int)

    total = score.sum(axis=1)
    df["Total"] = total

    n_items = len(item_cols)
    n_students = len(df)

    # ======================
    # UPPER-LOWER GROUP
    # ======================
    g = max(1, int(round(n_students * group_percent / 100)))

    df_sorted = df.sort_values("Total", ascending=False)
    upper = df_sorted.head(g).index
    lower = df_sorted.tail(g).index

    # ======================
    # ITEM ANALYSIS
    # ======================
    results = []

    for i, item in enumerate(item_cols):

        x = score[item]

        # Difficulty index
        p = x.mean()
        q = 1 - p

        # Item variance (for dichotomous items)
        var = p * q

        # Discrimination (Kelley index)
        p_upper = x.loc[upper].mean()
        p_lower = x.loc[lower].mean()
        D = p_upper - p_lower

        # Point-biserial correlation (corrected)
        corrected_total = total - x

        if x.nunique() > 1:
            r_pb, _ = pointbiserialr(x, corrected_total)
            if np.isnan(r_pb):
                r_pb = 0
        else:
            r_pb = 0

        # ======================
        # INTERPRETATION
        # ======================
        if p < 0.30:
            diff = "Difficult"
        elif p > 0.70:
            diff = "Easy"
        else:
            diff = "Moderate"

        if D >= 0.40:
            disc = "Excellent"
        elif D >= 0.30:
            disc = "Good"
        elif D >= 0.20:
            disc = "Fair"
        else:
            disc = "Poor"

        valid = "Valid" if r_pb >= r_threshold else "Invalid"

        # ======================
        # DECISION RULE (CTT-BASED)
        # ======================
        if (r_pb >= r_threshold) and (D >= 0.30) and (0.20 <= p <= 0.90):
            decision = "RETAIN"
        elif (D < 0.20) or (r_pb < r_threshold):
            decision = "REJECT"
        else:
            decision = "REVISE"

        results.append({
            "Item": item,
            "p": p,
            "q": q,
            "Variance": var,
            "Discrimination_D": D,
            "Point_Biserial": r_pb,
            "Difficulty": diff,
            "Discrimination_Level": disc,
            "Validity": valid,
            "Decision": decision
        })

    df_res = pd.DataFrame(results)

    # ======================
    # RELIABILITY (KR-20)
    # ======================
    total_var = total.var(ddof=1)

    if total_var > 0:
        kr20 = (n_items / (n_items - 1)) * (1 - df_res["Variance"].sum() / total_var)
    else:
        kr20 = 0

    alpha = kr20  # dichotomous equivalence

    sem = np.sqrt(total_var) * np.sqrt(1 - kr20)

    # ======================
    # OUTPUT
    # ======================
    st.subheader("Test Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", n_students)
    c2.metric("Items", n_items)
    c3.metric("KR-20", f"{kr20:.3f}")
    c4.metric("SEM", f"{sem:.3f}")

    st.subheader("Item Statistics")
    st.dataframe(df_res, use_container_width=True)

    # ======================
    # EXPORT
    # ======================
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_res.to_excel(writer, index=False, sheet_name="CTT_Item_Analysis")

    st.download_button(
        "Download Report",
        data=buffer.getvalue(),
        file_name="CTT_Item_Analysis.xlsx"
    )
