# ======================================================================
# ITEM ANALYSIS - STREAMLIT VERSION (COMPLETE + BUG FIXED)
# ======================================================================
# BUG FIX: Upper_N = 0 issue - Fixed group formation
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import io
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# PAGE CONFIGURATION
# ======================================================================
st.set_page_config(page_title="Item Analysis Tool", page_icon="📊", layout="wide")

st.title("📊 ITEM ANALYSIS TOOL")
st.markdown("---")

# ======================================================================
# SIDEBAR - THRESHOLD PARAMETERS
# ======================================================================
with st.sidebar:
    st.header("⚙️ Threshold Parameters")
    st.caption("Adjust these values based on your assessment standards")
    
    st.subheader("Difficulty Index (p)")
    col1, col2 = st.columns(2)
    with col1:
        difficult_threshold = st.number_input("Difficult (<)", value=0.30, step=0.05)
    with col2:
        easy_threshold = st.number_input("Easy (>)", value=0.80, step=0.05)
    
    st.subheader("Discrimination Index (D)")
    col1, col2 = st.columns(2)
    with col1:
        poor_threshold = st.number_input("Poor (<)", value=0.20, step=0.05)
    with col2:
        good_threshold = st.number_input("Good (>=)", value=0.40, step=0.05)
    
    st.subheader("Validity (r_it)")
    valid_threshold = st.number_input("Valid (>=)", value=0.20, step=0.05)
    
    st.subheader("Group Classification")
    group_percent = st.slider("Upper/Lower Group Percentage", min_value=10, max_value=50, value=27, step=1)
    
    st.markdown("---")
    st.caption("Scripted by Muhaimin Abdullah")

# ======================================================================
# FUNCTIONS
# ======================================================================
def interpret_p(p, difficult_threshold, easy_threshold):
    if p < difficult_threshold:
        return "Difficult"
    elif p <= easy_threshold:
        return "Moderate"
    else:
        return "Easy"

def interpret_d(d, poor_threshold, good_threshold):
    if d < poor_threshold:
        return "Poor"
    elif d < good_threshold:
        return "Fair"
    else:
        return "Very Good"

def interpret_ddi(ddi):
    if ddi > 0:
        return "Functional"
    elif ddi == 0:
        return "Neutral"
    else:
        return "Non-Functional"

# ======================================================================
# INITIALIZE SESSION STATE
# ======================================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'answer_key_df' not in st.session_state:
    st.session_state.answer_key_df = None
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False

# ======================================================================
# DATA INPUT
# ======================================================================
tab1, tab2 = st.tabs(["📁 Upload Data", "📊 Analysis Results"])

with tab1:
    st.subheader("Upload Student Response File")
    student_file = st.file_uploader("Choose CSV file with student responses", type=['csv'], key="student")
    
    if student_file is not None:
        if student_file.size > 5 * 1024 * 1024:
            st.error("❌ File size exceeds 5MB limit!")
        else:
            try:
                student_file.seek(0)
                df = pd.read_csv(student_file, dtype=str)
                
                if df.empty:
                    st.error("❌ Empty data!")
                elif len(df.columns) < 2:
                    st.error("❌ Need at least 2 columns!")
                else:
                    st.success(f"✅ File uploaded: {student_file.name}")
                    st.write(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
                    st.subheader("Data Preview (First 5 rows)")
                    st.dataframe(df.head())
                    
                    st.session_state.df = df
                    st.session_state.file_loaded = True
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.subheader("Upload Answer Key File (Required for Multiple Choice)")
    key_file = st.file_uploader("Choose CSV file with answer keys", type=['csv'], key="answer_key")
    
    if key_file is not None:
        if key_file.size > 5 * 1024 * 1024:
            st.error("❌ File too large!")
        else:
            try:
                key_file.seek(0)
                df_key = pd.read_csv(key_file, dtype=str)
                if not df_key.empty:
                    st.success(f"✅ Answer key uploaded")
                    st.session_state.answer_key_df = df_key
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ======================================================================
# ANALYSIS PROCESS
# ======================================================================
if st.session_state.file_loaded and st.session_state.df is not None:
    
    df = st.session_state.df.copy()
    df_key = st.session_state.answer_key_df
    
    item_columns = df.columns[1:].tolist()
    n_items = len(item_columns)
    n_students = len(df)
    
    if n_items == 0:
        with tab2:
            st.error("❌ No item columns found!")
    elif df_key is None or df_key.empty:
        with tab2:
            st.error("❌ ANSWER KEY IS REQUIRED! Please upload answer key file.")
    else:
        # Read answer key from row 0
        try:
            if df_key.shape[1] > 1:
                answer_key = [str(x).strip().upper() for x in df_key.iloc[0, 1:].values]
            else:
                answer_key = [str(df_key.iloc[0, 0]).strip().upper()]
        except Exception as e:
            with tab2:
                st.error(f"❌ Failed to read answer key: {str(e)}")
                answer_key = None
        
        if answer_key is None:
            with tab2:
                st.error("❌ Could not read answer key!")
        elif len(answer_key) != n_items:
            with tab2:
                st.error(f"❌ Answer key length ({len(answer_key)}) does not match number of items ({n_items})")
        else:
            # Convert to binary scores (1/0)
            df_scores = pd.DataFrame()
            for i, item in enumerate(item_columns):
                key_value = answer_key[i]
                df_scores[item] = (df[item].astype(str).str.strip().str.upper() == key_value).astype(int)
            
            df['total_score'] = df_scores.sum(axis=1)
            
            # ==================================================================
            # CRITICAL BUG FIX: Correct upper and lower group formation
            # ==================================================================
            # Sort by total_score
            df_sorted = df.sort_values('total_score', ascending=False).reset_index(drop=True)
            
            # Calculate number of students in each group (27% by default)
            n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
            
            # Create upper and lower groups using iloc on sorted dataframe
            upper_group = df_sorted.iloc[:n_group].copy()
            lower_group = df_sorted.iloc[-n_group:].copy()
            
            # Debug: Show group total scores to verify groups are formed correctly
            st.caption(f"**Debug:** Upper group total scores: {upper_group['total_score'].tolist()[:5]}...")
            st.caption(f"**Debug:** Lower group total scores: {lower_group['total_score'].tolist()[:5]}...")
            st.caption(f"**Debug:** Group size: {n_group} students (top {group_percent}%)")
            
            # ==================================================================
            # ITEM STATISTICS
            # ==================================================================
            results = []
            p_values = []
            q_values = []
            pq_values = []
            p_upper_values = []
            p_lower_values = []
            d_values = []
            se_values = []
            r_values = []
            alpha_if_deleted_values = []
            
            for i, item in enumerate(item_columns):
                p_val = df_scores[item].mean()
                p_values.append(p_val)
                
                q_val = 1 - p_val
                q_values.append(q_val)
                
                pq_val = p_val * q_val
                pq_values.append(pq_val)
                
                key_value = answer_key[i]
                
                # Calculate proportion correct using the CORRECTLY formed groups
                upper_correct = (upper_group[item].astype(str).str.strip().str.upper() == key_value).sum()
                p_upper = upper_correct / n_group
                
                lower_correct = (lower_group[item].astype(str).str.strip().str.upper() == key_value).sum()
                p_lower = lower_correct / n_group
                
                p_upper_values.append(p_upper)
                p_lower_values.append(p_lower)
                
                d_val = p_upper - p_lower
                d_values.append(d_val)
                
                se_val = np.sqrt(pq_val / n_students) if n_students > 0 else 0
                se_values.append(se_val)
                
                total_minus_item = df['total_score'] - df_scores[item]
                if df_scores[item].var() == 0 or total_minus_item.var() == 0:
                    r_it = 0.0
                else:
                    r_it, _ = pointbiserialr(df_scores[item], total_minus_item)
                r_values.append(r_it)
                
                total_without = df['total_score'] - df_scores[item]
                var_without = total_without.var(ddof=1)
                pq_without = 0
                for j, item2 in enumerate(item_columns):
                    if j != i:
                        p2 = df_scores[item2].mean()
                        pq_without += p2 * (1 - p2)
                if var_without > 0 and (n_items - 1) > 1:
                    alpha = ((n_items - 1) / (n_items - 2)) * (1 - (pq_without / var_without))
                else:
                    alpha = 0
                alpha_if_deleted_values.append(alpha)
                
                p_int = interpret_p(p_val, difficult_threshold, easy_threshold)
                d_int = interpret_d(d_val, poor_threshold, good_threshold)
                v_int = "Valid" if r_it >= valid_threshold else "Invalid"
                
                if r_it >= valid_threshold and d_val >= poor_threshold and difficult_threshold <= p_val <= easy_threshold:
                    rec = "RETAIN"
                elif r_it < 0.10 or d_val < 0.10:
                    rec = "DROP"
                else:
                    rec = "REVISE"
                
                results.append([
                    item, 
                    round(p_val, 4), 
                    round(q_val, 4), 
                    round(pq_val, 4),
                    round(p_upper, 4), 
                    round(p_lower, 4), 
                    round(d_val, 4), 
                    d_int,
                    round(se_val, 6),
                    round(r_it, 4), 
                    v_int, 
                    round(alpha, 4), 
                    rec, 
                    p_int
                ])
            
            # KR-20
            total_variance = df['total_score'].var(ddof=1)
            sum_pq = sum(pq_values)
            if total_variance > 0 and n_items > 1:
                kr20 = (n_items/(n_items-1)) * (1 - (sum_pq / total_variance))
            else:
                kr20 = 0
            sem = df['total_score'].std(ddof=1) * np.sqrt(max(0, 1 - kr20))
            
            # ==================================================================
            # DISTRACTOR ANALYSIS
            # ==================================================================
            distractor_results = []
            
            all_options = set()
            for item in item_columns:
                values = df[item].astype(str).str.strip().str.upper().dropna()
                for v in values:
                    if v.isalpha() and len(v) == 1:
                        all_options.add(v)
            option_list = sorted(all_options)
            
            for i, item in enumerate(item_columns):
                key_value = answer_key[i]
                item_data = df[item].astype(str).str.strip().str.upper()
                
                for option in option_list:
                    if option == key_value:
                        continue
                    
                    total_select = (item_data == option).sum()
                    percent = (total_select / n_students) * 100 if n_students > 0 else 0
                    
                    # Use the CORRECTLY formed groups
                    upper_select = (upper_group[item].astype(str).str.strip().str.upper() == option).sum()
                    lower_select = (lower_group[item].astype(str).str.strip().str.upper() == option).sum()
                    
                    prop_upper = upper_select / n_group if n_group > 0 else 0
                    prop_lower = lower_select / n_group if n_group > 0 else 0
                    
                    ddi = prop_lower - prop_upper
                    ddi_int = interpret_ddi(ddi)
                    
                    meets_percent = percent >= 5.0
                    meets_lower_upper = lower_select > upper_select
                    
                    distractor_results.append([
                        item, key_value, option,
                        total_select, round(percent, 1),
                        upper_select, lower_select,
                        round(prop_upper, 4), round(prop_lower, 4),
                        round(ddi, 4), ddi_int,
                        "Yes" if meets_percent else "No",
                        "Yes" if meets_lower_upper else "No"
                    ])
            
            df_results = pd.DataFrame(results, columns=[
                'Item', 'p', 'q', 'pq', 'p_upper', 'p_lower', 'D', 'D_Interpretation',
                'SE', 'r_it', 'Validity', 'Alpha_if_deleted', 'Recommendation', 'p_Interpretation'
            ])
            
            # ==================================================================
            # DISPLAY RESULTS
            # ==================================================================
            with tab2:
                st.markdown("## 📋 ITEM ANALYSIS SUMMARY")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Students", n_students)
                with col2:
                    st.metric("Items", n_items)
                with col3:
                    st.metric("Mode", "MULTIPLE_CHOICE")
                with col4:
                    st.metric("KR-20", f"{kr20:.4f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if kr20 >= 0.80:
                        st.success(f"✅ Reliability: {kr20:.4f} (Very Good)")
                    elif kr20 >= 0.70:
                        st.info(f"📘 Reliability: {kr20:.4f} (Good)")
                    elif kr20 >= 0.60:
                        st.warning(f"⚠️ Reliability: {kr20:.4f} (Fair)")
                    else:
                        st.error(f"❌ Reliability: {kr20:.4f} (Poor)")
                with col2:
                    st.info(f"📏 SEM: {sem:.4f} | 95% CI: ±{sem*1.96:.2f}")
                    st.caption(f"Σpq = {sum_pq:.4f} | Total Variance = {total_variance:.4f}")
                
                st.markdown("---")
                st.markdown("## 📊 ITEM STATISTICS")
                st.dataframe(df_results, use_container_width=True)
                
                if distractor_results:
                    st.markdown("---")
                    st.markdown("## 🎯 DISTRACTOR ANALYSIS")
                    st.caption("**>=5%** = Selected by at least 5% of students | **Lower > Upper** = More low-ability than high-ability students")
                    
                    df_distractor = pd.DataFrame(distractor_results, columns=[
                        'Item', 'Key', 'Option', 'N_Select', 'Percent',
                        'Upper_N', 'Lower_N', 'Prop_Upper', 'Prop_Lower',
                        'DDI', 'DDI_Interpretation', '>=5%', 'Lower > Upper'
                    ])
                    st.dataframe(df_distractor, use_container_width=True)
                    
                    st.markdown("### 📊 DDI Summary by Item")
                    ddi_summary = []
                    for item in item_columns:
                        item_distractors = [r for r in distractor_results if r[0] == item]
                        if item_distractors:
                            ddi_values = [r[9] for r in item_distractors]
                            mean_ddi = np.mean(ddi_values)
                            functional_count = sum(1 for d in ddi_values if d > 0)
                            ddi_summary.append([item, len(item_distractors), round(mean_ddi, 4), functional_count])
                    if ddi_summary:
                        df_ddi_summary = pd.DataFrame(ddi_summary, columns=['Item', '# Distractors', 'Mean DDI', '# Functional'])
                        st.dataframe(df_ddi_summary, use_container_width=True)
                
                # Visualizations
                st.markdown("---")
                st.markdown("## 📊 VISUALIZATIONS")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(7, 4))
                    colors_p = ['red' if x < difficult_threshold else ('green' if x <= easy_threshold else 'orange') for x in p_values]
                    ax1.bar(range(1, n_items+1), p_values, color=colors_p)
                    ax1.axhline(difficult_threshold, color='red', linestyle='--')
                    ax1.axhline(easy_threshold, color='orange', linestyle='--')
                    ax1.set_xlabel('Item')
                    ax1.set_ylabel('p')
                    ax1.set_title('Item Difficulty')
                    ax1.set_xticks(range(1, n_items+1))
                    st.pyplot(fig1)
                    plt.close()
                
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(7, 4))
                    colors_d = ['green' if x >= good_threshold else ('orange' if x >= poor_threshold else 'red') for x in d_values]
                    ax2.bar(range(1, n_items+1), d_values, color=colors_d)
                    ax2.axhline(good_threshold, color='green', linestyle='--')
                    ax2.axhline(poor_threshold, color='orange', linestyle='--')
                    ax2.set_xlabel('Item')
                    ax2.set_ylabel('D')
                    ax2.set_title('Item Discrimination')
                    ax2.set_xticks(range(1, n_items+1))
                    st.pyplot(fig2)
                    plt.close()
                
                # Download
                st.markdown("---")
                st.markdown("## 📥 DOWNLOAD")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_results.to_excel(writer, sheet_name='Item_Statistics', index=False)
                    if distractor_results:
                        df_distractor.to_excel(writer, sheet_name='Distractor_Analysis', index=False)
                    pd.DataFrame({'KR-20': [kr20], 'SEM': [sem], 'Students': [n_students], 'Items': [n_items]}).to_excel(writer, sheet_name='Summary', index=False)
                output.seek(0)
                st.download_button("📥 Download Excel Report", data=output, file_name="item_analysis_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                st.success("✅ Analysis complete!")

else:
    with tab2:
        st.info("👈 Please upload CSV and answer key files in the 'Upload Data' tab")
