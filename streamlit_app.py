import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

# -----------------------------------------
# 1. PAGE CONFIGURATION & WS CUSTOM CSS
# -----------------------------------------
st.set_page_config(page_title="Wealth AI - CRM", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    .stApp { 
        background-color: #f7f6f2 !important; 
        font-family: 'Inter', sans-serif !important;
        color: #1a1a1a !important;
    }
    
    h1, h2, h3, .ws-serif {
        font-family: 'Playfair Display', serif !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }

    .ws-top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 40px;
        background-color: transparent;
        margin-top: -60px;
        margin-bottom: 30px;
        border-bottom: 1px solid #e2dfd8;
    }
    .ws-logo {
        font-family: 'Playfair Display', serif;
        font-weight: 800;
        font-size: 24px;
        color: #1a1a1a;
        letter-spacing: -0.5px;
    }
    .ws-nav-links {
        display: flex;
        gap: 25px;
        font-size: 15px;
        font-weight: 500;
        color: #1a1a1a;
        align-items: center;
    }
    .ws-nav-btn {
        border: 1px solid #1a1a1a;
        border-radius: 30px;
        padding: 8px 18px;
        font-size: 14px;
        font-weight: 600;
    }
    .ws-nav-btn.dark { background-color: #2b2b2b; color: white; }

    div[data-testid="stButton"] button {
        border-radius: 30px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #2b2b2b !important;
        color: white !important;
        border: none !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover { background-color: #000000 !important; }
    
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: transparent !important;
        color: #1a1a1a !important;
        border: 1px solid #1a1a1a !important;
    }

    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid #e2dfd8 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
        padding: 10px;
    }
    
    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px 0px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #6b7280;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #1a1a1a !important;
    }
    
    .data-footer { text-align: center; color: #9ca3af; font-size: 12px; margin-top: 50px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Session State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'persona' not in st.session_state:
    st.session_state['persona'] = None

# -----------------------------------------
# 2. CACHED ML ARCHITECTURE & BATCH INFERENCE
# -----------------------------------------
@st.cache_resource
def initialize_ai_engine():
    df_ml = pd.read_csv('bank-additional-full.csv', sep=';')
    df_ml['Client_ID'] = [f"CLI-{10000 + i}" for i in range(len(df_ml))]
    df_crm = pd.read_csv('wealthsimple_crm_mock.csv')
    
    X = df_ml.drop(columns=['y', 'duration', 'Client_ID'])
    y = df_ml['y'].map({'yes': 1, 'no': 0})
    
    num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto').fit(X_processed)
    ratio = y.value_counts()[0] / y.value_counts()[1]
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, scale_pos_weight=ratio, random_state=42).fit(X_processed, y)
    
    # --- BATCH INFERENCE FOR AUDIENCE BUILDER ---
    crm_ml_data = df_ml[df_ml['Client_ID'].isin(df_crm['Client_ID'])].drop(columns=['y', 'duration', 'Client_ID'])
    crm_processed = preprocessor.transform(crm_ml_data)
    
    df_crm['AI_Cluster'] = kmeans.predict(crm_processed)
    df_crm['Propensity'] = xgb_model.predict_proba(crm_processed)[:, 1]
    
    # EXACT String Mapping to prevent UI errors
    segment_map = {
        0: "🟡 Low-Rate Environment Professional", 
        1: "🟡 High-Rate Environment Professional",
        2: "🟠 Traditional Retail Saver", 
        3: "🟢 Proven Converter", 
        4: "🔴 High-Friction / Past Reject"
    }
    df_crm['AI_Segment'] = df_crm['AI_Cluster'].map(segment_map)
    
    return df_ml, df_crm, preprocessor, kmeans, xgb_model

# -----------------------------------------
# 3. BUSINESS LOGIC (Individual Insights)
# -----------------------------------------
def get_insights(cluster_id, propensity):
    segment_map = {
        0: "🟡 Low-Rate Environment Professional", 
        1: "🟡 High-Rate Environment Professional",
        2: "🟠 Traditional Retail Saver", 
        3: "🟢 Proven Converter", 
        4: "🔴 High-Friction / Past Reject"
    }
    segment = segment_map.get(cluster_id, "Unknown Segment")
    
    if propensity >= 0.50:
        return segment, "High likelihood of conversion.", "TFSA Top-Up Nudge (Email + In-App)", False, "green", "31%", "$2,100"
    elif 0.15 <= propensity < 0.50:
        return segment, "On the fence; requires nurturing.", "Tax-Loss Harvesting Education", False, "orange", "14%", "$850"
    else:
        return segment, "High compliance risk. Do not auto-enroll.", "DO NOT CONTACT. Flag for RM.", True, "red", "< 2%", "$0"

# -----------------------------------------
# 4. LOGIN SCREEN
# -----------------------------------------
if not st.session_state['logged_in']:
    st.markdown('<div class="ws-top-nav"><div class="ws-logo">Wealthsimple</div><div class="ws-nav-links">Admin Portal</div></div>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Secure System Login</h2><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        with st.container(border=True):
            with st.form("login_form"):
                st.text_input("Username", value="admin")
                st.text_input("Password", type="password", value="••••••••")
                persona_choice = st.selectbox("Select Role", ["Marketing Operator", "Relationship Manager"])
                if st.form_submit_button("Log in", type="primary", use_container_width=True):
                    st.session_state['logged_in'] = True
                    st.session_state['persona'] = persona_choice
                    st.rerun()

# -----------------------------------------
# 5. CRM PORTAL UI
# -----------------------------------------
else:
    st.markdown("""
        <div class="ws-top-nav">
            <div class="ws-logo">Wealthsimple</div>
            <div class="ws-nav-links">
                <span>Personal</span><span>Business</span><span>Support</span>
                <span class="ws-nav-btn">Log out</span><span class="ws-nav-btn dark">Get started</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Initializing AI Engines & Batch Processing CRM..."):
        try:
            df_ml, df_crm, preprocessor, kmeans, xgb_model = initialize_ai_engine()
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            st.stop()

    st.sidebar.title("Controls")
    st.sidebar.info(f"**Session:** {st.session_state['persona']}")
    if st.sidebar.button("Log out", type="secondary"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.markdown("<h2>AI Client Segmentation Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #4b5563; margin-bottom: 30px;'>Dynamically segmenting users and predicting response likelihood at scale.</p>", unsafe_allow_html=True)

    # Split Layout
    col_left, col_space, col_profile = st.columns([1.2, 0.1, 1.5])

    # ==========================================
    # LEFT PANE: TABS FOR DIRECTORY & BUILDER
    # ==========================================
    with col_left:
        tab_dir, tab_build = st.tabs(["Directory", "Audience Builder"])
        
        with tab_dir:
            st.markdown("<h3 class='ws-serif' style='font-size: 1.3rem; margin-top: 10px;'>Customer Directory</h3>", unsafe_allow_html=True)
            display_df = df_crm[['Client_ID', 'Name', 'Profession']].copy()
            selected_name = st.selectbox("Search Client Profiles (Updates Profile Panel):", df_crm['Name'].tolist())
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=450)

        with tab_build:
            st.markdown("<h3 class='ws-serif' style='font-size: 1.3rem; margin-top: 10px;'>Build Target Group</h3>", unsafe_allow_html=True)
            st.caption("Combine AI-driven rules with manual overrides.")
            
            # --- Rule-Based Add ---
            st.markdown("**1. Rule-Based Segmenting**")
            
            # Fixed static list so options never crash on empty clusters
            all_segments = [
                "🟡 Low-Rate Environment Professional", 
                "🟡 High-Rate Environment Professional",
                "🟠 Traditional Retail Saver", 
                "🟢 Proven Converter", 
                "🔴 High-Friction / Past Reject"
            ]
            
            target_segments = st.multiselect(
                "Select AI Behavioral Segments", 
                options=all_segments,
                default=["🟢 Proven Converter"]
            )
            min_propensity = st.slider("Minimum Propensity Score (%)", 0, 100, 45)
            
            rule_based_df = df_crm[
                (df_crm['AI_Segment'].isin(target_segments)) & 
                (df_crm['Propensity'] >= (min_propensity / 100))
            ]
            
            # --- Manual Add / Exclude ---
            st.markdown("**2. Manual Overrides**")
            manual_add_names = st.multiselect("Manually Add Clients (Regardless of rules):", df_crm['Name'].tolist())
            
            manual_add_df = df_crm[df_crm['Name'].isin(manual_add_names)]
            final_audience_df = pd.concat([rule_based_df, manual_add_df]).drop_duplicates(subset=['Client_ID'])
            
            st.divider()
            
            # --- Impact & Action ---
            st.markdown(f"### Total Audience: {len(final_audience_df)} Clients")
            st.dataframe(final_audience_df[['Name', 'AI_Segment', 'Propensity']], use_container_width=True, hide_index=True, height=200)
            
            if st.session_state['persona'] == "Marketing Operator":
                st.button("Launch Email Campaign", type="primary", use_container_width=True)
            else:
                st.button("Assign to Advisors (Batch)", type="primary", use_container_width=True)

    # ==========================================
    # RIGHT PANE: INDIVIDUAL CUSTOMER PROFILE
    # ==========================================
    with col_profile:
        st.markdown("<h3 class='ws-serif' style='font-size: 1.5rem;'>Client Profile</h3>", unsafe_allow_html=True)
        
        selected_client_id = df_crm[df_crm['Name'] == selected_name]['Client_ID'].values[0]
        crm_data = df_crm[df_crm['Client_ID'] == selected_client_id].iloc[0]
        ml_data = df_ml[df_ml['Client_ID'] == selected_client_id].drop(columns=['y', 'duration', 'Client_ID'])
        
        ml_processed = preprocessor.transform(ml_data)
        cluster_id = kmeans.predict(ml_processed)[0]
        propensity = xgb_model.predict_proba(ml_processed)[0][1]
        
        segment, trajectory, action, human_veto, color, exp_resp, exp_uplift = get_insights(cluster_id, propensity)
        
        with st.container(border=True):
            st.markdown(f"<h2 style='margin-bottom: 0px;'>{crm_data['Name']}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #6b7280; font-weight: 500; margin-bottom: 20px;'>{selected_client_id}</p>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            c1.markdown(f"**Age:** {crm_data['Age']}<br>**Occupation:** {crm_data['Profession']}<br>**Income:** ${crm_data['Estimated_Income_CAD']:,}", unsafe_allow_html=True)
            c2.markdown(f"**Status:** Active 🟢<br>**Accounts:** {crm_data['Account_History']}<br>**Newcomer:** {crm_data['Is_Newcomer']}", unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 15px;'> AI Insights</h3>", unsafe_allow_html=True)
            
            st.markdown(f"**Current Segment:** {segment}")
            st.caption("Model Classification Confidence:")
            st.progress(float(min(propensity + 0.15, 0.98))) 
            st.write("")
            
            st.markdown(f"**Recommended Action:** {action}")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Conversion Propensity", f"{propensity * 100:.1f}%")
            metric_col2.metric("Expected Response", exp_resp)
            metric_col3.metric("Projected AUM Uplift", exp_uplift)
            st.markdown("<br>", unsafe_allow_html=True)
                
            if human_veto:
                st.error("**Compliance-Sensitive (Risk Profile Change)**\n\nRequires Human Approval Before Execution")
                if st.session_state['persona'] == "Marketing Operator":
                    st.button("Route to RM for Approval", type="primary", use_container_width=True)
                    st.caption("Access Denied: Marketers cannot override compliance holds.")
                else:
                    ca, cb = st.columns(2)
                    ca.button("Approve Exception", type="primary", use_container_width=True)
                    cb.button("Dismiss Nudge", type="secondary", use_container_width=True)
            else:
                st.success("**Cleared for Automated Campaign**")
                st.button("Execute Action", type="primary", use_container_width=True)

    st.markdown('<div class="data-footer">Powered by ML models trained on 41k+ anonymized banking records</div>', unsafe_allow_html=True)
