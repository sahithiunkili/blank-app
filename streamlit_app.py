import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

# -----------------------------------------
# 1. PAGE CONFIGURATION & WS CUSTOM CSS
# -----------------------------------------
st.set_page_config(page_title="Wealth AI - CRM", layout="wide", initial_sidebar_state="collapsed")

# Injecting Wealthsimple-inspired CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    /* Main Background & Text */
    .stApp { 
        background-color: #f7f6f2 !important; /* WS Warm Off-White */
        font-family: 'Inter', sans-serif !important;
        color: #1a1a1a !important;
    }
    
    /* Headers - Premium Serif */
    h1, h2, h3, .ws-serif {
        font-family: 'Playfair Display', serif !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }

    /* Custom Top Navigation Bar */
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

    /* Buttons (Pill-shaped like WS "Register Now") */
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
    div[data-testid="stButton"] button[kind="secondary"]:hover { background-color: #e2dfd8 !important; }

    /* Cards & Containers */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid #e2dfd8 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
        padding: 10px;
    }
    
    /* Hide Streamlit branding elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'persona' not in st.session_state:
    st.session_state['persona'] = None

# -----------------------------------------
# 2. BULLETPROOF CACHED ML ARCHITECTURE
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
    
    return df_ml, df_crm, preprocessor, kmeans, xgb_model

# -----------------------------------------
# 3. BUSINESS LOGIC
# -----------------------------------------
def get_insights(cluster_id, propensity):
    segments = {
        0: "🟡 Low-Rate Environment Professional",
        1: "🟡 High-Rate Environment Professional",
        2: "🟠 Traditional Retail Saver",
        3: "🟢 Proven Converter",
        4: "🔴 High-Friction / Past Reject"
    }
    segment = segments.get(cluster_id, "Unknown Segment")
    
    if propensity >= 0.75:
        return segment, "High likelihood of immediate conversion.", "Send aggressive TFSA Top-Up push notification.", False, "green"
    elif 0.40 <= propensity < 0.75:
        return segment, "On the fence; requires nurturing.", "Send educational email series on tax-loss harvesting.", False, "orange"
    else:
        if cluster_id == 4:
            return segment, "High risk of fatigue or churn if pushed.", "DO NOT CONTACT. Flag for RM review.", True, "red"
        return segment, "High risk of fatigue or churn if pushed.", "Maintain dormant state. No action recommended.", False, "red"

# -----------------------------------------
# 4. LOGIN SCREEN
# -----------------------------------------
if not st.session_state['logged_in']:
    st.markdown('<div class="ws-top-nav"><div class="ws-logo">Wealthsimple</div><div class="ws-nav-links">Admin Portal</div></div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center;'>Secure System Login</h2>", unsafe_allow_html=True)
    st.write("")
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
    # 1. Custom Top Nav Bar
    st.markdown("""
        <div class="ws-top-nav">
            <div class="ws-logo">Wealthsimple</div>
            <div class="ws-nav-links">
                <span>Personal</span>
                <span>Business</span>
                <span>Support</span>
                <span class="ws-nav-btn">Log out</span>
                <span class="ws-nav-btn dark">Get started</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Load Backend
    with st.spinner("Initializing AI Engines..."):
        try:
            df_ml, df_crm, preprocessor, kmeans, xgb_model = initialize_ai_engine()
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            st.stop()

    # Sidebar controls
    st.sidebar.title("Controls")
    st.sidebar.info(f"**Session:** {st.session_state['persona']}")
    if st.sidebar.button("Log out", type="secondary"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.markdown("<h2>AI Client Segmentation Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #4b5563; margin-bottom: 30px;'>Dynamically segmenting users and predicting response likelihood at scale.</p>", unsafe_allow_html=True)

    # 3. Split Layout
    col_list, col_space, col_profile = st.columns([1.2, 0.1, 1.5])

    with col_list:
        st.markdown("<h3 class='ws-serif' style='font-size: 1.5rem;'>Customer Directory</h3>", unsafe_allow_html=True)
        
        display_df = df_crm[['Client_ID', 'Name', 'Profession']].copy()
        
        selected_name = st.selectbox("Search Client Profiles:", df_crm['Name'].tolist())
        selected_client_id = df_crm[df_crm['Name'] == selected_name]['Client_ID'].values[0]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=550)

    with col_profile:
        st.markdown("<h3 class='ws-serif' style='font-size: 1.5rem;'>Client Profile</h3>", unsafe_allow_html=True)
        
        crm_data = df_crm[df_crm['Client_ID'] == selected_client_id].iloc[0]
        ml_data = df_ml[df_ml['Client_ID'] == selected_client_id].drop(columns=['y', 'duration', 'Client_ID'])
        
        # Run Real-Time Inference
        ml_processed = preprocessor.transform(ml_data)
        cluster_id = kmeans.predict(ml_processed)[0]
        propensity = xgb_model.predict_proba(ml_processed)[0][1]
        segment, trajectory, action, human_veto, color = get_insights(cluster_id, propensity)
        
        # Profile Card Design
        with st.container(border=True):
            st.markdown(f"<h2 style='margin-bottom: 0px;'>{crm_data['Name']}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #6b7280; font-weight: 500; margin-bottom: 20px;'>{selected_client_id}</p>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            c1.markdown(f"**Age:** {crm_data['Age']}<br>**Occupation:** {crm_data['Profession']}<br>**Income:** ${crm_data['Estimated_Income_CAD']:,}", unsafe_allow_html=True)
            c2.markdown(f"**Status:** Active 🟢<br>**Accounts:** {crm_data['Account_History']}<br>**Newcomer:** {crm_data['Is_Newcomer']}", unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>🧠 AI Insights</h3>", unsafe_allow_html=True)
            st.info(f"**Current Segment:** {segment}")
            
            if color == "green":
                st.success(f"**Propensity Score:** {propensity * 100:.1f}%\n\n**Action:** {action}")
            elif color == "orange":
                st.warning(f"**Propensity Score:** {propensity * 100:.1f}%\n\n**Action:** {action}")
            else:
                st.error(f"**Propensity Score:** {propensity * 100:.1f}%\n\n**Action:** {action}")
                
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
                
            # Governance / RBAC Buttons
            if human_veto:
                st.error("🚨 **COMPLIANCE HOLD: Human Veto Required**")
                if st.session_state['persona'] == "Marketing Operator":
                    st.button("Route to RM for Approval", type="primary", use_container_width=True)
                    st.caption("Access Denied: Marketers cannot override compliance holds.")
                else:
                    ca, cb = st.columns(2)
                    ca.button("Approve Exception", type="primary", use_container_width=True)
                    cb.button("Dismiss Nudge", type="secondary", use_container_width=True)
            else:
                st.success("✅ **Cleared for Automated Campaign**")
                st.button("Execute Action", type="primary", use_container_width=True)
