import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------
st.set_page_config(page_title="Wealth AI - CRM Portal", layout="wide", initial_sidebar_state="collapsed")

# Injecting CSS for the Top Nav and CRM styling
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fc; }
    
    /* Custom Top Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        background-color: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 30px;
        margin-top: -60px; /* Pulls it up to the very top of the Streamlit canvas */
    }
    .nav-icons { font-size: 22px; color: #4b5563; cursor: pointer; position: relative;}
    .nav-center {
        background-color: #e5e7eb;
        padding: 6px 20px;
        border-radius: 20px;
        font-weight: 600;
        color: #1f2937;
        font-size: 15px;
    }
    .nav-right { display: flex; gap: 20px; }
    .dot {
        height: 8px; width: 8px; background-color: #ef4444;
        border-radius: 50%; display: inline-block;
        position: absolute; top: 0px; right: 0px;
    }
    
    /* CRM Header */
    .crm-header { background-color: #1e3a8a; color: white; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'persona' not in st.session_state:
    st.session_state['persona'] = None

# -----------------------------------------
# 2. LOAD DECOUPLED ML ARCHITECTURE
# -----------------------------------------
@st.cache_resource
def load_architecture():
    """Loads the pre-trained .joblib models from the repository."""
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        kmeans = joblib.load('engine_a_kmeans.joblib')
        xgb_model = joblib.load('engine_b_xgboost.joblib')
        return preprocessor, kmeans, xgb_model, True
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data
def load_data():
    """Loads and aligns the datasets."""
    df_ml = pd.read_csv('bank-additional-full.csv', sep=';')
    df_ml['Client_ID'] = [f"CLI-{10000 + i}" for i in range(len(df_ml))]
    df_crm = pd.read_csv('wealthsimple_crm_mock.csv')
    return df_ml, df_crm

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
    st.markdown('<div class="crm-header"><h2>🔒 Wealth AI - Secure System Login</h2></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.container(border=True):
            with st.form("login_form"):
                st.text_input("Username", value="admin")
                st.text_input("Password", type="password", value="••••••••")
                persona_choice = st.selectbox("Select Role", ["Marketing Operator", "Relationship Manager"])
                if st.form_submit_button("Authenticate", use_container_width=True):
                    st.session_state['logged_in'] = True
                    st.session_state['persona'] = persona_choice
                    st.rerun()

# -----------------------------------------
# 5. CRM PORTAL UI
# -----------------------------------------
else:
    # 1. Render the Custom Top Nav Bar
    st.markdown("""
        <div class="top-nav">
            <div class="nav-icons">🔔<span class="dot"></span></div>
            <div class="nav-center">Home</div>
            <div class="nav-right">
                <span class="nav-icons">🎁</span>
                <span class="nav-icons">👤</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Load Backend
    preprocessor, kmeans, xgb_model, models_loaded = load_architecture()
    if getattr(models_loaded, 'startswith', lambda x: False)('Error'):
        st.error(f"Model Load Error. Ensure Scikit-Learn versions match. Details: {models_loaded}")
        st.stop()
        
    df_ml, df_crm = load_data()

    # Sidebar controls
    st.sidebar.title("Controls")
    st.sidebar.info(f"**Session:** {st.session_state['persona']}")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.markdown('<div class="crm-header"><h2>💼 CRM Portal - Welcome, Admin</h2></div>', unsafe_allow_html=True)

    # 3. Split Layout
    col_list, col_space, col_profile = st.columns([1.2, 0.1, 1.5])

    with col_list:
        st.markdown("### Customer List")
        
        display_df = df_crm[['Client_ID', 'Name', 'Profession']].copy()
        
        # Select client to populate the right pane
        selected_name = st.selectbox("Quick Search:", df_crm['Name'].tolist())
        selected_client_id = df_crm[df_crm['Name'] == selected_name]['Client_ID'].values[0]
        
        # Show interactive dataframe
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=550)

    with col_profile:
        st.markdown("### Customer Profile")
        
        crm_data = df_crm[df_crm['Client_ID'] == selected_client_id].iloc[0]
        ml_data = df_ml[df_ml['Client_ID'] == selected_client_id].drop(columns=['y', 'duration', 'Client_ID'])
        
        # Run Real-Time Inference
        ml_processed = preprocessor.transform(ml_data)
        cluster_id = kmeans.predict(ml_processed)[0]
        propensity = xgb_model.predict_proba(ml_processed)[0][1]
        segment, trajectory, action, human_veto, color = get_insights(cluster_id, propensity)
        
        # Profile Card Design
        with st.container(border=True):
            # Top profile section
            st.markdown(f"### 👤 {crm_data['Name']}")
            c1, c2 = st.columns(2)
            c1.markdown(f"**Age:** {crm_data['Age']}<br>**Occupation:** {crm_data['Profession']}<br>**Income:** ${crm_data['Estimated_Income_CAD']:,}", unsafe_allow_html=True)
            c2.markdown(f"**Status:** Active 🟢<br>**Accounts:** {crm_data['Account_History']}<br>**Newcomer:** {crm_data['Is_Newcomer']}", unsafe_allow_html=True)
            
            st.divider()
            
            # AI Insights Section
            st.markdown("#### 🧠 AI Engine Insights")
            st.info(f"**Current Segment:** {segment}")
            
            if color == "green":
                st.success(f"**Propensity Score:** {propensity * 100:.1f}%\n\n**Action:** {action}")
            elif color == "orange":
                st.warning(f"**Propensity Score:** {propensity * 100:.1f}%\n\n**Action:** {action}")
            else:
                st.error(f"**Propensity Score:** {propensity * 100:.1f}%\n\n**Action:** {action}")
                
            # Governance / RBAC Buttons
            if human_veto:
                st.error("🚨 **COMPLIANCE HOLD: Human Veto Required**")
                if st.session_state['persona'] == "Marketing Operator":
                    st.button("Route to RM for Approval", type="primary", use_container_width=True)
                    st.caption("Access Denied: Marketers cannot override compliance holds.")
                else:
                    ca, cb = st.columns(2)
                    ca.button("Approve Exception", type="primary", use_container_width=True)
                    cb.button("Dismiss Nudge", use_container_width=True)
            else:
                st.success("✅ **Cleared for Automated Campaign**")
                st.button("Execute Action", type="primary", use_container_width=True)
