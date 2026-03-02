import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------
st.set_page_config(page_title="Wealth AI - CRM Portal", layout="wide", initial_sidebar_state="collapsed")

# Injecting CSS to mimic the clean, white-card CRM look from your mockup
st.markdown("""
    <style>
    .stApp { background-color: #f4f6f9; }
    .css-1d391kg { padding-top: 1rem; }
    .crm-header { background-color: #1e3a5f; color: white; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px;}
    .profile-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 2. CACHED ARCHITECTURE LOADING
# -----------------------------------------
@st.cache_resource
def load_models():
    """Loads the pre-trained architecture from disk."""
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        kmeans = joblib.load('engine_a_kmeans.joblib')
        xgb_model = joblib.load('engine_b_xgboost.joblib')
        return preprocessor, kmeans, xgb_model, True
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data
def load_data():
    """Loads and aligns the ML dataset with the CRM mock data."""
    # 1. Load ML Data
    df_ml = pd.read_csv('bank-additional-full.csv', sep=';')
    df_ml['Client_ID'] = [f"CLI-{10000 + i}" for i in range(len(df_ml))]
    
    # 2. Load CRM Presentation Data
    df_crm = pd.read_csv('wealthsimple_crm_mock.csv')
    return df_ml, df_crm

# Initialize connection
preprocessor, kmeans, xgb_model, models_loaded = load_models()
df_ml, df_crm = load_data()

if getattr(models_loaded, 'startswith', lambda x: False)('Error'):
    st.error(f"Failed to load models. Ensure .joblib files are in the repository. Error: {models_loaded}")
    st.stop()

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
# 4. CRM PORTAL UI
# -----------------------------------------
# Header mapping to the mockup
st.markdown('<div class="crm-header"><h2>💼 Wealth AI CRM Portal - Welcome, Admin</h2></div>', unsafe_allow_html=True)

# Layout Split (Left: CRM List, Right: Profile Card)
col_list, col_space, col_profile = st.columns([1.5, 0.1, 1.2])

with col_list:
    st.markdown("### Customer List")
    
    # Create a nice display dataframe for the left side
    display_df = df_crm[['Client_ID', 'Name', 'Profession', 'Estimated_Income_CAD', 'Account_History']].copy()
    display_df.columns = ['ID', 'Client Name', 'Occupation', 'Est. Income', 'Current Accounts']
    
    # We use a selectbox right above the table to ensure rock-solid interactivity during the demo
    selected_name = st.selectbox("Search Client Profiles:", df_crm['Name'].tolist())
    selected_client_id = df_crm[df_crm['Name'] == selected_name]['Client_ID'].values[0]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

with col_profile:
    st.markdown("### Customer Profile")
    
    # Fetch unified client data
    crm_data = df_crm[df_crm['Client_ID'] == selected_client_id].iloc[0]
    ml_data = df_ml[df_ml['Client_ID'] == selected_client_id].drop(columns=['y', 'duration', 'Client_ID'])
    
    # Run Inference!
    ml_processed = preprocessor.transform(ml_data)
    cluster_id = kmeans.predict(ml_processed)[0]
    propensity = xgb_model.predict_proba(ml_processed)[0][1]
    
    segment, trajectory, action, human_veto, color = get_insights(cluster_id, propensity)
    
    # --- Profile Card UI (Matching the mockup) ---
    with st.container(border=True):
        st.markdown(f"## 👤 {crm_data['Name']}")
        
        c1, c2 = st.columns(2)
        c1.markdown(f"**Age:** {crm_data['Age']}<br>**Occupation:** {crm_data['Profession']}<br>**Income:** ${crm_data['Estimated_Income_CAD']:,}", unsafe_allow_html=True)
        c2.markdown(f"**Accounts:** {crm_data['Account_History']}<br>**Newcomer:** {crm_data['Is_Newcomer']}", unsafe_allow_html=True)
        
        st.divider()
        
        # --- AI Insights Block ---
        st.markdown("#### 🧠 AI Engine Insights")
        st.info(f"**Current Segment:** {segment}")
        
        if color == "green":
            st.success(f"**Propensity Score:** {propensity * 100:.1f}%\n\n**Action:** {action}")
        elif color == "orange":
            st.warning(f"**Propensity Score:** {propensity * 100:.1f}%\n\n**Action:** {action}")
        else:
            st.error(f"**Propensity Score:** {propensity * 100:.1f}%\n\n**Action:** {action}")
            
        if human_veto:
            st.error("🚨 **COMPLIANCE HOLD: Human Veto Required**")
            ca, cb = st.columns(2)
            ca.button("Approve Exception", type="primary", use_container_width=True)
            cb.button("Dismiss Nudge", use_container_width=True)
        else:
            st.success("✅ **Cleared for Automated Campaign**")
            st.button("Execute Action", type="primary", use_container_width=True)
