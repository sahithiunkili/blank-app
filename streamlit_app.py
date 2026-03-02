import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

# -----------------------------------------
# 1. PAGE CONFIGURATION & STATE MANAGEMENT
# -----------------------------------------
st.set_page_config(page_title="Wealth AI - Client Insights", page_icon="📈", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'persona' not in st.session_state:
    st.session_state['persona'] = None

# -----------------------------------------
# 2. CACHED MACHINE LEARNING PIPELINE
# -----------------------------------------
@st.cache_resource
def load_and_train_models():
    """Loads data, builds the preprocessing pipeline, and trains Engines A & B."""
    # 1. Load Data
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    
    # Simulate Client IDs for the UI
    df['client_id'] = [f"CLI-{10000 + i}" for i in range(len(df))]
    
    # 2. Preprocessing Setup (Dropping duration to prevent leakage)
    X = df.drop(columns=['y', 'duration', 'client_id'])
    y = df['y'].map({'yes': 1, 'no': 0})
    
    numeric_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                        'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                            'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # 3. Train Engine A (K-Means)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    kmeans.fit(X_processed)
    
    # 4. Train Engine B (XGBoost)
    majority_class_count = y.value_counts()[0]
    minority_class_count = y.value_counts()[1]
    imbalance_ratio = majority_class_count / minority_class_count
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, 
        scale_pos_weight=imbalance_ratio, random_state=42, eval_metric='logloss'
    )
    xgb_model.fit(X_processed, y)
    
    return df, preprocessor, kmeans, xgb_model, numeric_features, categorical_features

# -----------------------------------------
# 3. BUSINESS LOGIC
# -----------------------------------------
def generate_client_insights(cluster_id, propensity_score):
    segment_map = {
        0: "🟡 Low-Rate Environment Professional",
        1: "🟡 High-Rate Environment Professional",
        2: "🟠 Traditional Retail Saver",
        3: "🟢 Highly Engaged / Proven Converter",
        4: "🔴 High-Friction / Past Reject"
    }
    
    current_segment = segment_map.get(cluster_id, "Unknown Segment")
    
    if propensity_score >= 0.75:
        trajectory = "High likelihood of immediate conversion."
        action = "Send aggressive TFSA Top-Up push notification."
        requires_human = False
        color = "green"
    elif 0.40 <= propensity_score < 0.75:
        trajectory = "On the fence; requires nurturing."
        action = "Send educational email series on tax-loss harvesting."
        requires_human = False
        color = "orange"
    else:
        trajectory = "High risk of fatigue or churn if pushed."
        color = "red"
        if cluster_id == 4: 
            action = "DO NOT CONTACT. Flag for relationship manager review."
            requires_human = True
        else:
            action = "Maintain dormant state. No action recommended."
            requires_human = False
            
    return current_segment, trajectory, action, requires_human, color

# -----------------------------------------
# 4. LOGIN SCREEN UI
# -----------------------------------------
if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center;'>Wealth AI Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Native Client Segmentation Engine</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            st.text_input("Username", value="admin")
            st.text_input("Password", type="password", value="password")
            persona_choice = st.selectbox("Select Persona", 
                                          ["Marketing Operator", "Relationship Manager", "Compliance Officer"])
            submit = st.form_submit_button("Secure Login")
            
            if submit:
                st.session_state['logged_in'] = True
                st.session_state['persona'] = persona_choice
                st.rerun()

# -----------------------------------------
# 5. MAIN DASHBOARD UI
# -----------------------------------------
else:
    # Load Models & Data
    with st.spinner("Initializing AI Models..."):
        df, preprocessor, kmeans, xgb_model, num_cols, cat_cols = load_and_train_models()
    
    # Sidebar
    st.sidebar.title("Wealth AI Dashboard")
    st.sidebar.info(f"**Logged in as:**\n{st.session_state['persona']}")
    
    # Client Selector (Simulating scrolling a CRM)
    # Let's pick a mix of known yes/no clients for a good demo
    demo_clients = df.iloc[[0, 100, 35000, 40000, 41180]]['client_id'].tolist()
    selected_client_id = st.sidebar.selectbox("Search Client ID", demo_clients)
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['persona'] = None
        st.rerun()

    # Main Area
    st.title("Client Overview")
    
    # Get Client Data
    client_data = df[df['client_id'] == selected_client_id].iloc[0]
    
    # Display Basic CRM Info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Age", client_data['age'])
    col2.metric("Profession", str(client_data['job']).title())
    col3.metric("Education", str(client_data['education']).replace('.', ' ').title())
    col4.metric("Last Interaction", f"{client_data['pdays']} days ago" if client_data['pdays'] != 999 else "Never")
    
    st.divider()

    # AI Engine Execution
    st.subheader("🧠 AI Insights Engine")
    
    # Format client data for the model
    X_single = df[df['client_id'] == selected_client_id].drop(columns=['y', 'duration', 'client_id'])
    X_single_processed = preprocessor.transform(X_single)
    
    # Get Predictions
    cluster_id = kmeans.predict(X_single_processed)[0]
    propensity = xgb_model.predict_proba(X_single_processed)[0][1]
    
    # Get Business Logic
    segment, trajectory, action, human_veto, color = generate_client_insights(cluster_id, propensity)
    
    # Display AI Outputs
    ui_col1, ui_col2 = st.columns(2)
    
    with ui_col1:
        st.markdown("### Engine A: Current State")
        st.info(f"**{segment}**")
        
        st.markdown("### Engine B: Future Trajectory")
        if color == "green":
            st.success(f"Propensity Score: **{propensity * 100:.1f}%**\n\n{trajectory}")
        elif color == "orange":
            st.warning(f"Propensity Score: **{propensity * 100:.1f}%**\n\n{trajectory}")
        else:
            st.error(f"Propensity Score: **{propensity * 100:.1f}%**\n\n{trajectory}")
            
    with ui_col2:
        st.markdown("### ⚡ Next Best Action")
        st.write(action)
        
        # Displaying the Human Veto Logic dynamically based on the Persona logged in
        if human_veto:
            st.error("🚨 **COMPLIANCE HOLD: Human Veto Required**")
            
            if st.session_state['persona'] == "Marketing Operator":
                st.write("You do not have permission to execute this campaign. Please route to a Relationship Manager.")
                st.button("Route for Approval", disabled=False)
            else:
                st.write("Review client history before approving automated contact.")
                col_a, col_b = st.columns(2)
                col_a.button("Approve Exception", type="primary")
                col_b.button("Dismiss Nudge")
        else:
            st.success("✅ **Cleared for Automated Campaign**")
            st.button("Execute Action", type="primary")
