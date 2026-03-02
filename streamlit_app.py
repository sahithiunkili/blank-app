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
        font-weight: 600
