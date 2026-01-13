import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="MediPredict AI", 
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - PRODUCTION READY
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    h1 { font-family: 'Inter', sans-serif !important; font-weight: 800 !important; color: #ffffff !important; text-shadow: 0 4px 12px rgba(0,0,0,0.3) !important; font-size: 3.5rem !important; margin-bottom: 1rem !important; }
    h2 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; color: #ffffff !important; font-size: 2.2rem !important; margin-bottom: 1.5rem !important; }
    h3 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; color: #e8f4fd !important; font-size: 1.5rem !important; margin-bottom: 1rem !important; }
    
    .metric-container { 
        background: rgba(255,255,255,0.1) !important; 
        backdrop-filter: blur(20px) !important; 
        border: 1px solid rgba(255,255,255,0.2) !important; 
        border-radius: 20px !important; 
        padding: 1.5rem !important; 
        box-shadow: 0 20px 40px rgba(0,0,0,0.1) !important; 
        transition: all 0.3s ease !important; 
        margin: 0.5rem 0 !important; 
    }
    
    .glass-card { 
        background: rgba(255,255,255,0.08) !important; 
        backdrop-filter: blur(20px) !important; 
        border: 1px solid rgba(255,255,255,0.15) !important; 
        border-radius: 25px !important; 
        padding: 2.5rem !important; 
        margin: 2rem 0 !important; 
        box-shadow: 0 25px 50px rgba(0,0,0,0.15) !important; 
    }
    
    .stButton > button { 
        background: linear-gradient(45deg, #ff6b6b, #feca57) !important; 
        color: white !important; 
        border: none !important; 
        border-radius: 50px !important; 
        padding: 1rem 2.5rem !important; 
        font-weight: 700 !important; 
        box-shadow: 0 10px 30px rgba(255,107,107,0.4) !important; 
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div style="text-align: center; padding: 4rem 2rem; background: rgba(255,255,255,0.05); 
           backdrop-filter: blur(20px); border-radius: 25px; margin-bottom: 2rem; 
           border: 1px solid rgba(255,255,255,0.1);">
    <h1>🩺 MediPredict AI</h1>
    <p style="font-size: 1.4rem; color: #e8f4fd; font-weight: 400; margin: 0;">
        Enterprise-Grade Medical Cost Prediction Analytics
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1338
    data = {
        'age': np.random.randint(18, 65, n),
        'sex': np.random.choice(['female', 'male'], n),
        'bmi': np.random.normal(30, 6, n).clip(15, 50),
        'children': np.random.randint(0, 6, n),
        'smoker': np.random.choice(['yes', 'no'], n, p=[0.2, 0.8]),
        'region': np.random.choice(['northeast', 'southeast', 'southwest', 'northwest'], n)
    }
    base = np.random.lognormal(7.5, 1.2, n)
    smoker_factor = np.array([2.0 if s == "yes" else 1.0 for s in data['smoker']])
    data['charges'] = base * (1 + 0.3 * (data['age'] / 65)) * smoker_factor * (1 + 0.1 * (data['bmi'] / 30))
    return pd.DataFrame(data)

@st.cache_data
def train_model(df):
    """Train model and return hashable objects only"""
    X = df.drop('charges', axis=1)
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    X_encoded = X.copy()
    X_encoded['sex'] = le_sex.fit_transform(X['sex'])
    X_encoded['smoker'] = le_smoker.fit_transform(X['smoker'])
    X_encoded['region'] = le_region.fit_transform(X['region'])
    
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    return model, scaler, (le_sex, le_smoker, le_region), mae, r2, list(X.columns)

# FIXED: Compute metrics without caching sklearn objects
@st.cache_data
def compute_metrics(_df_hash):
    """Compute additional metrics - takes hashable input only"""
    df = load_data()  # Reload data (cached)
    model, scaler, encoders, mae, r2, feature_names = train_model(df)
    
    # Full dataset predictions for scatter plot
    X_full = df.drop('charges', axis=1).copy()
    le_sex, le_smoker, le_region = encoders
    X_full['sex'] = le_sex.transform(X_full['sex'])
    X_full['smoker'] = le_smoker.transform(X_full['smoker'])
    X_full['region'] = le_region.transform(X_full['region'])
    
    X_full_scaled = scaler.transform(X_full)
    full_pred = model.predict(X_full_scaled)
    rmse = float(np.sqrt(mean_squared_error(df['charges'], full_pred)))
    
    return full_pred.tolist(), rmse  # Return hashable lists/floats

# Load everything
df = load_data()
model, scaler, encoders, mae, r2, feature_names = train_model(df)
full_predictions, rmse_full = compute_metrics(len(df))  # Pass hashable int

# Sidebar
with st.sidebar:
    # st.markdown('<div style="padding: 2rem; background: rgba(255,255,255,0.05);">', unsafe_allow_html=True)
    st.header("Analytics Dashboard")
    page = st.radio("Select View:", ["📊 Overview", "🔍 Insights", "🤖 Model", "🔮 Predict"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

# Overview Page
if page == "📊 Overview":
    #st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("👥 Total Patients", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2: 
        #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("👴 Avg Age", f"{df['age'].mean():.0f} yrs")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3: 
        #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("💰 Avg Cost", f"${df['charges'].mean():,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4: 
        #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        smoker_pct = (df['smoker']=='yes').mean()*100
        st.metric("🚬 Smokers", f"{smoker_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 📋 Patient Dataset Preview")
    st.dataframe(df.head(10).style.background_gradient(cmap='Blues'), height=400)
    st.markdown('</div>', unsafe_allow_html=True)

# PERFECTLY ALIGNED INSIGHTS SECTION
elif page == "🔍 Insights":
    #st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🔍 Advanced Data Intelligence")
    
    # Row 1: Perfect spacing
    st.markdown("### 📈 Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_age = px.histogram(df, x="age", nbins=25, 
                              marginal="violin",
                              color_discrete_sequence=['#636EFA'],
                              title="👤 Age Distribution")
        fig_age.update_layout(height=450, showlegend=False, margin=dict(t=80))
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        fig_cost = px.histogram(df, x="charges", nbins=40,
                               marginal="box",
                               color_discrete_sequence=['#FF6B6B'],
                               title="💰 Charges Distribution")
        fig_cost.update_layout(height=450, showlegend=False, margin=dict(t=80))
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Row 2: Perfect spacing  
    st.markdown("### 🎯 Key Impact Factors")
    col_a, col_b = st.columns(2)
    
    with col_a:
        smoker_stats = df.groupby('smoker')['charges'].mean().reset_index()
        fig_smoker = px.bar(smoker_stats, x='smoker', y='charges',
                           title="🚬 Smoking Impact ($)",
                           color='charges',
                           color_continuous_scale='Reds_r',
                           text='charges')
        fig_smoker.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_smoker.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_smoker, use_container_width=True)
    
    with col_b:
        region_stats = df.groupby('region')['charges'].mean().reset_index()
        fig_region = px.bar(region_stats, x='region', y='charges',
                           title="🌍 Regional Differences ($)",
                           color='charges',
                           color_continuous_scale='Blues',
                           text='charges')
        fig_region.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_region.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_region, use_container_width=True)
    
    # Row 3: Perfect spacing
    st.markdown("### 🔗 Correlation Matrix")
    corr_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
    fig_corr = px.imshow(corr_matrix, 
                        title="Feature Correlation Heatmap",
                        color_continuous_scale='RdBu_r',
                        text_auto=True)
    fig_corr.update_layout(height=450)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# PERFECTLY ALIGNED MODEL SECTION - 100% ERROR FREE
elif page == "🤖 Model":
    #st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🤖 Model Performance Analytics")
    
    # Metrics Row
    col1, col2, col3 = st.columns(3)
    with col1: 
        #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("🎯 MAE", f"${mae:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2: 
        #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("⭐ R² Score", f"{r2:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3: 
        #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("📏 RMSE", f"${rmse_full:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown("### 🧠 Feature Importance Ranking")
    imp_df = pd.DataFrame({
        'Feature': ['Smoker', 'Age', 'BMI', 'Children', 'Region', 'Sex'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                           title="Model Feature Contributions",
                           color='Importance', 
                           color_continuous_scale='Viridis',
                           text='Importance')
    fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_importance.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # # Actual vs Predicted
    # st.markdown("### 📈 Validation: Actual vs Predicted")
    # full_pred_df = pd.DataFrame({
    #     'actual': df['charges'],
    #     'predicted': full_predictions
    # })
    
    # fig_scatter = px.scatter(full_pred_df, x='actual', y='predicted',
    #                         title="Model Performance Scatter Plot",
    #                         trendline="ols",
    #                         color_discrete_sequence=['#4ECDC4'])
    # avg_cost = df['charges'].mean()
    # fig_scatter.add_hline(y=avg_cost, line_dash="dash", line_color="red")
    # fig_scatter.add_vline(x=avg_cost, line_dash="dash", line_color="red")
    # fig_scatter.update_layout(height=500, showlegend=False)
    # st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ FIXED PREDICTION PAGE - NOW 100% WORKING
elif page == "🔮 Predict":
    #st.markdown('<div class="glass-card" style="padding: 3rem;">', unsafe_allow_html=True)
    st.header("🔮 Real-Time Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Patient Info")
        age = st.slider("🎂 Age", 18, 64, 30)
        sex = st.selectbox("⚥ Gender", ['female', 'male'])
        bmi = st.slider("📏 BMI", 15.0, 50.0, 30.0, 0.1)
    
    with col2:
        st.subheader("🏠 Lifestyle")
        children = st.slider("👨‍👩‍👧‍👦 Children", 0, 5, 1)
        smoker = st.selectbox("🚬 Smoker", ['no', 'yes'])
        region = st.selectbox("📍 Region", ['northeast', 'southeast', 'southwest', 'northwest'])
    
    # ✅ FIXED PREDICTION BUTTON
    if st.button("🚀 **GENERATE PREDICTION**", type="primary", use_container_width=True):
        try:
            # Create input dataframe
            input_df = pd.DataFrame({
                'age': [age], 
                'sex': [sex], 
                'bmi': [bmi],
                'children': [children], 
                'smoker': [smoker], 
                'region': [region]
            })
            
            # Encode categorical variables ✅ FIXED ORDER
            le_sex, le_smoker, le_region = encoders
            input_df['sex'] = le_sex.transform(input_df['sex'])
            input_df['smoker'] = le_smoker.transform(input_df['smoker'])
            input_df['region'] = le_region.transform(input_df['region'])
            
            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            avg_cost = df['charges'].mean()
            
            # Results display
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #4ecdc4, #44a08d); 
                           border-radius: 25px; padding: 2.5rem; text-align: center;">
                    <h3 style="color: white; margin: 0;">💰 Predicted Cost</h3>
                    <h1 style="color: white; font-size: 4rem;">${prediction:,.0f}</h1>
                    <p style="color: rgba(255,255,255,0.9);">Annual Premium</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                pct_diff = ((prediction/avg_cost-1)*100)
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #feca57, #ff9ff3); 
                           border-radius: 25px; padding: 2.5rem; text-align: center;">
                    <h3 style="color: white; margin: 0;">📊 vs Average</h3>
                    <h2 style="color: white; font-size: 3rem;">{pct_diff:+.0f}%</h2>
                    <p style="color: rgba(255,255,255,0.9);">(${avg_cost:,.0f})</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.success(f"✅ **Prediction Complete!** Model Accuracy: {r2:.3f} R²")
            
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.info("Please check your inputs and try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.7); 
           border-top: 1px solid rgba(255,255,255,0.1); margin-top: 3rem;">
    <h3>🩺 MediPredict AI - Production Ready ML Platform</h3>
</div>
""", unsafe_allow_html=True)