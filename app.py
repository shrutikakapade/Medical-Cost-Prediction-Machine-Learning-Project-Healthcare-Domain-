import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🏥 Medical Cost Predictor", layout="wide")

@st.cache(allow_output_mutation=True)  # ✅ FIXED: Cache warning
def load_data():
    """Generate realistic insurance data - NO CSV NEEDED!"""
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

@st.cache(allow_output_mutation=True)  # ✅ FIXED: Cache warning
def train_model(df):
    """Train model - returns immutable objects"""
    X = df.drop('charges', axis=1).copy()
    
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    X_copy = X.copy()  # ✅ FIXED: Avoid mutation
    X_copy['sex'] = le_sex.fit_transform(X['sex'])
    X_copy['smoker'] = le_smoker.fit_transform(X['smoker'])
    X_copy['region'] = le_region.fit_transform(X['region'])
    
    y = df['charges'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X_copy, y, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, (le_sex, le_smoker, le_region), float(mae), float(r2), list(X.columns)

# Load everything
df = load_data()
model, scaler, encoders, mae, r2, feature_names = train_model(df)

st.title("🏥 Medical Cost Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Select Page:", ["📈 Overview", "🔍 Insights", "🤖 Model", "🔮 Predict"])

if page == "📈 Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients", len(df))
    col2.metric("Avg Age", f"{df['age'].mean():.0f}")
    col3.metric("Avg Cost", f"${df['charges'].mean():,.0f}")
    col4.metric("Smokers %", f"{(df['smoker']=='yes').mean()*100:.0f}%")
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10))  # ✅ FIXED: No use_container_width

elif page == "🔍 Insights":
    st.header("🔍 Data Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="age", title="Age Distribution")
        st.plotly_chart(fig1)
    
    with col2:
        fig2 = px.histogram(df, x="charges", title="Charges Distribution")
        st.plotly_chart(fig2)
    
    # Smoker comparison
    smoker_avg = df.groupby('smoker')['charges'].mean().reset_index()
    fig3 = px.bar(smoker_avg, x='smoker', y='charges', title="Smoker vs Non-Smoker")
    st.plotly_chart(fig3)

elif page == "🤖 Model":
    st.header("🤖 Model Performance")
    
    col1, col2 = st.columns(2)
    col1.metric("MAE Error", f"${mae:,.0f}")
    col2.metric("R² Score", f"{r2:.3f}")
    
    # Feature importance
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=True)
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig)

elif page == "🔮 Predict":
    st.header("🔮 Predict Costs")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 64, 30)
        sex = st.selectbox("Sex", ['female', 'male'])
        bmi = st.slider("BMI", 15.0, 50.0, 30.0)
    
    with col2:
        children = st.slider("Children", 0, 5, 1)
        smoker = st.selectbox("Smoker", ['yes', 'no'])
        region = st.selectbox("Region", ['northeast', 'southeast', 'southwest', 'northwest'])
    
    if st.button("🚀 Predict Cost"):
        input_df = pd.DataFrame({
            'age': [age], 'sex': [sex], 'bmi': [bmi],
            'children': [children], 'smoker': [smoker], 'region': [region]
        })
        
        le_sex, le_smoker, le_region = encoders
        input_df['sex'] = le_sex.transform(input_df['sex'])
        input_df['smoker'] = le_smoker.transform(input_df['smoker'])
        input_df['region'] = le_region.transform(input_df['region'])
        
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        col1, col2 = st.columns(2)
        col1.success(f"**Predicted: ${pred:,.0f}**")
        col2.info(f"Average: ${df['charges'].mean():,.0f}")
        
        st.balloons()


