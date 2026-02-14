import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("🚨 PAM: Production Anomaly Monitor")
st.markdown("**Upload factory CSV → Detect risks early**")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data loaded:", df.shape)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = numeric_cols[:5]
    st.write(f"Analyzing: {list(features)}")
    
    X = df[features].dropna()
    if len(X) < 10:
        st.error("Need 10+ rows")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = IsolationForest(contamination=0.03, random_state=42)
        model.fit(X_scaled)
        
        anomalies = model.predict(X_scaled)
        df.loc[df[features].index, 'anomaly'] = anomalies
        
        count = (anomalies == -1).sum()
        pct = count / len(anomalies) * 100
        st.metric("🚨 Anomalies Found", count, f"{pct:.1f}%")
        
        fig = px.scatter(df, x=features[0], y=features[1], 
                         color='anomaly', 
                         title="Anomalies (Red = Risk)",
                         color_discrete_map={1: 'green', -1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
        
        df.to_csv('results.csv', index=False)
        st.download_button("📥 Download Results", data=open('results.csv','rb'), 
                          file_name='anomalies_detected.csv')
        st.success("✅ PAM ready!")
