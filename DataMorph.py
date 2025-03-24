import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import io

# Function for Data Preprocessing
def data_preprocessing_pipeline(data):
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    # Handle missing values
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())
    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

    # Handle outliers using IQR
    for feature in numeric_features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[feature] = np.clip(data[feature], lower_bound, upper_bound)

    # Normalize numeric features
    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    return data

# Streamlit UI
st.set_page_config(page_title="📊 Data Preprocessing Dashboard", layout="wide")

# Center-align the title
st.markdown("<h1 style='text-align: center;'>🔍 DataMorph</h1>", unsafe_allow_html=True)

# Sidebar for File Upload
st.sidebar.header("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File Uploaded Successfully!")

    # Display Original Data
    st.subheader("📊 Original Data")
    st.write(df.head())

    # Process Data Button
    if st.button("🚀 Process Data"):
        processed_df = data_preprocessing_pipeline(df.copy())
        st.subheader("✅ Processed Data")
        st.write(processed_df.head())

        # Download Processed Data
        csv_buffer = io.StringIO()
        processed_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download Processed Data",
            data=csv_data,
            file_name="processed_data.csv",
            mime="text/csv"
        )

        # Dashboard Visualization
        st.subheader("📈 Data Insights & Visualization")

        # Convert categorical to numeric for heatmap
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Correlation Heatmap
        st.write("### 🔥 Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

        # Histogram for Numeric Features
        st.write("### 📊 Histograms of Numeric Features")
        for col in processed_df.select_dtypes(include=['float64', 'int64']).columns:
            fig = px.histogram(processed_df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.sidebar.warning("⚠️ Please upload a CSV file to proceed.")

# Footer
st.markdown("<footer style='text-align: center;'>Developed by CodexCasper © 2025</footer>", unsafe_allow_html=True)
