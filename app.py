import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# Load environment variables from .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(layout="wide")
st.title("📊 Data Exploration and Analysis Tool")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Display first 5 rows
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Display basic info
    st.write("### Data Summary")
    st.write(df.describe())

    # Generate Automated EDA Report
    if st.button("🔍 Generate Basic Data Profile"):
        with st.spinner("Generating profile... ⏳"):
            # Display data types
            st.write("#### Data Types")
            st.write(df.dtypes)
            
            # Display missing values
            st.write("#### Missing Values")
            missing = df.isnull().sum()
            st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values")
            
            # Display unique values for each column
            st.write("#### Unique Values")
            for col in df.columns:
                unique_count = df[col].nunique()
                st.write(f"{col}: {unique_count} unique values")

    # Visualization Section
    st.write("### 📊 Data Visualizations")

    # Select numerical columns for visualization
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if num_cols:
        selected_column = st.selectbox("Choose a numeric column:", num_cols)

        # Histogram
        fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
        st.plotly_chart(fig)

        # Correlation Heatmap (Only if more than one numeric column exists)
        if len(num_cols) > 1 and st.button("📈 Show Correlation Heatmap"):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")

# OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")

if openai_api_key and 'df' in locals():
    os.environ["OPENAI_API_KEY"] = openai_api_key  # Ensure it's set in the environment
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)

    st.write("### 🤖 AI-Powered Data Insights")
    query = st.text_input("Ask a question about the dataset:")
    if query:
        prompt = PromptTemplate(
            input_variables=["data", "query"],
            template="Given the dataset:\n{data}\n\nAnswer this question:\n{query}"
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(data=df.head().to_string(), query=query)

        st.write("### 🧠 GPT Insights")
        st.write(response)

st.write("### 🚀 Machine Learning Models")

if 'df' in locals():
    # Machine Learning Section
    target = st.selectbox("Select target variable:", df.columns)
    
    if st.button("🔍 Train Models"):
        # Prepare the data
        is_classification = df[target].dtype == 'object' or df[target].nunique() < 10
        
        if is_classification:
            # Convert categorical target to numeric
            le = LabelEncoder()
            y = le.fit_transform(df[target])
            st.write(f"Training classification models (detected {df[target].nunique()} classes)")
        else:
            y = df[target]
            st.write("Training regression models")
        
        # Handle categorical features
        X = pd.get_dummies(df.drop(columns=[target]))
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with st.spinner("Training models... ⏳"):
            results = []
            
            if is_classification:
                # Train classification models
                models = {
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "SVC": SVC(random_state=42)
                }
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    results.append((name, accuracy, "Accuracy"))
            else:
                # Train regression models
                models = {
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Linear Regression": LinearRegression(),
                    "SVR": SVR()
                }
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    results.append((name, mse, "MSE"))
                    results.append((name, r2, "R²"))
            
            # Display results
            results_df = pd.DataFrame(results, columns=["Model", "Score", "Metric"])
            st.write("### Model Performance")
            st.dataframe(results_df)
            
            # Plot results
            fig = px.bar(results_df, x="Model", y="Score", color="Metric", 
                         title="Model Performance Comparison",
                         barmode="group")
            st.plotly_chart(fig)