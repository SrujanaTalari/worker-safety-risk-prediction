import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataload import load_data
from preprocessing import preprocess_data
from model import train_model
from featureimportance import get_feature_importance
from evaluation import evaluate_model

# Page setup
st.set_page_config(page_title="New Delhi Construction Safety Risk", layout="wide")
st.title("🏗️ New Delhi Construction Worker Safety Risk Prediction")

# Load Dataset
df = load_data()
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# Download Dataset
st.download_button(
    label="⬇ Download Dataset (CSV)",
    data=df.to_csv(index=False),
    file_name="new_delhi_construction_safety.csv",
    mime="text/csv"
)

# Preprocess and Train
X_train, X_test, y_train, y_test = preprocess_data(df)
model = train_model(X_train, y_train)

# Evaluation
results = evaluate_model(model, X_test, y_test)
st.subheader("📊 Model Evaluation")
st.write(f"**Accuracy:** {results['accuracy']:.2f}")
st.text("Classification Report:")
st.text(results["report"])

# Confusion Matrix
st.subheader("🧾 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(results["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Feature Importance
st.subheader("🌟 Feature Importance")
importance_df = get_feature_importance(model, X_train)
st.dataframe(importance_df)

# Vertical Bar Chart
st.subheader("📊 Vertical Bar Chart")
fig, ax = plt.subplots()
sns.barplot(x="Feature", y="Importance", data=importance_df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Line Chart
st.subheader("📈 Line Chart")
fig, ax = plt.subplots()
sns.lineplot(x="Feature", y="Importance", data=importance_df, marker="o", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Pie Chart
st.subheader("🥧 Pie Chart")
fig, ax = plt.subplots()
ax.pie(
    importance_df["Importance"],
    labels=importance_df["Feature"],
    autopct="%1.1f%%",
    startangle=90
)
ax.axis("equal")
st.pyplot(fig)

# Histogram
st.subheader("📉 Histogram")
fig, ax = plt.subplots()
sns.histplot(importance_df["Importance"], bins=10, kde=True, ax=ax)
st.pyplot(fig)