import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Title and description
st.title("House Price Prediction and Classification App")
st.markdown("""
This application predicts house prices (regression) and classifies whether prices are above or below the median (classification).
Select a model from the sidebar to view performance metrics and visualizations.
""")

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'data.csv' not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# Preprocessing
def preprocess_data(df):
    # Check for required column
    if "price" not in df.columns:
        st.error("Error: 'price' column is missing from the dataset.")
        st.stop()
    
    # Drop unnecessary columns
    df_cleaned = df.drop(columns=["date", "street", "city", "statezip", "country"], errors='ignore')
    
    # Check data size before dropping NA
    original_size = len(df_cleaned)
    df_cleaned = df_cleaned.dropna()
    if len(df_cleaned) == 0:
        st.error("Error: No data remains after removing missing values.")
        st.stop()
    if len(df_cleaned) < original_size * 0.5:
        st.warning(f"Warning: {original_size - len(df_cleaned)} rows ({(original_size - len(df_cleaned)) / original_size * 100:.1f}%) were removed due to missing values.")
    
    return df_cleaned

df_cleaned = preprocess_data(df)

# Create classification target (above/below median price)
median_price = df_cleaned["price"].median()
df_cleaned["price_class"] = (df_cleaned["price"] > median_price).astype(int)

# Feature-target split
X = df_cleaned.drop(["price", "price_class"], axis=1)
y_reg = df_cleaned["price"]
y_clf = df_cleaned["price_class"]

# Check for non-numeric features
non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
if len(non_numeric_cols) > 0:
    st.error(f"Error: Non-numeric columns found: {', '.join(non_numeric_cols)}. Please encode categorical variables.")
    st.stop()

# Single train-test split for consistency
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)  # Same split for classification

# Feature scaling (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

# Cache model training
@st.cache_resource
def train_model(model, X_train, y_train, model_name):
    model.fit(X_train, y_train)
    return model

# Train and evaluate selected model
results = {}
selected_model = models[model_name]
trained_model = train_model(selected_model, X_train_scaled, y_train_clf, model_name)
preds_clf = trained_model.predict(X_test_scaled)
probs_clf = trained_model.predict_proba(X_test_scaled)[:, 1]

# Classification metrics
accuracy = accuracy_score(y_test_clf, preds_clf)
f1 = f1_score(y_test_clf, preds_clf)
roc_auc = roc_auc_score(y_test_clf, probs_clf)
cm = confusion_matrix(y_test_clf, preds_clf)

results[model_name] = {
    "model": trained_model,
    "accuracy": accuracy,
    "f1": f1,
    "roc_auc": roc_auc,
    "cm": cm,
    "probs": probs_clf,
    "preds_clf": preds_clf
}

# Display results
st.header(f"Results for {model_name}")

# Classification metrics
st.subheader("Classification Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{results[model_name]['accuracy']:.4f}")
col2.metric("F1 Score", f"{results[model_name]['f1']:.4f}")
col3.metric("ROC AUC", f"{results[model_name]['roc_auc']:.4f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
sns.heatmap(results[model_name]['cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title(f"Confusion Matrix ({model_name})")
st.pyplot(fig_cm)
plt.close(fig_cm)

# ROC Curve
st.subheader("ROC Curve")
fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
fpr, tpr, _ = roc_curve(y_test_clf, results[model_name]['probs'])
ax_roc.plot(fpr, tpr, label=f"ROC Curve (AUC = {results[model_name]['roc_auc']:.4f})")
ax_roc.plot([0, 1], [0, 1], 'r--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title(f"ROC Curve ({model_name})")
ax_roc.legend()
ax_roc.grid(True)
st.pyplot(fig_roc)
plt.close(fig_roc)

# Regression model (using RandomForestRegressor for regression)
st.subheader("Regression Metrics (Random Forest)")
@st.cache_resource
def train_regressor(X_train, y_train):
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    return rf_reg

rf_reg = train_regressor(X_train_scaled, y_train_reg)
preds_reg = rf_reg.predict(X_test_scaled)
mse = mean_squared_error(y_test_reg, preds_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, preds_reg)

col1, col2, col3 = st.columns(3)
col1.metric