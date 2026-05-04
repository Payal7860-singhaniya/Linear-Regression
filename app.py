
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("📊 Interactive Linear Regression Learning System")

# =========================
# 1. DATASET UPLOAD
# =========================
st.header("1. Dataset Upload")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.write("Shape:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)

    # =========================
    # 2. PREPROCESSING
    # =========================
    st.header("2. Preprocessing")

    if st.checkbox("Handle Missing Values"):
        st.write("Using Mean Imputation")
        df = df.fillna(df.mean(numeric_only=True))

    if st.checkbox("Normalize Data"):
        df = (df - df.min()) / (df.max() - df.min())
        st.write("Data Normalized")

    st.write("Processed Data:")
    st.write(df.head())

    # =========================
    # 3. EDA
    # =========================
    st.header("3. Exploratory Data Analysis")

    col1 = st.selectbox("Select Feature (X)", df.columns)
    col2 = st.selectbox("Select Target (Y)", df.columns)

    st.subheader("Scatter Plot")
    fig, ax = plt.subplots()
    ax.scatter(df[col1], df[col2])
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    st.pyplot(fig)

    st.subheader("Correlation Matrix")
    st.write(df.corr(numeric_only=True))

    # =========================
    # 4. LEARNING MODULE
    # =========================
    st.header("4. Linear Regression Learning")

    st.subheader("Hypothesis Function")
    st.latex(r"y = b_0 + b_1 x")
    st.write("""
    - y → predicted value  
    - b₀ → intercept  
    - b₁ → slope (coefficient)  
    - x → input feature  
    """)

    st.subheader("Cost Function (Mean Squared Error)")
    st.latex(r"MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2")
    st.write("Lower MSE means better model performance.")

    # =========================
    # 5. MODEL TRAINING
    # =========================
    st.header("5. Model Training")

    X = df[[col1]]
    y = df[col2]

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.write("Coefficient:", model.coef_[0])
    st.write("Intercept:", model.intercept_)

    # =========================
    # 6. REGRESSION VISUALIZATION
    # =========================
    st.header("6. Regression Visualization")

    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y)
    ax2.plot(X, model.predict(X))
    ax2.set_xlabel(col1)
    ax2.set_ylabel(col2)
    st.pyplot(fig2)

    # =========================
    # 7. PREDICTION
    # =========================
    st.header("7. Prediction")

    input_value = st.number_input("Enter new value for prediction")

    if st.button("Predict"):
        prediction = model.predict(np.array([[input_value]]))

        st.write("Prediction:", prediction[0])

        st.write("Step Calculation:")
        st.write(f"y = {model.intercept_:.3f} + {model.coef_[0]:.3f} × {input_value}")

    # =========================
    # 8. EVALUATION METRICS
    # =========================
    st.header("8. Evaluation Metrics")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("MSE:", mse)
    st.write("MAE:", mae)
    st.write("R2 Score:", r2)

    # Model Interpretation
    if r2 > 0.8:
        st.success("Excellent Model Performance")
    elif r2 > 0.5:
        st.warning("Moderate Model Performance")
    else:
        st.error("Poor Model Performance")

    # =========================
    # 9. RESIDUAL PLOT
    # =========================
    st.subheader("Residual Plot")

    residuals = y_test - y_pred
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_pred, residuals)
    ax3.axhline(y=0)
    ax3.set_xlabel("Predicted Values")
    ax3.set_ylabel("Residuals")
    st.pyplot(fig3)

    # =========================
    # 10. DOWNLOAD DATA
    # =========================
    st.header("10. Download Processed Dataset")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")

