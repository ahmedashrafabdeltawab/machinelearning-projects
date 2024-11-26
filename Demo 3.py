import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# App title
st.title("Sales Prediction System")

# Upload dataset
uploaded_file = st.file_uploader(r"C:\Users\Ahmed Ashraf\Desktop\archive (3)\data1.csv")
if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)
    st.write("**Initial Data Preview:**")
    st.dataframe(data.head())

    # Show columns in the data for user to choose from
    st.write("**Available columns:**", data.columns)
    column_to_predict = st.selectbox("Select a column to predict:", data.columns)

    # Exploratory Data Analysis (EDA)
    st.write("**Data Analysis:**")
    if column_to_predict in data.columns:
        st.bar_chart(data[column_to_predict])  # Visualizing the selected column
    else:
        st.warning(f"Column '{column_to_predict}' not found in your data!")

    # Prediction model
    if st.button("Run Prediction"):
        if column_to_predict in data.columns:
            # Handle missing values
            # Identify numeric columns
            numeric_cols = data.select_dtypes(include=['number']).columns

            # Fill missing values in numeric columns with their mean
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

            # For categorical columns, fill missing values with the most frequent value
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                data[col].fillna(data[col].mode()[0], inplace=True)

            # Convert categorical columns to numeric using One-Hot Encoding
            data = pd.get_dummies(data)

            # Prepare data for the model
            X = data.drop(columns=[column_to_predict])  # Features
            y = data[column_to_predict]  # Target column

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Display predictions
            st.write("**Predictions:**")
            st.line_chart(predictions)

            # Display the model's performance (R^2 and MSE)
            st.write("**Model Performance:**")
            st.write(f"Test Set Score (R^2): {model.score(X_test, y_test)}")

            # Calculate Mean Squared Error (MSE)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Mean Squared Error (MSE): {mse}")

            # Optionally, plot the actual vs predicted values
            st.write("**Actual vs Predicted Values:**")
            fig, ax = plt.subplots()
            ax.plot(y_test.values, label="Actual Values", color="blue")
            ax.plot(predictions, label="Predicted Values", color="red")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning(f"Cannot run prediction as the column '{column_to_predict}' is missing!")
