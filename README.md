# sales-prediction-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import time  # Import the time library

# Load the advertising data
data = pd.read_csv("data/advertising.csv")

# Create a linear regression model
X = data[["TV"]]
y = data["Sales"]
model = LinearRegression()
model.fit(X, y)

# Streamlit app
def main():
    st.title("Advertising Analysis")
    st.write("This app explores the relationship between TV advertising and sales.")

    # Sidebar for user input
    st.sidebar.header("Choose Advertising Budget")
    tv_budget = st.sidebar.slider("TV Budget ($)", min_value=0, max_value=300, step=10)

    # Predict sales based on TV budget
    predicted_sales = model.predict([[tv_budget]])

    # Display results
    st.write(f"TV Budget: ${tv_budget}")
    st.write(f"Predicted Sales: {predicted_sales[0]:.2f}")

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data["TV"], y=data["Sales"])
    plt.xlabel("TV Budget ($)")
    plt.ylabel("Sales")
    plt.title("TV Budget vs. Sales")
    plt.axvline(x=tv_budget, color="r", linestyle="--", label=f"TV Budget: ${tv_budget}")
    plt.legend()
    st.pyplot(plt)

# Rerun the app every 5 hours
while True:
    main()
    time.sleep(5 * 60 * 60)  # Sleep for 5 hours (in seconds)
