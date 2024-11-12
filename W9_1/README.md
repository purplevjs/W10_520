# Loan Approval Prediction App

This is a simple Streamlit application that predicts loan approval based on user input. The app uses a decision tree classifier trained on a synthetic dataset for demonstration purposes.

## Prerequisites

- Docker Desktop
- Git

## Getting Started

Follow these steps to run the Loan Approval Prediction App on your local machine:

1. Clone the repository: git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction

2. Build the Docker image: docker build -t demo-app .

3. Run the Docker container: docker run -p 5000:5000 demo-app

4. Open your web browser and go to: http://0.0.0.0:5000


## Using the App

1. Select your gender from the dropdown menu.
2. Choose your marital status.
3. Select your education level.
4. Enter your applicant income.
5. Click the "Predict" button to see the loan approval prediction.

## Project Structure

- `model.py`: Contains the Streamlit app code and the machine learning model.
- `requirements.txt`: Lists all Python dependencies.
- `Dockerfile`: Defines the Docker image for the app.
- `loan_prediction.csv`: The dataset used for training the model (not used in this demo version).

## Disclaimer

This app uses a simplified model for demonstration purposes only. It should not be used for actual loan decisions.
