import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Example function to load model and scaler
def load_model_and_scaler():
    # Load the trained XGBoost model
    model = joblib.load('xgboost_model.pkl')

    # Load the StandardScaler
    scaler = joblib.load('scaler.pkl')

    return model, scaler

# Example function to preprocess data
def preprocess_data(data, scaler):
    # Example: scaling numerical features
    numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                          'oldbalanceDest', 'newbalanceDest']
    data[numerical_features] = scaler.transform(data[numerical_features])

    return data

if __name__ == '__main__':
    # Example usage to load model and scaler
    model, scaler = load_model_and_scaler()

    # Example usage to preprocess data
    data = pd.DataFrame({
        'step': [100],
        'type': ['TRANSFER'],
        'amount': [10000],
        'oldbalanceOrg': [50000],
        'newbalanceOrig': [40000],
        'oldbalanceDest': [0],
        'newbalanceDest': [0],
        'errorBalanceOrig': [0],
        'errorBalanceDest': [10000]
    })

    # Drop columns not used in scaling
    data = data.drop(columns=['errorBalanceOrig', 'errorBalanceDest'])
    # One-hot encode the 'type' column
    data = pd.get_dummies(data, columns=['type'])

    # Ensure all expected columns are present
    expected_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER', 'isFlaggedFraud']
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    # Preprocess the data
    data_processed = preprocess_data(data, scaler)

    # Example prediction using the loaded model
    prediction = model.predict(data_processed)

    print("Prediction:", prediction)