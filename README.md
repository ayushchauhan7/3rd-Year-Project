# Credit Card Fraud Detection System

This project aims to develop a machine learning-based system to detect fraudulent credit card transactions. Utilizing algorithms like XGBoost, the system analyzes transaction data to identify anomalies indicative of fraud.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [License](#license)

## Overview

The Credit Card Fraud Detection System leverages machine learning techniques to classify transactions as legitimate or fraudulent. By training on historical transaction data, the model learns patterns associated with fraudulent activities, enabling real-time detection and prevention.

## Features

- Data preprocessing and scaling
- Model training using XGBoost classifier
- Serialization of trained model and scaler for deployment
- Web interface for user interaction and prediction
- Jupyter Notebook for exploratory data analysis and visualization

## Prerequisites

Ensure you have the following installed:

- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ayushchauhan7/3rd-Year-Project.git
   cd 3rd-Year-Project
   ```


2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```


3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```


   *Note: If `requirements.txt` is not present, manually install the necessary packages:*

   ```bash
   pip install pandas numpy scikit-learn xgboost flask
   ```


## Usage

1. **Data Preprocessing:**

   ```bash
   python create_scaler.py
   ```


   This script preprocesses the data and saves the scaler object for future use.

2. **Model Training:**

   ```bash
   python train_model.py
   ```


   Trains the XGBoost model and saves it as `xgboost_model.pkl`.

3. **Running the Application:**

   ```bash
   python app.py
   ```


   Starts the Flask web application. Navigate to `http://localhost:5000` in your browser to interact with the system.

## Project Structure


```plaintext
3rd-Year-Project/
├── Credit_Card_Fraud_Detection_System.ipynb  # Jupyter Notebook for EDA
├── README.md                                 # Project documentation
├── app.py                                    # Flask application
├── create_scaler.py                          # Data preprocessing script
├── model.py                                  # Model definition (if applicable)
├── scaler.pkl                                # Saved scaler object
├── splitpy.py                                # Data splitting script
├── train_model.py                            # Model training script
├── xgboost_model.pkl                         # Trained XGBoost model
```


## Contributors

- Ayush Chauhan
- Abhay Tyagi
- Adarsh Kumar
- Mohammad Noorul Hoda

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
