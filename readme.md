# Diabetes Prediction Web App

A user-friendly web application to predict the likelihood of diabetes in patients, built with **Streamlit** and **Logistic Regression** (using the Pima Indians Diabetes Dataset). Users can input their medical parameters, and the model will instantly indicate whether they are likely to have diabetes.

## Table of Contents

- [Features](#features)
- [Project Overview](#project-overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Screenshots](#screenshots)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Interactive Web Interface**: Enter patient data via input boxes.
- **Instant Prediction**: Model outputs "Diabetic" or "Not Diabetic".
- **Preprocessing Included**: Handles missing data and feature scaling.
- **End-to-End ML Pipeline**: Model trained, saved, and used seamlessly.
- **Easy Deployment**: Just run and use locally or deploy to the cloud.

## Project Overview

This project applies a supervised binary classification model (logistic regression) to real-world medical data in order to:

- Explore the link between patient metrics (e.g., blood pressure, insulin, BMI) and diabetes.
- Provide an accessible interface for real-time, model-driven medical advice (not for clinical use).
- Serve as a template for deploying machine learning models as web applications.

## Demo

![Demo GIF or Image Placeholder]

- **Source**: [Pima Indians Diabetes Dataset](https://www.openml.org/d/37)
- **Features Used**:
  - Number of Pregnancies
  - Plasma Glucose Concentration
  - Diastolic Blood Pressure
  - Triceps Skinfold Thickness
  - Serum Insulin
  - BMI (Body Mass Index)
  - Diabetes Pedigree Function
  - Age

## Machine Learning Workflow

1. **Data Exploration & Preprocessing**
   - Replacement of missing/invalid values
   - Feature scaling using StandardScaler

2. **Model Training**
   - Logistic Regression (scikit-learn)
   - Evaluation with accuracy, confusion matrix, and classification report

3. **Model Saving**
   - Trained model and scaler saved with `joblib`

4. **Web Deployment**
   - Streamlit UI for data entry and instant prediction

## Requirements

- Python 3.7+
- streamlit
- scikit-learn
- pandas
- numpy
- joblib

(See [requirements.txt](requirements.txt) in repo.)

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-app.git
   cd diabetes-prediction-app
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Retrain the Model**
   - If you want to retrain, run:
     ```bash
     python train_model.py
     ```
   - This creates/updates `diabetes_model.pkl` and `scaler.pkl`.

4. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

5. **Using the App**
   - Open the local address provided in your browser.
   - Enter patient data and click "Predict" to see the result.

## File Structure

```
diabetes-prediction-app/
│
├── app.py                # Streamlit web application
├── train_model.py        # Model training script
├── diabetes_model.pkl    # Trained logistic regression model
├── scaler.pkl            # Saved StandardScaler object
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── demo.gif              # (Optional) Demo of web app
```

## Screenshots

*(Insert images or gifs showing the app interface and output example)*

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)
- [OpenML Diabetes Dataset](https://www.openml.org/d/37)
- Streamlit and scikit-learn developers

## Disclaimer

This tool is for educational purposes only and **should not be used for real medical diagnosis or treatment decisions**.

**Happy Predicting!**