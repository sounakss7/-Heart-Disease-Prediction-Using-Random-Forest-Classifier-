---
# Heart Disease Prediction Using Random Forest Classifier

## Introduction

This project aims to predict heart disease risk using a **Random Forest Classifier**. The model analyzes patient health metrics such as **age, cholesterol levels, and maximum heart rate achieved** to determine the likelihood of heart disease. The application is deployed on **Streamlit Cloud** for easy access.

🔗 **Live Deployment:** [Heart Disease Prediction App](https://heartdieaseuppy-nv4tmjre98gmrnhvgkhcxa.streamlit.app/)

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Conclusion](#conclusion)

## Project Structure
```
|-- 📂 Heart-Disease-Prediction
    |-- 📄 heartdieaseup.py           # Main Streamlit app for cloud deployment
    |-- 📄 sk7_random_forest.ipynb    # Jupyter notebook for training and evaluation
    |-- 📄 README.md                  # Documentation
    |-- 📄 heart.csv                   # UCI Heart Disease Dataset
    |-- 📂 models                      # Trained models (if applicable)
    |-- 📂 images                      # Visualizations (Confusion Matrix, Feature Importance)
```

## Dataset

The dataset used is the **UCI Heart Disease Dataset**, which contains 14 features:

- **age**: Age of the patient
- **sex**: Gender of the patient
- **cp**: Chest pain type
- **trestbps**: Resting blood pressure
- **chol**: Serum cholesterol level
- **fbs**: Fasting blood sugar
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy
- **thal**: Thalassemia
- **target**: 1 = presence of heart disease, 0 = absence of heart disease

## Installation

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required Python libraries

### Required Libraries
To install the required dependencies, run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit
```

## Usage

### 1️⃣ Run the Notebook Locally (For Training & Testing)
1. Clone the repository:
   ```bash
   git clone https://github.com/sounakss7/-Heart-Disease-Prediction-Using-Random-Forest-Classifier-
   ```
2. Navigate to the project directory:
   ```bash
   cd Heart-Disease-Prediction
   ```
3. Open Jupyter Notebook and run `sk7_random_forest.ipynb` to train and evaluate the model.
4. Visualizations such as **Feature Importance** and **Confusion Matrix** will be generated.

### 2️⃣ Run the Streamlit App Locally
1. Execute the following command to run the Streamlit app:
   ```bash
   streamlit run heartdieaseup.py
   ```
2. Open `http://localhost:8501/` in your browser to interact with the app.

## Model Evaluation

The **Random Forest Classifier** was trained and evaluated using various metrics:

- **Confusion Matrix**: Displays True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
- **Accuracy Score**: Measures overall model accuracy.
- **Precision, Recall & F1-Score**: Evaluates classification performance.

### Example Results:
- **Accuracy**: ~99% on the test dataset.
- **Confusion Matrix Visualization:**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```

## Deployment

The application is deployed on **Streamlit Cloud** for easy access.

### Deployment Steps
1. **Prepare the App**: Ensure `heartdieaseup.py` contains the Streamlit UI logic.
2. **Push to GitHub**: Ensure all required files are committed.
3. **Deploy to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Select `heartdieaseup.py` as the entry point
   - Deploy!

## Conclusion

This project successfully applies **Random Forest Classification** for **heart disease prediction**. The **Streamlit app** provides an intuitive interface for users to interact with the model, making early heart disease detection more accessible. Future improvements may include **hyperparameter tuning**, **feature selection**, and **integration with real-world medical records**.

---


