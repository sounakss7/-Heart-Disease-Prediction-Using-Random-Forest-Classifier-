
---
# Heart Disease Prediction Using Random Forest Classifier

## Introduction

This project involves building a machine learning model to predict heart disease using the **Random Forest Classifier**. The model is trained on a dataset containing various patient health metrics such as age, cholesterol levels, and maximum heart rate achieved. The goal of this project is to accurately predict whether a person is at risk of heart disease based on their medical data.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)

## Project Structure

```
|-- sk7_random_forest.ipynb  # Jupyter notebook containing the project code
|-- README.md                # Readme file
|-- heart.csv                # Dataset used for training and testing the model
```

## Dataset

The dataset used for this project is the **Heart Disease UCI Dataset**, which contains 14 features:

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
- Required Python libraries (listed below)

### Required Libraries
To run the project, install the following Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository and navigate to the project directory.
2. Load the Jupyter notebook `sk7_random_forest.ipynb`.
3. Ensure the dataset `heart.csv` is available in the same directory.
4. Run the notebook to train the Random Forest Classifier and evaluate the model.
5. Visualizations such as **Feature Importance** and **Confusion Matrix** will be generated in the notebook.

## Model Evaluation

The Random Forest Classifier was used to train the model, which splits the dataset into training and testing sets. Below are the evaluation metrics used:

- **Confusion Matrix**: Shows the true positives, true negatives, false positives, and false negatives.
- **Accuracy Score**: Measures the overall accuracy of the model.
- **Classification Report**: Provides precision, recall, and F1-score for each class (presence/absence of heart disease).

### Example Results:
- **Accuracy**: 99% on the test dataset.
- **Confusion Matrix**:
  
  | True Positives | True Negatives | False Positives | False Negatives |
  |----------------|----------------|-----------------|-----------------|
  | High           | Low            | Low             | Low             |

## Conclusion

This project demonstrates the use of a Random Forest Classifier for heart disease prediction. The model achieves high accuracy, making it a promising tool for identifying patients at risk of heart disease based on their medical data.

---

This `README.md` file provides a clear overview of the project, its structure, and how to use it. You can modify it as per your specific requirements.
