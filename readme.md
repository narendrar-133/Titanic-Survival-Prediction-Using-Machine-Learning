# Titanic Survival Prediction using LGBMClassifier (Inbuilt + Custom Model)

This project applies both an **inbuilt LightGBM LGBMClassifier** and a **custom-implemented LGBM-style classifier** to predict passenger survival on the Titanic dataset. The objective is to understand how boosting-based classification works by comparing a real-world optimized model with a manually coded version.

The repository includes the dataset, model files, complete code, and training notebook.

---

## 📌 Overview

This project uses:

### 🔹 Inbuilt LGBMClassifier (LightGBM)
- High-performance gradient boosting classifier  
- Handles numerical and categorical features efficiently  
- Fast training and strong predictive performance  
- Saved as `lgbm_inbuilt_model.pkl`

### 🔹 Custom LGBM-Style Model
- A manually coded boosting classifier that mimics LightGBM’s behavior  
- Useful for understanding how boosting works internally  
- Saved as `custom_lgbm_model.pkl`

Both models are trained on the same dataset to compare behavior and predictions.

---

## 📊 Dataset

The dataset used is the Titanic training dataset containing:

- Passenger class  
- Name, Age, Sex  
- Number of siblings/spouses aboard  
- Number of parents/children aboard  
- Ticket, Fare, Cabin, Embarked  
- **Survived** (target variable)

The dataset is included in:

data/train.csv

---

## 🧠 Models Included

Trained models are stored in:

models/
├── inbuilt_LGBM_classifier_model.pkl
└── custom_LGBM_classifier_model.pkl

hey can be loaded directly without retraining.

---

## 🗂️ Repository Structure

project/
│
├── data/
│ └── train.csv
│
├── models/
│ ├── inbuilt_LGBM_classifier_model.pkl
│ └── custom_LGBM_classifier_model.pkl
│
├── notebooks/
│ └── training.ipynb
│
├── requirements.txt
└── README.md

---

## 🚀 How to Use

### Install dependencies:
pip install -r requirements.txt


### Load inbuilt LGBM model:
```python
import joblib
model = joblib.load("models/lgbm_inbuilt_model.pkl")
predictions = model.predict(X_test)
```
###Load custom model:
```python
import pickle
with open("models/custom_lgbm_model.pkl", "rb") as f:
    custom_model = pickle.load(f)

preds = custom_model.predict(X_test)
```

##🙌 Author

This project demonstrates both practical machine learning using LightGBM and a deeper understanding of boosting through a custom model. Feel free to explore, use, or contribute!
