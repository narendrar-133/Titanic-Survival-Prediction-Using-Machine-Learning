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
