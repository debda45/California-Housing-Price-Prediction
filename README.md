# 🏡 California Housing Price Prediction: Statistical Inference & Modeling

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20|%20XGBoost%20|%20SHAP-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project involves an end-to-end statistical analysis and regression modeling of the **California Housing Dataset** (20,640 districts). The goal was not just to predict prices, but to understand the **non-linear economic drivers** of the housing market by moving beyond the limitations of Ordinary Least Squares (OLS) regression.

The analysis benchmarks multiple algorithms, optimizes the Bias-Variance trade-off using **GridSearchCV**, and employs **SHAP (SHapley Additive exPlanations)** to interpret the "Black Box" decisions of the final XGBoost model.

## 📊 Key Features
* **Advanced Model Selection:** Progressed from Baseline Linear Models (Ridge/Lasso) to Tree-based Ensembles (Random Forest) and Gradient Boosting (XGBoost).
* **Statistical Inference:** Used **SHAP values** to identify non-linear threshold effects in Median Income and Location that traditional coefficients missed.
* **Model Diagnostics:** Conducted residual analysis to detect **heteroscedasticity** and identify data-censoring artifacts (capped values) in the dataset.
* **Hyperparameter Optimization:** Utilized GridSearchCV to tune learning rates, tree depth, and estimators for optimal performance.

## 🛠️ Technologies Used
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`
* **Concepts:** Regression Analysis, Gradient Boosting, Game Theory (SHAP), Cross-Validation.

## 📈 Model Performance
The project systematically compared models to minimize Mean Squared Error (MSE). The final **XGBoost Regressor** significantly outperformed the baseline.

| Model | R² Score | Key Observation |
| :--- | :--- | :--- |
| **Linear Regression** | 0.60 | Failed to capture non-linear geographic patterns. |
| **Decision Tree** | 0.64 | High variance (overfitting). |
| **Random Forest** | 0.81 | Reduced variance through bagging. |
| **XGBoost (Optimized)**| **0.86** | **Best Fit.** Reduced error by **42%** vs Baseline. |

## 🧠 Explainability (SHAP Analysis)
To ensure the model is reliable and not just "fitting noise," I performed a forensic audit using SHAP:
1.  **Global Importance:** Identified **Location (Latitude/Longitude)** and **Median Income** as the primary drivers of price.
2.  **Non-Linearity:** Revealed that housing prices do not increase linearly with income; there is an exponential spike after a specific income threshold.
3.  **Geographic Interaction:** The model learned complex spatial clusters (e.g., coastal vs. inland) without explicit feature engineering.

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/debda45/California-Housing-Price-Prediction.git](https://github.com/debda45/California-Housing-Price-Prediction.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook Boston_Project_Housing_Analysis.ipynb
    ```

## 📜 Dataset
**California Housing Dataset** (Derived from the 1990 U.S. Census).
* **Rows:** 20,640
* **Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.
* **Target:** Median House Value.

---
*Created by Debdeep Das | MSc Statistics, IIT Kanpur*
