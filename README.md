Solar-Power-Prediction-Using-Regression

## Overview
This project was developed during my **Data Science internship** and focuses on predicting **solar power generation** using regression-based machine learning models.  
Accurate prediction of solar power output is essential for **energy planning, grid management, and improving renewable energy efficiency**.

The project follows a complete **end-to-end machine learning pipeline**, from data preprocessing and EDA to model building, comparison, and evaluation.

---

## Problem Statement
Solar power generation depends heavily on environmental and weather-related factors such as temperature, solar radiation, and other atmospheric conditions.  
The objective of this project is to **build a machine learning model that can accurately predict solar power generation** based on historical and environmental data.

---

## Dataset
- Type: Real-world solar energy dataset  
- Features include:
  - Weather and environmental variables
  - Time-based and operational parameters
- Target Variable:
  - Solar power generation (continuous numerical value)

---

## Approach
1. **Data Cleaning & Preprocessing**
   - Handling missing values
   - Feature selection and scaling
2. **Exploratory Data Analysis (EDA)**
   - Understanding relationships between environmental factors and power output
   - Visualization of trends and correlations
3. **Feature Engineering**
   - Preparing meaningful input features for modeling
4. **Model Training**
   - Training multiple regression models
5. **Model Evaluation & Selection**
   - Comparing models using standard evaluation metrics

---

## Model Comparison & Selection
Multiple regression models were trained and evaluated to identify the most effective approach:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  

Each model was evaluated using:
- **R² Score**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**

Among all tested models, **Random Forest Regressor achieved the best performance**, demonstrating higher accuracy and better generalization on unseen data.  
This is because Random Forest effectively captures **non-linear relationships** between environmental variables and solar power generation.

---

## Results & Insights
- Random Forest Regressor provided the most accurate predictions
- Key environmental factors influencing solar power generation were identified
- The model demonstrates how machine learning can support **renewable energy forecasting and decision-making**

---

## Tools & Technologies
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

