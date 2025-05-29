# 🏠 Housing Price Prediction Using Linear Regression

This project uses a **Linear Regression** model to predict housing prices based on various features in a dataset. The model is built using Python's popular machine learning and data visualization libraries.

## 📊 Dataset

- **Filename**: `Housing.csv`
- **Target variable**: `price`
- **Features**: Includes variables such as `area`, `bedrooms`, `bathrooms`, and other encoded categorical variables.

## 🔍 Project Overview

The main steps in this project include:

1. Loading and preprocessing the dataset
2. Encoding categorical variables using one-hot encoding
3. Splitting the data into training and testing sets
4. Training a linear regression model
5. Evaluating model performance using:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score
6. Visualizing:
   - Feature importance based on regression coefficients
   - A simple regression plot for `area` vs `price`

## 📦 Requirements

- Python 3.7+
- Libraries:
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

### Install Dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
