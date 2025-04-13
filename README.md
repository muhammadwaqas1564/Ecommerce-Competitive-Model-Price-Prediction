# Dynamic Pricing Model for E-commerce

This project leverages machine learning to build an intelligent dynamic pricing model for e-commerce platforms. The goal is to optimize product pricing based on historical sales data, competitor prices, product demand, and other market variables.

## Dataset

The dataset used is the **Retail Price Optimization** dataset from Kaggle. It includes key features such as:

- `qty`: Quantity sold
- `unit_price`: Current price of the product
- `product_score`: Popularity rating of the product
- `customers`: Unique customers
- `comp_1`: Competitor price
- `lag_price`: Historical price of the product
- `comp_price_diff`: Price difference with competitors
- `rolling_avg_price`: Average price over a time window

## Model

We use **XGBoost Regressor** for predicting the optimal price based on the given features. The model is trained on standardized data using `StandardScaler`.


These results indicate high prediction accuracy and generalization of the model.

## Features

- Real-time price prediction
- Competitor price monitoring
- Demand-driven dynamic pricing
- Revenue growth strategy

## Requirements

- Python 3.10+
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib (for visualization)
- pickle (for model and scaler saving)
