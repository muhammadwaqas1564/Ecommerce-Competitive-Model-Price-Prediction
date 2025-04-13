import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------->   Load dataset
df = pd.read_csv("Cleaned_Dataset.csv")

# ---------->   Convert 'month_year' to datetime and extract features
df['month_year'] = pd.to_datetime(df['month_year'])
df['year'] = df['month_year'].dt.year
df['month'] = df['month_year'].dt.month
df['day'] = df['month_year'].dt.day
df.drop(columns=['month_year'], inplace=True)

# ---------->   Create 'time_index'
df['time_index'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df.drop(columns=['year', 'month'], inplace=True)

# ---------->   Feature Engineering
df['comp_price_diff'] = df['total_price'] - df['comp_1']
df['rolling_avg_price'] = df['total_price'].rolling(window=3, min_periods=1).mean()
df['quarter'] = df['time_index'].dt.quarter
df['day_of_week'] = df['time_index'].dt.dayofweek

# ---------->   Handling missing values
df.fillna(df.median(), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# ---------->   Feature Scaling
numerical_cols = ['qty', 'total_price', 'freight_price', 'unit_price', 'product_name_lenght',
                  'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_score',
                  'customers', 'weekday', 'weekend', 'holiday', 's', 'volume',
                  'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# ---------->   Train-Test Split
X = df.drop(columns=['total_price', 'time_index', 'freight_price', 'comp_2', 'comp_3', 'product_photos_qty', 
                     'product_description_lenght', 'day_of_week', 'holiday', 'ps2', 'fp2', 'ps3', 'fp3',
                     'day', 'volume', 'product_name_lenght', 'product_weight_g', 's', 'fp1', 'ps1',
                     'weekend', 'weekday', 'quarter'])

y = df['total_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------->   Train Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# ---------->   Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# ---------->   Model Evaluation
def evaluate_model(y_test, y_pred, model_name):
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R² Score": r2_score(y_test, y_pred),
        "Accuracy (%)": round(r2_score(y_test, y_pred) * 100, 2)
    }

results = []
results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))
results.append(evaluate_model(y_test, y_pred_xgb, "XGBoost"))

results_df = pd.DataFrame(results)


# ---------->  Distribution of Target Variable (Total Price)
plt.figure(figsize=(8,5))
sns.histplot(y, bins=50, kde=True, color="blue")
plt.title("Distribution of Total Price")
plt.xlabel("Total Price")
plt.ylabel("Frequency")
plt.grid()
plt.show()


# ---------->  Feature Importance (Combined for RF & XGBoost) To show key factors driving the model.
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Random Forest": rf_model.feature_importances_,
    "XGBoost": xgb_model.feature_importances_
}).set_index("Feature")

feature_importance_df.plot(kind="barh", figsize=(10,6), title="Feature Importance Comparison (RF vs XGBoost)")
plt.show()


# ----------> Predicted VS Acutal Price
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred_xgb, color="red", label="XGBoost Predictions")
sns.scatterplot(x=y_test, y=y_pred_rf, color="blue", label="Random Forest Predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', color="black")  # Ideal predictions line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.title("Actual vs Predicted Prices")
plt.grid()
plt.show()


# ----------> Residual Plot (Error Distribution) To check if the model errors are normally distributed.


plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_test - y_pred_xgb, color="green", label="XGBoost Residuals")
sns.scatterplot(x=y_test, y=y_test - y_pred_rf, color="blue", label="Random Forest Residuals")
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Actual Prices")
plt.ylabel("Residuals (Error)")
plt.legend()
plt.title("Residual Plot for Model Errors")
plt.grid()
plt.show()

# ---------->  Box Plot of Competitor Prices vs. Total Price  To compare how competitor prices influence the total price.
plt.figure(figsize=(8,5))
sns.boxplot(x="comp_1", y="total_price", data=df)
plt.title("Distribution of Total Price Across Competitor 1 Prices")
plt.xlabel("Competitor 1 Price")
plt.ylabel("Total Price")
plt.grid()
plt.show()

# ----------> Monthly Sales Trend Over Time To observe seasonality and trends in sales.

df.groupby(df['time_index'])['total_price'].sum().plot(kind="line", marker='o', figsize=(10,5), color="purple")
plt.title("Monthly Sales Trend Over Time")
plt.xlabel("Time (Year-Month)")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid()
plt.show()



# ---------->   Model Performance Comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy (%)"], color=['blue', 'green'])
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--")
plt.show()

# ---------->   Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Cleaned Dataset")
plt.show()

# ---------->   Price Trends Over Time
df.groupby('time_index')['total_price'].mean().plot(kind='line', figsize=(10,5), marker='o', title="Average Price Over Time")
plt.xlabel("Time (Year-Month)")
plt.ylabel("Average Total Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# ---------->   Competitor Price Impact on Sales
sns.scatterplot(x=df["comp_1"], y=df["total_price"])
plt.title("Competitor 1 Price vs Total Price")
plt.xlabel("Competitor 1 Price")
plt.ylabel("Total Price")
plt.show()

# ---------->   Feature Importance (Random Forest)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_importances.nlargest(10).plot(kind='barh', title="Top 10 Important Features (Random Forest)")
plt.show()

# ---------->   Feature Importance (XGBoost)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
xgb_importances.nlargest(10).plot(kind='barh', title="Top 10 Important Features (XGBoost)", color="red")
plt.show()

# ---------->   Residual Analysis
plt.figure(figsize=(10, 5))
sns.histplot(y_test - y_pred_rf, bins=30, kde=True, color='blue', label="Random Forest Residuals")
sns.histplot(y_test - y_pred_xgb, bins=30, kde=True, color='green', label="XGBoost Residuals")
plt.legend()
plt.title("Residual Distribution")
plt.show()

# ---------->   Revenue Impact Analysis
df_test = X_test.copy()
df_test['Predicted_Price'] = xgb_model.predict(X_test)
df_test['Final_Price'] = df_test['Predicted_Price']

df_test.loc[df_test['comp_1'] < df_test['Predicted_Price'], 'Final_Price'] *= 0.98
df_test.loc[df_test['comp_1'] > df_test['Predicted_Price'], 'Final_Price'] *= 1.15
df_test['Final_Price'] = df_test['Final_Price'].clip(lower=np.maximum(df_test['comp_1'] * 0.90, 0.50))
df_test.loc[df_test['Final_Price'] == df_test['Predicted_Price'], 'Final_Price'] *= 1.02

df_test['Old_Price'] = y_test.clip(lower=0.50)
df_test['Price_Change'] = (df_test['Final_Price'] - df_test['Old_Price']) / df_test['Old_Price']
df_test['Price_Change'] = df_test['Price_Change'].clip(lower=-0.30, upper=2.00)
df_test['Demand_Change'] = np.random.uniform(-0.2, 0.2, size=len(df_test))
df_test['Price_Elasticity'] = np.abs(df_test['Demand_Change'] / df_test['Price_Change'])
df_test.replace([np.inf, -np.inf], 0, inplace=True)
df_test['Price_Elasticity'] = df_test['Price_Elasticity'].clip(-5, 5)

df_test['Estimated_Sales'] = np.random.randint(50, 500, size=len(df_test))
df_test['Old_Revenue'] = df_test['Old_Price'] * df_test['Estimated_Sales']
df_test['New_Revenue'] = df_test['Final_Price'] * df_test['Estimated_Sales']

# ---------->   Create a summary DataFrame with just 2 rows
summary_df = pd.DataFrame({
    "Revenue Type": ["Old Revenue", "New Revenue"],
    "Total Revenue": [df_test['Old_Revenue'].sum(), df_test['New_Revenue'].sum()]
})

# ---------->   Plot using the summary DataFrame
sns.barplot(data=summary_df, x="Revenue Type", y="Total Revenue")

# ---------->   Show the plot
plt.show()



# ✅ Save final results to CSV for reporting
df_test.to_csv("Final_Dynamic_Pricing_Results.csv", index=False)

# ✅ Generate Key Summary Statistics
summary_report = {
    "Average Old Price": df_test['Old_Price'].mean(),
    "Average Final Price": df_test['Final_Price'].mean(),
    "Total Old Revenue": df_test['Old_Revenue'].sum(),
    "Total New Revenue": df_test['New_Revenue'].sum(),
    "Revenue Increase (%)": ((df_test['New_Revenue'].sum() - df_test['Old_Revenue'].sum()) / df_test['Old_Revenue'].sum()) * 100
}

# ✅ Convert to DataFrame and Save
summary_df = pd.DataFrame([summary_report])
summary_df.to_csv("Project_Summary_Report.csv", index=False)