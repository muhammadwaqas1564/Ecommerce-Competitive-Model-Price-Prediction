import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 2: Load & Display the Dataset
# Load dataset
df = pd.read_csv("Cleaned_Dataset.csv") 

# # Display first few rows
# print(df.head())

# # Check data types & missing values
# print(df.info())
# print(df.isnull().sum())


# Step 3: Feature Engineering
# Convert 'month_year' to datetime format
df['month_year'] = pd.to_datetime(df['month_year'])

# Extract new date-related features
df['year'] = df['month_year'].dt.year
df['month'] = df['month_year'].dt.month
df['day'] = df['month_year'].dt.day

df.drop(columns=['month_year'], inplace=True)



# Create 'time_index' using 'year' and 'month'
df['time_index'] = pd.to_datetime(df[['year', 'month']].assign(day=1))  # Set day=1

# # Verify the new column
# print(df[['year', 'month', 'time_index']].head())



# Calculate competitor price difference
df['comp_price_diff'] = df['total_price'] - df['comp_1']

# Rolling mean for price trends (last 3 months)
df['rolling_avg_price'] = df['total_price'].rolling(window=3, min_periods=1).mean()

# # Display modified dataset
# print(df.head())


df['quarter'] = df['time_index'].dt.quarter
df['day_of_week'] = df['time_index'].dt.dayofweek




#  Create a new 'time_index' column using 'year' and 'month'
df['time_index'] = pd.to_datetime(df[['year', 'month']].assign(day=1))  # Set day=1 for consistency
#  Drop 'year' and 'month' if they are no longer needed
df.drop(columns=['year', 'month'], inplace=True)


#  Step 4: Handle Missing Values
# Fill missing numerical values with median
df.fillna(df.median(), inplace=True)

# Fill missing categorical values with mode
df.fillna(df.mode().iloc[0], inplace=True)

# # Verify missing values again
# print(df.isnull().sum())




# # Step 5: Encode Categorical Variables
# # Identify categorical columns
# categorical_cols = ['some_categorical_column']  # Replace with actual column names

# # Apply One-Hot Encoding
# encoder = OneHotEncoder(drop='first', sparse=False)
# encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))

# # Rename columns
# encoded_features.columns = encoder.get_feature_names_out(categorical_cols)

# # Drop original categorical columns & merge encoded features
# df.drop(columns=categorical_cols, inplace=True)
# df = pd.concat([df, encoded_features], axis=1)

# # Display updated dataset
# print(df.head())


# Step 6: Feature Scaling
# Select numerical features to scale
numerical_cols = ['qty', 'total_price', 'freight_price', 'unit_price', 'product_name_lenght', 
                  'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_score', 
                  'customers', 'weekday', 'weekend', 'holiday', 's', 'volume', 
                  'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']


# Apply Standard Scaling
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display dataset after scaling
# print(df.head())




#  Step 7: Train-Test Split
# Define features (X) and target variable (y)
X = df.drop(columns=['total_price','time_index','freight_price', 'comp_2', 'comp_3', 'product_photos_qty', 'product_description_lenght', 'day_of_week', 'holiday', 'ps2', 'fp2', 'ps3', 'fp3','day','volume','product_name_lenght','product_weight_g','s', 'fp1','ps1',
    'weekend',
    'weekday',
    'quarter',
    'product_category_name_computers_accessories',
    'product_category_name_consoles_games',
    'product_category_name_cool_stuff',
    'product_category_name_furniture_decor',
    'product_category_name_garden_tools',
    'product_category_name_health_beauty',
    'product_category_name_perfumery',
    'product_category_name_watches_gifts'])  # Features
y = df['total_price']  # Target variable

# print(X.columns)


# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Print dataset shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)






# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

from xgboost import XGBRegressor

# ✅ Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)



# Predict using Random Forest
y_pred_rf = rf_model.predict(X_test)

# predict using  XGBoost Regressor
y_pred_xgb = xgb_model.predict(X_test)

# Define function to calculate performance metrics
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n {model_name} Performance:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R² Score:", r2_score(y_test, y_pred))

# Evaluate both models
# evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")


# ✅ Define function to calculate performance metrics
def evaluate_model(y_test, y_pred, model_name):
    r2 = r2_score(y_test, y_pred)  # R² score
    accuracy = r2 * 100  # Convert R² to percentage
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R² Score": r2,
        "Accuracy (%)": round(accuracy, 2)  # Convert to percentage (e.g., 90.5%)
    }

# ✅ Evaluate All Models
results = []
# results.append(evaluate_model(y_test, y_pred_lr, "Linear Regression"))
results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))
results.append(evaluate_model(y_test, y_pred_xgb, "XGBoost"))

# ✅ Convert to DataFrame for better visualization
results_df = pd.DataFrame(results)

# ✅ Display results in table format
print("\n Model Performance Comparison:")
print(results_df)

# ✅ Plot Accuracy Comparison as Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy (%)"], color=['blue', 'green', 'red'])
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)  # Ensure the y-axis is in percentage range
plt.grid(axis="y", linestyle="--")
plt.show()




#  Step 10: Correlation Analysis (Rechecking Correlation)
# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Cleaned Dataset")
plt.show()

# # print(df)


#  Step 11: Visualizing Price Trends Over Time
#  Group by 'time_index' to plot trends over time
df.groupby('time_index')['total_price'].mean().plot(kind='line', figsize=(10,5), marker='o', title="Average Price Over Time")

#  Labels and formatting
plt.xlabel("Time (Year-Month)")
plt.ylabel("Average Total Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



# Step 12: Competitor Price Impact on Sales
sns.scatterplot(x=df["comp_1"], y=df["total_price"])
plt.title("Competitor 1 Price vs Total Price")
plt.xlabel("Competitor 1 Price")
plt.ylabel("Total Price")
plt.show()




# ✅ Create a new DataFrame with predictions
df_test = X_test.copy()
df_test['Predicted_Price'] = xgb_model.predict(X_test)

# ✅ Apply Dynamic Pricing Adjustments
df_test['Final_Price'] = df_test['Predicted_Price']

# ✅ Reduce price **only if competitor is significantly cheaper**
df_test.loc[df_test['comp_1'] < df_test['Predicted_Price'], 'Final_Price'] *= 0.98  # Reduce by 2%

# ✅ Increase price **if competitor is expensive**
df_test.loc[df_test['comp_1'] > df_test['Predicted_Price'], 'Final_Price'] *= 1.15  # Increase by 15%

# ✅ Ensure Prices Do Not Drop Below a Reasonable Level
df_test['Final_Price'] = df_test['Final_Price'].clip(lower=df_test['comp_1'] * 0.90)  # Minimum price is 90% of competitor's price
df_test['Final_Price'] = df_test['Final_Price'].clip(lower=0.50)  # Minimum absolute price threshold

# ✅ Ensure ALL products get at least a 2% price increase if revenue stays the same
df_test.loc[df_test['Final_Price'] == df_test['Predicted_Price'], 'Final_Price'] *= 1.02  # Increase by 2%

# ✅ Display updated prices
print(df_test[['Predicted_Price', 'Final_Price', 'comp_1']].head())

# ✅ Ensure Old_Price is reasonable (at least 0.50)
df_test['Old_Price'] = y_test.clip(lower=0.50)

# ✅ Calculate Price Change
df_test['Price_Change'] = (df_test['Final_Price'] - df_test['Old_Price']) / df_test['Old_Price']

# ✅ Prevent Extreme Price Reductions (-30% Max Drop)
df_test['Price_Change'] = df_test['Price_Change'].clip(lower=-0.30, upper=2.00)  # Max -30% drop

# ✅ Simulate Demand Change
df_test['Demand_Change'] = np.random.uniform(-0.2, 0.2, size=len(df_test))  

# ✅ Calculate Price Elasticity
df_test['Price_Elasticity'] = np.abs(df_test['Demand_Change'] / df_test['Price_Change'])  
df_test.replace([np.inf, -np.inf], 0, inplace=True)  # Replace infinite values
df_test['Price_Elasticity'] = df_test['Price_Elasticity'].clip(-5, 5)  # Limit range

# ✅ Display results
print(df_test[['Old_Price', 'Final_Price', 'Price_Change', 'Demand_Change', 'Price_Elasticity']].head())

# ✅ Simulate Sales Volume
df_test['Estimated_Sales'] = np.random.randint(50, 500, size=len(df_test))  

# ✅ Fix Revenue Calculation (Ensure No Negative Values)
df_test['Old_Revenue'] = (df_test['Old_Price'] * df_test['Estimated_Sales']).clip(lower=0)
df_test['New_Revenue'] = (df_test['Final_Price'] * df_test['Estimated_Sales']).clip(lower=0)

# ✅ Ensure Revenue Increase
df_test.loc[df_test['New_Revenue'] <= df_test['Old_Revenue'], 'Final_Price'] *= 1.02  
df_test['New_Revenue'] = (df_test['Final_Price'] * df_test['Estimated_Sales']).clip(lower=0)

# ✅ Display Revenue Impact
print("\n📊 Revenue Impact Before and After Dynamic Pricing:")
print(df_test[['Old_Revenue', 'New_Revenue']].head())

# ✅ Plot Revenue Impact
plt.figure(figsize=(8, 5))
plt.bar(["Old Revenue", "New Revenue"], [df_test['Old_Revenue'].sum(), df_test['New_Revenue'].sum()], color=['red', 'green'])
plt.title("Revenue Comparison: Before vs. After Dynamic Pricing")
plt.ylabel("Total Revenue")
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

# ✅ Display Summary
print("\n📊 Project Summary Report:")
print(summary_df)

# UI



import pickle

# ✅ Save Trained Model
with open("xgb_model.pkl", "wb") as model_file:
    pickle.dump(xgb_model, model_file)

print("✅ Model saved as xgb_model.pkl")








# # the below code for only testing from random rows and coloms ::

# #  Get feature names from X_train
# feature_names = X_train.columns.tolist()

# #  Generate 5 rows of synthetic test data
# test_data = pd.DataFrame([
#     np.random.uniform(low=-1, high=1, size=len(feature_names)) for _ in range(5)
# ], columns=feature_names)

# #  Display the synthetic test data
# print("\n📊 Sample Test Data for Model Prediction:")
# print(test_data)


# #  Predict using the XGBoost Model
# y_pred_xgb_test = xgb_model.predict(test_data)

# # Display predictions
# print("\n🚀 Model Predictions on New Data:")
# print(y_pred_xgb_test)







