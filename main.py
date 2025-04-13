import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ----------> Load the Dataset, Cleaned dataset file path
df = pd.read_csv("Cleaned_Dataset.csv") 

# ----------> Display first few rows
# print(df.head())


# ----------> Feature Engineering Convert 'month_year' Column to datetime format
df['month_year'] = pd.to_datetime(df['month_year'])

# ----------> Now Extract new date-related features
df['year'] = df['month_year'].dt.year
df['month'] = df['month_year'].dt.month
df['day'] = df['month_year'].dt.day

df.drop(columns=['month_year'], inplace=True)



# ----------> Create 'time_index' using 'year' and 'month'
df['time_index'] = pd.to_datetime(df[['year', 'month']].assign(day=1))  # Set day=1

# ----------> Verify the new column
# print(df[['year', 'month', 'time_index']].head())



# ----------> Calculate competitor price difference
df['comp_price_diff'] = df['total_price'] - df['comp_1']

# ----------> Rolling mean for price trends (last 3 months)
df['rolling_avg_price'] = df['total_price'].rolling(window=3, min_periods=1).mean()

# ----------> Display modified dataset
# print(df.head())


df['quarter'] = df['time_index'].dt.quarter
df['day_of_week'] = df['time_index'].dt.dayofweek




# ----------> Create a new 'time_index' column using 'year' and 'month'
df['time_index'] = pd.to_datetime(df[['year', 'month']].assign(day=1))  # Set day=1 for consistency

#  Drop 'year' and 'month' if they are no longer needed
df.drop(columns=['year', 'month'], inplace=True)


# ----------> Handle Missing Values, Fill missing numerical values with median
df.fillna(df.median(), inplace=True)

# ----------> Fill missing categorical values with mode
df.fillna(df.mode().iloc[0], inplace=True)

# ----------> Verify missing values again
# print(df.isnull().sum())


# ----------> Display updated dataset
# print(df.head())
# ----------> Train-Test Split, To Define features (X) and target variable (y)
Y = df['total_price']  # Target variable

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


# ----------> Feature Scaling, Select features to scale
Columns = ['qty', 'unit_price', 'product_score', 'customers', 'comp_1', 'lag_price', 'comp_price_diff', 'rolling_avg_price']


# ----------> Apply Standard Scaling
scaler = StandardScaler()
X[Columns] = scaler.fit_transform(X[Columns])
# scaler = MinMaxScaler()
# X[Columns] = scaler.fit_transform(X[Columns])


# Save the scaler for later use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ----------> Display dataset after scaling
# print(X)

# ----------> Split into 80% training and 20% testing data from Scaled Variables
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



# # ----------> Print dataset shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



#----------> Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ----------> rain XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# ----------> Predict using Random Forest
y_pred_rf = rf_model.predict(X_test)

# ----------> predict using  XGBoost Regressor
y_pred_xgb = xgb_model.predict(X_test)

# ----------> Define function to calculate performance metrics
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n {model_name} Performance:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R² Score:", r2_score(y_test, y_pred))

# ----------> Evaluate both models
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")


# ----------> Define function to calculate performance metrics
def Performance_matrics(y_test, y_pred, model_name):
    r2 = r2_score(y_test, y_pred)  # ---------->R² score
    accuracy = r2 * 100  # ----------> Convert R² to percentage
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R² Score": r2,
        "Accuracy (%)": round(accuracy, 2)  # ----------> Convert to percentage (e.g., 90.5%)
    }

# ----------> add the performance matrixs data og both models into list 
results = []
results.append(Performance_matrics(y_test, y_pred_rf, "Random Forest"))
results.append(Performance_matrics(y_test, y_pred_xgb, "XGBoost"))

# ----------> Convert to DataFrame for better visualization , Display results in dataframe format
results_df = pd.DataFrame(results)
print("\n Model Performance Comparison:")
print(results_df)

# ----------> Plot Accuracy Comparison as Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy (%)"], color=['blue', 'green', 'red'])
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)  # Ensure the y-axis is in percentage range
plt.grid(axis="y", linestyle="--")
plt.show()




# ----------> Visualizing Price Trends Over Time, Group by 'time_index' to plot trends over time
df.groupby('time_index')['total_price'].mean().plot(kind='line', figsize=(10,5), marker='o', title="Average Price Over Time")

plt.xlabel("Time (Year-Month)")
plt.ylabel("Average Total Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



# ----------> Competitor Price Impact on Sales
sns.scatterplot(x=df["comp_1"], y=df["total_price"])
plt.title("Competitor 1 Price vs Total Price")
plt.xlabel("Competitor 1 Price")
plt.ylabel("Total Price")
plt.show()


#  ----------> Save Trained Model
with open("xgb_model.pkl", "wb") as model_file:
    pickle.dump(xgb_model, model_file)
    print("Model saved as xgb_model.pkl")



# ================================================================================================================
# ----------> Dynamic pricing adjustment by competiter, demand trends, 

# ----------> Create a new DataFrame with predictions
Dynamic_data = X_test.copy()
# print(Dynamic_data)
y_pred = xgb_model.predict(Dynamic_data)

# ----------> Convert y_pred to a DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=["Predicted_Price"])

# print(y_pred_df)
scaler = pickle.load(open("scaler.pkl", "rb"))
# ----------> Reshape and inverse transform the predictions
Dynamic_data_original = scaler.inverse_transform(Dynamic_data)

# ----------> Convert the transformed data back to a DataFrame
Dynamic_data_original = pd.DataFrame(Dynamic_data_original, columns=Dynamic_data.columns)


Dynamic_data = pd.concat([Dynamic_data_original, y_pred_df], axis=1)

# print(Dynamic_data)

# ----------> Apply Dynamic Pricing Adjustments
Dynamic_data['Final_Price'] = Dynamic_data['Predicted_Price']

# ----------> Reduce price **only if competitor is significantly cheaper**
Dynamic_data.loc[Dynamic_data['comp_1'] < Dynamic_data['Predicted_Price'], 'Final_Price'] *= 0.98  # Reduce by 2%

# ----------> Increase price **if competitor is expensive**
Dynamic_data.loc[Dynamic_data['comp_1'] > Dynamic_data['Predicted_Price'], 'Final_Price'] *= 1.15  # Increase by 15%

# ----------> Ensure Prices Do Not Drop Below a Reasonable Level
Dynamic_data['Final_Price'] = Dynamic_data['Final_Price'].clip(lower=Dynamic_data['comp_1'] * 0.90)  # Minimum price is 90% of competitor's price
Dynamic_data['Final_Price'] = Dynamic_data['Final_Price'].clip(lower=0.50)  # Minimum absolute price threshold

# ----------> Ensure ALL products get at least a 2% price increase if revenue stays the same
Dynamic_data.loc[Dynamic_data['Final_Price'] == Dynamic_data['Predicted_Price'], 'Final_Price'] *= 1.02  # Increase by 2%

# ----------> Display updated prices
print(Dynamic_data[['Predicted_Price', 'Final_Price', 'comp_1']].head())

# ----------> Ensure Old_Price is reasonable (at least 0.50)
Dynamic_data['Old_Price'] = y_test.clip(lower=0.50)

# ----------> Calculate Price Change
Dynamic_data['Price_Change'] = (Dynamic_data['Final_Price'] - Dynamic_data['Old_Price']) / Dynamic_data['Old_Price']

# ----------> Prevent Extreme Price Reductions (-30% Max Drop)
Dynamic_data['Price_Change'] = Dynamic_data['Price_Change'].clip(lower=-0.30, upper=2.00)  # Max -30% drop

# ----------> Simulate Demand Change
Dynamic_data['Demand_Change'] = np.random.uniform(-0.2, 0.2, size=len(Dynamic_data))  

# ----------> Calculate Price Elasticity
Dynamic_data['Price_Elasticity'] = np.abs(Dynamic_data['Demand_Change'] / Dynamic_data['Price_Change'])  
Dynamic_data.replace([np.inf, -np.inf], 0, inplace=True)  # Replace infinite values
Dynamic_data['Price_Elasticity'] = Dynamic_data['Price_Elasticity'].clip(-5, 5)  # Limit range

# # ----------> Display results
print(Dynamic_data[['Old_Price', 'Final_Price', 'Price_Change', 'Demand_Change', 'Price_Elasticity']].head())

# ----------> Simulate Sales Volume
Dynamic_data['Estimated_Sales'] = np.random.randint(50, 500, size=len(Dynamic_data))  

# ----------> Fix Revenue Calculation (Ensure No Negative Values)
Dynamic_data['Old_Revenue'] = (Dynamic_data['Old_Price'] * Dynamic_data['Estimated_Sales']).clip(lower=0)
Dynamic_data['New_Revenue'] = (Dynamic_data['Final_Price'] * Dynamic_data['Estimated_Sales']).clip(lower=0)

# ----------> Ensure Revenue Increase
Dynamic_data.loc[Dynamic_data['New_Revenue'] <= Dynamic_data['Old_Revenue'], 'Final_Price'] *= 1.02  
Dynamic_data['New_Revenue'] = (Dynamic_data['Final_Price'] * Dynamic_data['Estimated_Sales']).clip(lower=0)

# # ----------> Display Revenue Impact
print("\n Revenue Impact Before and After Dynamic Pricing:")
print(Dynamic_data[['Old_Revenue', 'New_Revenue']].head())

# ----------> Plot Revenue Impact
plt.figure(figsize=(8, 5))
plt.bar(["Old Revenue", "New Revenue"], [Dynamic_data['Old_Revenue'].sum(), Dynamic_data['New_Revenue'].sum()], color=['red', 'green'])
plt.title("Revenue Comparison: Before vs. After Dynamic Pricing")
plt.ylabel("Total Revenue")
plt.show()


# ----------> Save final results to CSV for reporting
Dynamic_data.to_csv("Final_Dynamic_Pricing_Results.csv", index=False)


# ===========================================================================================================


# ----------> Generate Key Summary Statistics
# summry_report = {
#     "Average Old Price": Dynamic_data['Old_Price'].mean(),
#     "Average Final Price": Dynamic_data['Final_Price'].mean(),
#     "Total Old Revenue": Dynamic_data['Old_Revenue'].sum(),
#     "Total New Revenue": Dynamic_data['New_Revenue'].sum(),
#     "Revenue Increase (%)": ((Dynamic_data['New_Revenue'].sum() - Dynamic_data['Old_Revenue'].sum()) / Dynamic_data['Old_Revenue'].sum()) * 100
# }

# ----------> Convert the summry into data frame and then save the data 
# summary_df = pd.DataFrame([summry_report])
# summary_df.to_csv("Project_Summary_Report.csv", index=False)

# ----------> Display Summary
# print("\n Project Summary Report:")
# print(summary_df)








# **********************************************************************************************************************************
# **********************************************************************************************************************************
# **********************************************************************************************************************************
# **********************************************************************************************************************************
# **********************************************************************************************************************************

"""                         Informtion
input feature for the model is: [['qty', 'unit_price', 'product_score', 'customers', 'comp_1',
       'lag_price', 'comp_price_diff', 'rolling_avg_price']]

       
       use the scaler.pkl sclaer for input feature

       the use the xgbost_model.pkl for the prediction

       after that the Scaler.inverse_transform(input_data) function for original values

"""

# **********************************************************************************************************************************
# **********************************************************************************************************************************
# **********************************************************************************************************************************
# **********************************************************************************************************************************
# **********************************************************************************************************************************
