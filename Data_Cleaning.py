import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# ---------->  Load dataset
df = pd.read_csv("Dataset.csv")

# ---------->  Convert month_year to datetime format
df['month_year'] = pd.to_datetime(df['month_year'])

# ---------->  Remove missing values
df.dropna(inplace=True)

# ---------->  Remove product_id (Not needed for model training)
df.drop(columns=['product_id'], inplace=True)

# ---------->  Encode categorical variables
df = pd.get_dummies(df, columns=['product_category_name'], drop_first=True)

# ---------->  Remove duplicate rows
df.drop_duplicates(inplace=True)

# ---------->  Function to remove outliers using IQR
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# ---------->  Apply outlier removal to selected numeric columns
num_cols = ['total_price', 'unit_price', 'freight_price', 'volume']
for col in num_cols:
    df = remove_outliers_iqr(df, col)  # ---------->  Remove outliers

# ---------->  Check if outliers were removed
outliers_remaining = {}
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_remaining[col] = len(outliers)

# ---------->  Print outlier removal results
for col, count in outliers_remaining.items():
    if count == 0:
        print(f"All outliers removed from {col}")
    else:
        print(f"{count} outliers still remaining in {col}")

# # ---------->  Apply MinMaxScaler (Normalization)
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
# ---------->Correlation Analysis (Rechecking Correlation), Compute correlation matrix
corr_matrix = df.corr()

# ----------> Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Cleaned Dataset")
plt.show()
# ---------->  Save the final cleaned dataset
df.to_csv("Cleaned_Dataset.csv", index=False)

print("Data cleaning complete. Final cleaned dataset saved as 'Cleaned_Dataset.csv'")
