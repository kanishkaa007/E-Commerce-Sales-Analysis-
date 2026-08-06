import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")

# Load Dataset
print("E-Commerce CSV Data")
df = pd.read_csv('C:/Users/narwa/Downloads/cleaned_ecommerce.csv', low_memory=False)

# create features
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Year'] = df['InvoiceDate'].dt.year
df['Quarter'] = df['InvoiceDate'].dt.to_period('Q').astype(str)

if 'LineTotal' in df.columns:
    df['Sales'] = df['LineTotal']

print(f"Dataset Loaded Successfully: {df.shape[0]:,} records found.\n")
print(df.head(), "\n")

# (EDA)
print(" Exploratory Data Analysis")
# Monthly Sales Trend
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o', color='b', linewidth=2.5)
plt.title('Monthly Sales Trend')
plt.xticks(range(1, 13))
plt.ylabel('Total Sales ($)')
plt.show()
# Top 10 Selling Products
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=top_products, x='Quantity', y='Description', palette='viridis')
plt.title('Top 10 Selling Products by Quantity')
plt.xlabel('Units Sold')
plt.show()

#Sales Distribution by Top Countries
top_countries = df.groupby('Country')['Sales'].sum().sort_values(ascending=False).head(5)
other_sales = df['Sales'].sum() - top_countries.sum()
country_sales = pd.concat([top_countries, pd.Series({'Other': other_sales})])

plt.figure(figsize=(7, 7))
plt.pie(country_sales.values, labels=country_sales.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Sales Distribution by Country (Top 5 vs. Other)')
plt.show()

#  Seasonal Product Demand for Top 5 Items
top_5_items = top_products['Description'].head(5).tolist()
seasonal_df = df[df['Description'].isin(top_5_items)]
seasonal_demand = seasonal_df.groupby(['Quarter', 'Description'])['Quantity'].sum().reset_index()

plt.figure(figsize=(11, 5))
sns.barplot(data=seasonal_demand, x='Quarter', y='Quantity', hue='Description')
plt.title('Seasonal Product Demand (Quarterly for Top 5 Items)')
plt.ylabel('Units Sold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# MLP
print("\nPreparing Data for Machine Learning...")

# Sampling 50,000 rows for efficient model training
sample_df = df.sample(n=min(50000, len(df)), random_state=42).copy()

# Group rare countries to keep model lightweight
top_5_country_list = top_countries.index.tolist()
sample_df['Country_Grouped'] = sample_df['Country'].apply(lambda x: x if x in top_5_country_list else 'Other')

# One-Hot Encoding categorical features
ml_data = pd.get_dummies(sample_df, columns=['Country_Grouped'], drop_first=True)

# Select Features and Target Variable
features = ['Month', 'DayOfWeek', 'UnitPrice'] + [col for col in ml_data.columns if 'Country_Grouped_' in col]
X = ml_data[features]
y = ml_data['Quantity']

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Model 2: Random Forest Regressor
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 4. Model Evaluation
print("\nModel Evaluation Results:")

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] RMSE: {rmse:.4f} | R² Score: {r2:.4f}")

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)
