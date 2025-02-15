import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    try:
        df_path = input('Enter the CSV data path:\n ')
        return pd.read_csv(df_path)
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def clean_data(data, drop_columns=None):
    try:
        # drop useless columns
        if drop_columns:
            data = data.drop(columns=[col for col in drop_columns if col in data.columns]).copy()

        # drop constant columns
        constant_columns = [col for col in data.columns if data[col].nunique() == 1]
        data.drop(constant_columns, axis=1, inplace=True)

        # Standardize column names
        data.columns = data.columns.str.lower().str.strip().str.replace(" ", "_")

        # Ensure categorical data is consistent
        for col in data.select_dtypes(include=["object", "category"]).columns:
            data[col] = data[col].str.strip().str.lower()

        # Handling Missing Values
        mean_imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        mode_imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        for i in data.columns:
            if data[i].isna().any():
                if pd.api.types.is_numeric_dtype(data[i]):
                    data[i] = mean_imputer.fit_transform(data[i].values.reshape(-1, 1)).ravel()
                else:
                    data[i] = mode_imputer.fit_transform(data[i].values.reshape(-1, 1)).ravel()

        # Handling Outliers
        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

        return data

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return data


df = load_data()
df = clean_data(df, drop_columns=["id", "host_id", "listing_url", "scrape_id", "last_scraped", "name", "host_name",
                                       "neighbourhood_group", "license"])
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

df.head()
df.info()

# Basic statistics
print("Summary Statistics:")
print(df.describe())

# Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution of Airbnb Listings")
plt.xlabel("Price ($)")
plt.ylabel("Frequency")
plt.show()

# Most popular neighborhoods by listing count
top_neighbourhoods = df["neighbourhood"].value_counts().nlargest(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_neighbourhoods.index, y=top_neighbourhoods.values)
plt.xticks(rotation=45)
plt.title("Top 10 Neighbourhoods with Most Listings")
plt.ylabel("Number of Listings")
plt.show()

# Availability trends
plt.figure(figsize=(8, 5))
sns.histplot(df['availability_365'], bins=50, kde=True)
plt.title("Availability of Listings Over a Year")
plt.xlabel("Days Available")
plt.ylabel("Number of Listings")
plt.show()

# Room type distribution
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="room_type", palette="coolwarm", order=df["room_type"].value_counts().index)
plt.title("Room Type Distribution")
plt.show()

# using median because data is likely still right-skewed even after removing outliers .
median_prices = df.groupby('neighbourhood')['price'].median().sort_values(ascending=False)
plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x='neighbourhood', y='price', order=median_prices.index)
plt.xticks(rotation=90)
plt.title("Price Variation Across Neighborhoods")
plt.show()

# Price vs Room type
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="room_type", y="price")
plt.title("Price Comparison by Room Type")
plt.show()

# Price vs Availability
sns.scatterplot(data=df, x='price', y='availability_365')
plt.title('Price vs Availability')
plt.xlabel('Price ($)')
plt.ylabel('Availability (Days)')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.show()

#           INSIGHTS
print("\nKey Insights(For Albany, New York, United States):")
print(
    "1. Most listings are clustered at lower price points, with a long tail extending to the right due to a few exceptionally high-priced listings.")
print("2. The most popular location that attract the most bookings is the \"Sixth Ward\".")
print(
    "3. Availability varies widely, with some listings offering year-round access while others are only available for a short period.")
print("4. Entire homes have the highest booking rate among all room types.")
print(
    "5. Some neighborhoods have higher median prices, and the Fifteenth Ward stands out with the highest median price.")
print("6. Entire homes/apt are priced higher than shared/private rooms.")
print("7. There is no specific pattern between price and availability.")
print(
    "8. The weak negative relationship between price and availability indicates that pricing has little impact on how often listings are booked.")

#           BUSINESS RECOMMENDATIONS
print("\nBased on our analysis, we suggest:")
print("1. Hosts in high-demand neighborhoods should optimize pricing strategies to maximize revenue.")
print("2. Neighborhoods with high availability & high price are best for investment.")
print(
    "3. Prioritize entire homes: \n most travelers prefer entire homes over shared or private rooms.Hosts should focus on listing entire apartments for higher demand.")
print("4. Adjust pricing based on seasonal availability trends to increase occupancy rates.")
print("Offer discounts for longer stays to increase occupancy, especially for private and shared rooms.")
