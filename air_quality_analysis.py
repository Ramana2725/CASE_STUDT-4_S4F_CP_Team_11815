
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load CSV data
df = pd.read_csv('air_quality_traffic_weather.csv', parse_dates=['date'])

# Quick look
print(df.head())
print(df.info())

# Set datetime index
df.set_index('date', inplace=True)

# Handle missing values
df = df.fillna(method='ffill')

# Optional: Resample data by day
df_daily = df.resample('D').mean()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_daily.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Air Quality Trends Over Time
df_daily[['PM2.5', 'PM10', 'NO2', 'CO']].plot(figsize=(12, 6))
plt.title('Air Quality Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Concentration')
plt.show()

# Predict PM2.5 using traffic and weather data
features = ['traffic_volume', 'temperature', 'humidity', 'wind_speed']
target = 'PM2.5'

X = df_daily[features]
y = df_daily[target]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')

# Feature Importance
coefficients = pd.Series(model.coef_, index=features)
coefficients.plot(kind='barh')
plt.title('Feature Importance for Predicting PM2.5')
plt.show()
