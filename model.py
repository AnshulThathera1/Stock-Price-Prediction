import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('NFLX.csv')

# Data Cleaning
data = data.dropna()  # Drop rows with missing values

# Feature Engineering
features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
target = data['Close'].shift(-1)  # Predict the next day's closing price

# Remove the last row as it will have NaN target
features = features[:-1]
target = target[:-1]

# Data Normalization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression - MSE: {mse}, MAE: {mae}, R2: {r2}")

# Save the model
joblib.dump(lr_model, 'lr_model.pkl')

# Visualization Questions
# 1. Distribution of closing prices
plt.figure(figsize=(10, 6))
histplot = sns.histplot(data['Close'], kde=True, color='blue')
plt.title('Distribution of Closing Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')

# Add labels to each bar
for p in histplot.patches:
    height = p.get_height()
    if height > 0:
        histplot.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                          ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.savefig('static/images/visualization1.png')
plt.close()

# 2. Closing price over time
top_10_closing = data.nlargest(10, 'Close').sort_values('Date')
plt.figure(figsize=(10, 6))
plt.plot(top_10_closing['Date'], top_10_closing['Close'], marker='o', color='blue')
plt.title('Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)

# Add labels to each point
for i in range(len(top_10_closing)):
    plt.text(top_10_closing['Date'].values[i], top_10_closing['Close'].values[i], 
             f"{top_10_closing['Close'].values[i]:.2f}", fontsize=9, ha='right')

plt.savefig('static/images/visualization2.png')
plt.close()

# 3. Volume traded over time
top_10_volume = data.nlargest(10, 'Volume').sort_values('Date')
plt.figure(figsize=(10, 6))
plt.plot(top_10_volume['Date'], top_10_volume['Volume'], marker='o', color='green')
plt.title('Volumes Traded Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.xticks(rotation=45)

# Add labels to each point
for i in range(len(top_10_volume)):
    plt.text(top_10_volume['Date'].values[i], top_10_volume['Volume'].values[i], 
             f"{top_10_volume['Volume'].values[i]:.2f}", fontsize=9, ha='right')

plt.savefig('static/images/visualization3.png')
plt.close()

# 4. Open vs Close price
bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
data['Open_Binned'] = pd.cut(data['Open'], bins)
data['Close_Binned'] = pd.cut(data['Close'], bins)

# Count the occurrences in each bin
open_counts = data['Open_Binned'].value_counts().sort_index()
close_counts = data['Close_Binned'].value_counts().sort_index()

# Create a pie chart for Open prices
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
open_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Reds'))
plt.title('Open Price Distribution')

# Create a pie chart for Close prices
plt.subplot(1, 2, 2)
close_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Blues'))
plt.title('Close Price Distribution')

plt.savefig('static/images/visualization4.png')
plt.close()

# 5. High vs Low price
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['High', 'Low']])
plt.title('Box Plot of High vs Low Prices')
plt.xlabel('Price Type')
plt.ylabel('Price')

# Add labels to each box plot
for i in range(2):
    box = sns.boxplot(data=data[['High', 'Low']])
    box.annotate(f'{data.describe().loc["mean", ["High", "Low"]][i]:.2f}', 
                 xy=(i, data.describe().loc["mean", ["High", "Low"]][i]), 
                 xytext=(i, data.describe().loc["mean", ["High", "Low"]][i] + 10),
                 ha='center', color='black')

plt.savefig('static/images/visualization5.png')
plt.close()

# 6. Correlation heatmap
data['Rolling_Close'] = data['Close'].rolling(window=10).mean()
top_10_rolling = data.nlargest(10, 'Rolling_Close').sort_values('Date')

plt.figure(figsize=(10, 6))
barplot = sns.barplot(x=top_10_rolling['Date'], y=top_10_rolling['Rolling_Close'], color='orange')
plt.title('Rolling Average of Closing Price')
plt.xlabel('Date')
plt.ylabel('Rolling Average Close Price')
plt.xticks(rotation=45)

# Add labels to the bar plot
for p in barplot.patches:
    height = p.get_height()
    if not np.isnan(height):
        barplot.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.savefig('static/images/visualization6.png')
plt.close()