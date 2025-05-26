
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Загрузка данных
df = pd.read_csv('weather_clean.csv')

# Основная информация
print(df.info())
print(df.describe())

# Анализ баланса классов (если есть категориальный столбец с классами)
if 'class' in df.columns:
    print(df['class'].value_counts(normalize=True) * 100)

# Матрица корреляций
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Матрица корреляций")
plt.show()

# BoxPlot для продаж по станциям (замени 'sales' и 'station' на свои столбцы)
if 'sales' in df.columns and 'station' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='station', y='sales', data=df)
    plt.title("Распределение продаж по станциям")
    plt.show()

# Обнаружение выбросов (пример для столбца sales)
if 'sales' in df.columns:
    Q1 = df['sales'].quantile(0.25)
    Q3 = df['sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['sales'] < lower_bound) | (df['sales'] > upper_bound)]
    print(f"Найдено выбросов в 'sales': {len(outliers)}")
    df['sales'] = df['sales'].clip(lower_bound, upper_bound)  # Замена выбросов

# Feature Importance
features = [col for col in df.columns if col not in ['sales', 'date', 'station', 'class']]
df = df.dropna()
X = df[features]
y = df['sales']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("Feature Importance:")
print(importances)

plt.figure(figsize=(10, 6))
importances.plot(kind='bar')
plt.title("Важность признаков")
plt.show()
