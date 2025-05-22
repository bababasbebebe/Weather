
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv("weather.csv")
df['date'] = pd.to_datetime(df['date'])

# Обработка пропущенных и нечисловых значений
for col in df.columns:
    if df[col].dtype == 'object' and col != 'date' and col != 'codesum':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Сводная статистика
summary_stats = df.describe()
summary_stats.to_csv("summary_statistics_new.csv")

# Группировка по метеостанциям
station_summary = df.groupby("station_nbr").mean(numeric_only=True).reset_index()
station_summary.to_csv("station_summary.csv", index=False)

# График: средняя температура по времени
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="tavg", hue="station_nbr", legend=False)
plt.title("Средняя температура по времени для каждой метеостанции")
plt.xlabel("Дата")
plt.ylabel("Средняя температура (°F)")
plt.tight_layout()
plt.savefig("tavg_over_time.png")
plt.close()

# График: осадки по времени
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="preciptotal", hue="station_nbr", legend=False)
plt.title("Общее количество осадков по времени")
plt.xlabel("Дата")
plt.ylabel("Осадки (дюймы)")
plt.tight_layout()
plt.savefig("precip_total_over_time.png")
plt.close()

# Матрица корреляций
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Корреляционная матрица признаков")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()
