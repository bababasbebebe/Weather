
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("weather.csv")
df['date'] = pd.to_datetime(df['date'])

# Обработка пропусков и преобразование типов
for col in df.columns:
    if df[col].dtype == 'object' and col != 'date' and col != 'codesum':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Сводная статистика
summary_stats = df.describe()
summary_stats.to_csv("summary_statistics_new.csv")

# Группировка по станциям
station_summary = df.groupby("station_nbr").mean(numeric_only=True).reset_index()
station_summary.to_csv("station_summary.csv", index=False)

# Визуализация температуры
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="tavg", hue="station_nbr", legend=False)
plt.title("Средняя температура по времени для каждой метеостанции")
plt.xlabel("Дата")
plt.ylabel("Средняя температура (°F)")
plt.tight_layout()
plt.savefig("tavg_over_time.png")
plt.close()

# Визуализация осадков
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="preciptotal", hue="station_nbr", legend=False)
plt.title("Общее количество осадков по времени")
plt.xlabel("Дата")
plt.ylabel("Осадки (дюймы)")
plt.tight_layout()
plt.savefig("precip_total_over_time.png")
plt.close()
