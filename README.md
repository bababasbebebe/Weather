
# 🌤 Weather EDA by City

Анализ данных о погоде в разных городах на основе CSV-файла `weather_by_city.csv`.

## 📁 Содержимое проекта

| Файл | Описание |
|------|----------|
| `weather_by_city.csv` | Исходный датасет |
| `weather_eda_script.py` | Python-скрипт для анализа |
| `summary_statistics.csv` | Сводная статистика по всем данным |
| `city_summary.csv` | Средние значения температуры и осадков по городам |
| `avg_temp_over_time.png` | График изменения средней температуры |
| `precipitation_over_time.png` | График изменения осадков |

## 📊 Шаги анализа (EDA)

1. Загрузка и первичный осмотр данных.
2. Преобразование формата даты.
3. Сводная статистика по всем данным (`summary_statistics.csv`).
4. Агрегация данных по городам (`city_summary.csv`).
5. Визуализация температуры и осадков во времени.

### Средняя температура по времени
![Avg Temp Over Time](avg_temp_over_time.png)

### Осадки по времени
![Precipitation Over Time](precipitation_over_time.png)

## 📌 Основные выводы

- Температура и осадки варьируются по городам и по времени.
- Наблюдаются сезонные тренды в данных.
- Отсутствуют пропуски, данные готовы для дальнейшего анализа или ML.

## 🚀 Как запустить

```bash
pip install pandas matplotlib seaborn
python weather_eda_script.py
```
