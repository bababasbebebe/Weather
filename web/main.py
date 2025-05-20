from flask import Flask, render_template
import folium
from folium.plugins import HeatMap
import pandas as pd
import os
import random

app = Flask(__name__)

# Конфигурация - путь к файлу с данными
DATA_FILE = 'data/heatmap_data.csv'


def load_heatmap_data():
    """Загружает данные из файла"""
    try:
        if DATA_FILE.endswith('.csv'):
            df = pd.read_csv(DATA_FILE)
        elif DATA_FILE.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(DATA_FILE)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")

        required_columns = ['latitude', 'longitude', 'intensity']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

        return df[required_columns].values.tolist()

    except Exception as e:
        print(f"Error loading data: {e}")
        return generate_sample_data()


def generate_sample_data(num_points=100):
    """Генерирует тестовые данные"""
    data = []
    for _ in range(num_points):
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        intensity = random.uniform(0.1, 1.0)
        data.append([lat, lon, intensity])
    return data


@app.route('/')
def show_map():
    """Главная страница с тепловой картой"""
    # Создаем базовую карту мира с атрибуцией
    world_map = folium.Map(
        location=[30, 0],
        zoom_start=2,
        tiles='OpenStreetMap',
        attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )

    heat_data = load_heatmap_data()

    # Добавляем тепловой слой
    HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(world_map)

    # Добавляем маркеры
    for point in heat_data:
        folium.CircleMarker(
            location=[point[0], point[1]],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"Intensity: {point[2]:.2f}"
        ).add_to(world_map)

    # Добавляем альтернативные слои карты с атрибуциями
    folium.TileLayer(
        'Stamen Terrain',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(world_map)

    folium.TileLayer(
        'CartoDB dark_matter',
        attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>'
    ).add_to(world_map)

    folium.LayerControl().add_to(world_map)

    # Создаем папки если их нет
    os.makedirs('templates', exist_ok=True)

    # Сохраняем карту
    world_map.save('templates/map.html')

    return render_template('map.html')


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    if not os.path.exists(DATA_FILE):
        sample_data = pd.DataFrame(
            generate_sample_data(),
            columns=['latitude', 'longitude', 'intensity']
        )
        sample_data.to_csv(DATA_FILE, index=False)
        print(f"Sample data generated at {DATA_FILE}")

    app.run(debug=True, host='0.0.0.0', port=5000)