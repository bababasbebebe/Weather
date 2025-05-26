from flask import Flask, render_template, request, redirect, url_for, flash
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import json
from folium.plugins import HeatMapWithTime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
app.secret_key = '123'

# Координаты городов
city_coords = {
             'New York': [np.float64(40.7128), np.float64(-74.006)],
             'Los Angeles': [np.float64(34.0522), np.float64(-118.2437)],
             'Chicago': [np.float64(41.8781), np.float64(-87.6298)],
             'Houston': [np.float64(29.7604), np.float64(-95.3698)],
             'Phoenix': [np.float64(33.4484), np.float64(-112.074)],
             'Philadelphia': [np.float64(39.9526), np.float64(-75.1652)],
             'San Antonio': [np.float64(29.4241), np.float64(-98.4936)],
             'San Diego': [np.float64(32.7157), np.float64(-117.1611)],
             'Dallas': [np.float64(32.7767), np.float64(-96.797)],
             'San Jose': [np.float64(37.3382), np.float64(-121.8863)],
             'Austin': [np.float64(30.2672), np.float64(-97.7431)],
             'Jacksonville': [np.float64(30.3322), np.float64(-81.6557)],
             'Fort Worth': [np.float64(32.7555), np.float64(-97.3308)],
             'Columbus': [np.float64(39.9612), np.float64(-82.9988)],
             'Charlotte': [np.float64(35.2271), np.float64(-80.8431)],
             'San Francisco': [np.float64(37.7749), np.float64(-122.4194)],
             'Indianapolis': [np.float64(39.7684), np.float64(-86.1581)],
             'Seattle': [np.float64(47.6062), np.float64(-122.3321)],
             'Denver': [np.float64(39.7392), np.float64(-104.9903)],
             'Washington': [np.float64(38.9072), np.float64(-77.0369)]
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_weather_file(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        if df.isnull().any(axis=1).any() or df.replace('M', np.nan).isnull().any(axis=1).any():
            raise ValueError("Обнаружены строки с пропущенными значениями в weather.csv")
        df = df[['date', 'tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat',
               'cool', 'sunrise', 'sunset', 'codesum', 'snowfall', 'preciptotal',
               'stnpressure', 'sealevel', 'resultspeed', 'resultdir', 'avgspeed']]
        df[['tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat', 'cool', 'sunrise', 'sunset',
                      'resultdir']] = \
            df[['tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat', 'cool', 'sunrise', 'sunset', \
                          'resultdir']].astype('int')

        df = df.replace('  T', '0.001')  # Замена Trace на маленькое 0.001
        df[['snowfall', 'preciptotal', 'stnpressure', 'sealevel', 'resultspeed', 'avgspeed']] = \
            df[['snowfall', 'preciptotal', 'stnpressure', 'sealevel', 'resultspeed', 'avgspeed']].astype(
                'float')

        df['date'] = pd.to_datetime(df['date'])

        codesum_cat = ['+FC', 'FC', 'TS', 'GR', 'RA', 'DZ', 'SN', 'SG', 'GS', 'PL', 'IC', 'FG+', 'FG', 'BR', 'UP', \
                       'HZ', 'FU', 'VA', 'DU', 'DS', 'PO', 'SA', 'SS', 'PY', 'SQ', 'DR', 'SH', 'FZ', 'MI', 'PR', \
                       'BC', 'BL', 'VC']

        for code in codesum_cat:
            df[code] = [False for _ in range(df.shape[0])]  # One-Hot-Encoding

        # Заполнение Ohe
        for i in range(df.shape[0]):
            codes = df.loc[i, 'codesum'].split()
            for code in codes:
                if len(code) > 3:
                    df.loc[i, code[:2]] = True
                    df.loc[i, code[2:]] = True
                else:
                    df.loc[i, code] = True

        df = df.drop('codesum', axis=1)
        return df

    except Exception as e:
        print(f"[Ошибка] Обработка weather.csv: {e}")
        return None

def compute_intensity(weather_df: pd.DataFrame, coefs_df: pd.DataFrame, city_coords: dict, selected_cities=None, selected_products=None) -> pd.DataFrame:
    result_rows = []

    for _, row in weather_df.iterrows():
        date = row['date'].strftime('%Y-%m-%d')
        features = row.drop('date').values

        for city, coords in city_coords.items():
            if selected_cities and city not in selected_cities:
                continue

            products = {}
            total_intensity = 0

            for product_id in coefs_df.index:
                if selected_products and str(product_id) not in selected_products:
                    continue

                # ИЗМЕНИТЬ НА КОЭФЫ
                coefs = np.random.rand(len(features),1)
                result = np.dot(features, coefs)[0]
                if result > 0:
                    products[int(product_id)] = result
                    total_intensity += result

            result_rows.append({
                'date': date,
                'latitude': coords[0],
                'longitude': coords[1],
                'city': city,
                'intensity': total_intensity,
                'products': json.dumps(products)
            })

    result_rows = pd.DataFrame(result_rows)

    # intensity scale
    result_rows['intensity'] = (result_rows['intensity'] - result_rows['intensity'].min(axis=0)) / (result_rows['intensity'].max(axis=0) - result_rows['intensity'].min(axis=0))
    result_rows.loc[2, 'intensity'] = 0.6

    return result_rows

def get_color(intensity):
    if intensity > 0.8:
        return 'red'
    elif intensity > 0.5:
        return 'orange'
    else:
        return 'yellow'

def create_timestamped_map(df):
    if df is None or df.empty:
        return folium.Map(location=[30, 0], zoom_start=2)._repr_html_()

    df['date'] = pd.to_datetime(df['date'])
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    grouped = df.groupby('date_str')
    time_index = []
    heat_data = []

    m = folium.Map(location=[55, 37], zoom_start=4, tiles='CartoDB dark_matter')

    for date, group in grouped:
        time_index.append(date)
        day_heat = []

        for _, row in group.iterrows():
            day_heat.append([row['latitude'], row['longitude'], row['intensity']])

            popup_text = f"<b>{row['city']}</b><br>"
            product_dict = json.loads(row['products'])
            popup_text += '<br>'.join([f"{k}: {int(v)}" for k, v in product_dict.items() if v > 0])

            folium.CircleMarker(
                location=(row['latitude'], row['longitude']),
                radius=1,
                color='white',
                fill=True,
                fill_color='white',
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(m)

        heat_data.append(day_heat)

    HeatMapWithTime(
        heat_data,
        index=time_index,
        auto_play=False,
        max_opacity=0.8,
        radius=20,
        use_local_extrema=True
    ).add_to(m)

    return m._repr_html_()

@app.route('/', methods=['GET', 'POST'])
def index():
    heatmap_df = None
    min_date = ""
    max_date = ""
    selected_cities = request.form.getlist('cities')
    selected_products = request.form.getlist('products')

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename('weather.csv')
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                weather_df = process_weather_file(filepath)
                if weather_df is None:
                    flash('Ошибка: Невозможно обработать weather.csv', 'error')
                    os.remove(filepath)
                else:
                    try:
                        coefs_df = pd.read_csv('data/coefficients.csv', index_col=0)
                        heatmap_df = compute_intensity(weather_df, coefs_df, city_coords, selected_cities, selected_products)
                        flash('Файл успешно загружен и обработан', 'success')
                    except Exception as e:
                        flash(f'Ошибка при вычислении карты: {e}', 'error')

            elif file.filename != '':
                flash('Ошибка: Разрешены только файлы CSV или Excel', 'error')

    if heatmap_df is not None and not heatmap_df.empty:
        dates = pd.to_datetime(heatmap_df['date'])
        min_date = dates.min().strftime('%Y-%m-%d')
        max_date = dates.max().strftime('%Y-%m-%d')

    map_html = create_timestamped_map(heatmap_df)

    return render_template(
        'index.html',
        map_html=map_html,
        min_date=min_date,
        max_date=max_date,
        cities=city_coords.keys(),
        products=[str(i) for i in range(1, 112)]
    )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
