from flask import Flask, render_template, request, redirect, url_for, flash
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from datetime import datetime
#from sklearn.preprocessing import StandardScaler
from scipy.interpolate import BSpline


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
app.secret_key = 'your-secret-key-here'  # Необходимо для flash-сообщений


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def data_preparation(df):

    df = df[['date', 'tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat', 'cool',
       'sunrise', 'sunset', 'snowfall', 'preciptotal', 'stnpressure',
       'sealevel', 'resultspeed', 'resultdir', 'avgspeed', 'codesum']]

    df[['tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat', 'cool', 'sunrise', 'sunset', 'resultdir']] = \
        df[['tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat', 'cool', 'sunrise', 'sunset', \
                   'resultdir']].astype('int')

    df = df.replace('  T', '0.001')  # Замена Trace на маленькое 0.001
    df[['snowfall', 'preciptotal', 'stnpressure', 'sealevel', 'resultspeed', 'avgspeed']] = \
        df[['snowfall', 'preciptotal', 'stnpressure', 'sealevel', 'resultspeed', 'avgspeed']].astype('float')

    codesum_cat = ['+FC', 'FC', 'TS', 'GR', 'RA', 'DZ', 'SN', 'SG', 'GS', 'PL', 'IC', 'FG+', 'FG', 'BR', 'UP', \
                   'HZ', 'FU', 'VA', 'DU', 'DS', 'PO', 'SA', 'SS', 'PY', 'SQ', 'DR', 'SH', 'FZ', 'MI', 'PR', \
                   'BC', 'BL', 'VC']  # Вариации кодов

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

    df = df.drop(['codesum'], axis=1)

    try:
        date = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"Error loading dates: {e}")

    #scaler = StandardScaler()
    #df = scaler.fit_transform(df.drop('date', axis=1))
    df = df.drop('date', axis=1)
    return df, date


def load_heatmap_data():
    """Загружает данные из файла, возвращает None если формат неправильный"""
    try:
        df = pd.read_csv('data/test.csv ')

        # Проверяем обязательные колонки
        required_columns = ['date', 'tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat', 'cool',
       'sunrise', 'sunset', 'snowfall', 'preciptotal', 'stnpressure',
       'sealevel', 'resultspeed', 'resultdir', 'avgspeed', 'codesum']
        if not any(col in df.columns for col in required_columns):
            return None

        df, date = data_preparation(df)
        print(df)
        preds = get_intencity(df)
        print(preds)
        return df, date

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_coefs():
    try:
        return pd.read_csv('data/weather_coefs.csv')
    except Exception as e:
        print(f"Error get coords: {e}")

def get_intencity(df):
    coefs = get_coefs()

    def build_spline_matrix(X, n_splines=20, spline_order=3):
        # Узлы (knots) равномерно распределены по квантилям
        knots = np.linspace(X.min(), X.max(), n_splines - spline_order + 1)

        # Добавляем граничные узлы (extended knots)
        extended_knots = np.r_[
            [X.min()] * spline_order,
            knots,
            [X.max()] * spline_order
        ]

        # Создаем базисные функции B-сплайнов
        basis = np.zeros((len(X), n_splines))
        for i in range(n_splines):
            coefs = np.zeros(n_splines)
            coefs[i] = 1
            spline = BSpline(extended_knots, coefs, spline_order)
            basis[:, i] = spline(X)

        return basis


    try:
        pred = []
        for i in range(df[:,1].shape[0]):
            spline_part = build_spline_matrix(df[:, 1][i])
            intercept = np.zeros((len(df[:, 1][i]), 1))
            model_matrix = np.hstack([intercept, spline_part])
            sum = 0
            for city in ['Dallas']:
                for j in range(1, 112):
                    sum += model_matrix @ coefs.loc[1, city].values
            pred.append(sum)
        return pred


    except Exception as e:
        print(f"Error: {e}")

def create_timestamped_map(df, date):
    """Создает карту с анимированным тепловым слоем по датам"""
    if df is None or df.empty:
        # Для показа прототипа
        features = []
        prototype = pd.read_csv('data/prototype.csv')
        for i in range(prototype.shape[0]):
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [prototype.loc[i, 'longitude'], prototype.loc[i, 'latitude']]
                },
                'properties': {
                    #'time': date,
                    'intensity': prototype.loc[i, 'intencity'],
                    'style': {'color': get_color(prototype.loc[i, 'intencity'])}
                }
            })


        m = folium.Map(location=[30, 0], zoom_start=2, tiles='OpenStreetMap')
        TimestampedGeoJson(
            {'type': 'FeatureCollection', 'features': features},
            period='P1M',
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=1,
            loop_button=False,
            date_options='YYYY-MM-DD',
            time_slider_drag_update=True
        ).add_to(m)
        return m._repr_html_()

    m = folium.Map(location=[30, 0], zoom_start=2, tiles='OpenStreetMap')

    # Преобразуем даты в строки для JSON
    date = date.dt.strftime('%Y-%m-%d')

    # Создаем данные для TimestampedGeoJson
    features = []
    for i in range(date.shape[0]):
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['longitude'], row['latitude']]
            },
            'properties': {
                'time': date,
                'intensity': 0.5,
                'style': {'color': get_color(row['intensity'])}
            }
        })

    # Добавляем анимированный слой
    TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period='P1M',
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD',
        time_slider_drag_update=True
    ).add_to(m)

    return m._repr_html_()


def get_color(intensity):
    """Возвращает цвет в зависимости от интенсивности"""
    if intensity > 0.8:
        return 'red'
    elif intensity > 0.5:
        return 'orange'
    else:
        return 'yellow'


@app.route('/', methods=['GET', 'POST'])
def index():
    df = None
    date = None
    min_date = ""
    max_date = ""

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename('heatmap_data.csv')
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Проверяем загруженный файл
                df, date = load_heatmap_data()
                if df is None:
                    flash(
                        'Ошибка: Файл',
                        'error')
                    os.remove(filepath)  # Удаляем невалидный файл
                else:
                    flash('Файл успешно загружен', 'success')
            elif file.filename != '':
                flash('Ошибка: Разрешены только файлы CSV или Excel', 'error')

    # Получаем диапазон дат если данные есть
    if date is not None and not date.empty:
        min_date = date.min().strftime('%Y-%m-%d')
        max_date = date.max().strftime('%Y-%m-%d')

    map_html = create_timestamped_map(df, date)

    return render_template(
        'index.html',
        map_html=map_html,
        min_date=min_date,
        max_date=max_date
    )


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
