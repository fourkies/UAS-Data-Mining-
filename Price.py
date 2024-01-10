import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import locale
import warnings

# Set level warning ke 'ignore' untuk mengabaikan peringatan ini
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set local ke Indonesia
locale.setlocale(locale.LC_ALL, 'id_ID')

# Load dataset
data = pd.read_csv('Price Range Phone Dataset.csv')

# Train-test split
X = data.drop('price_range', axis=1)
y = data['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Load Decision Tree model
decision_tree_model = pickle.load(open('DecisionTreeClassifier.pkl', 'rb'))

# Streamlit App
st.title('Aplikasi Prediksi Harga Telepon')
st.sidebar.header('Input Pengguna')

def user_input_features():
    battery_power = st.sidebar.slider('battery power', int(X['battery_power'].min()), int(X['battery_power'].max()), int(X['battery_power'].mean()))
    blue = st.sidebar.slider('blue', int(X['blue'].min()), int(X['blue'].max()), int(X['blue'].mean()))
    clock_speed = st.sidebar.slider('clock_speed', int(X['clock_speed'].min()), int(X['clock_speed'].max()), int(X['clock_speed'].mean()))
    dual_sim = st.sidebar.slider('dual_sim', int(X['dual_sim'].min()), int(X['dual_sim'].max()), int(X['dual_sim'].mean()))
    fc = st.sidebar.slider('fc', int(X['fc'].min()), int(X['fc'].max()), int(X['fc'].mean()))
    four_g = st.sidebar.slider('four_g', int(X['four_g'].min()), int(X['four_g'].max()), int(X['four_g'].mean()))
    int_memory = st.sidebar.slider('int_memory', int(X['int_memory'].min()), int(X['int_memory'].max()), int(X['int_memory'].mean()))
    m_dep = st.sidebar.slider('m_dep', int(X['m_dep'].min()), int(X['m_dep'].max()), int(X['m_dep'].mean()))
    mobile_wt = st.sidebar.slider('mobile_wt', int(X['mobile_wt'].min()), int(X['mobile_wt'].max()), int(X['mobile_wt'].mean()))
    n_cores = st.sidebar.slider('n_cores', int(X['n_cores'].min()), int(X['n_cores'].max()), int(X['n_cores'].mean()))
    pc = st.sidebar.slider('pc', int(X['pc'].min()), int(X['pc'].max()), int(X['pc'].mean()))
    px_height = st.sidebar.slider('px_height', int(X['px_height'].min()), int(X['px_height'].max()), int(X['px_height'].mean()))
    px_width = st.sidebar.slider('px_width', int(X['px_width'].min()), int(X['px_width'].max()), int(X['px_width'].mean()))
    ram = st.sidebar.slider('ram', int(X['ram'].min()), int(X['ram'].max()), int(X['ram'].mean()))
    sc_h = st.sidebar.slider('sc_h', int(X['sc_h'].min()), int(X['sc_h'].max()), int(X['sc_h'].mean()))
    sc_w = st.sidebar.slider('sc_w', int(X['sc_w'].min()), int(X['sc_w'].max()), int(X['sc_w'].mean()))
    talk_time = st.sidebar.slider('talk_time', int(X['talk_time'].min()), int(X['talk_time'].max()), int(X['talk_time'].mean()))
    threee_g = st.sidebar.slider('three_g', int(X['three_g'].min()), int(X['three_g'].max()), int(X['three_g'].mean()))
    touch_screen = st.sidebar.slider('touch_screen', int(X['touch_screen'].min()), int(X['touch_screen'].max()), int(X['touch_screen'].mean()))
    wifi = st.sidebar.slider('wifi', int(X['wifi'].min()), int(X['wifi'].max()), int(X['wifi'].mean()))
    
    data = {
        'battery_power': battery_power,
        'blue': blue,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'fc': fc,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': threee_g,
        'touch_screen': touch_screen,
        'wifi': wifi,
    }
    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

st.subheader('Data Pengguna:')
st.write(user_input)

# Prediksi harga dengan model Decision Tree
decision_tree_prediction = decision_tree_model.predict(user_input.values)
formatted_decision_tree_prediction = locale.currency(float(decision_tree_prediction[0]), grouping=True)

# Tampilkan hasil prediksi Decision Tree dalam format Rupiah
st.subheader('Hasil Prediksi Harga Telepon (Decision Tree):')
st.write(formatted_decision_tree_prediction)

# Informasi tentang dataset
if st.checkbox("Detail Kumpulan Data Telepon Kisaran Harga"):
    st.subheader('Detail Dataset:')
    st.write(data)
    st.bar_chart(data['price_range'].value_counts())
