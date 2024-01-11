import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import locale
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set level warning ke 'ignore' untuk mengabaikan peringatan ini
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def set_locale():
    # Set local ke Indonesia
    locale.setlocale(locale.LC_ALL, 'id_ID')

try:
    # Load dataset
    data = pd.read_csv('Price Range Phone Dataset.csv')

    # Pisahkan fitur (X) dan target (y)
    X = data.drop('price_range', axis=1)
    y = data['price_range']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Membuat model Decision Tree Classifier
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)

    # Melakukan prediksi
    decision_tree_prediction = decision_tree_model.predict(X_test)

    # Menampilkan confusion matrix dan classification report
    st.subheader('Confusion Matrix:')
    st.write(confusion_matrix(y_test, decision_tree_prediction))

    st.subheader('Classification Report:')
    st.write(classification_report(y_test, decision_tree_prediction))
    
    # Load Decision Tree model
    decision_tree_model = pickle.load(open('DecisionTreeClassifier.pkl', 'rb'))
except Exception as e:
    st.error(f"Terjadi kesalahan dalam memuat dataset atau model: {e}")

# Streamlit App
set_locale()
st.title('Aplikasi Prediksi Harga Telepon')
st.sidebar.header('Input Fitur')

def user_input_features():
    features = {}
    for column in data.columns:
        features[column] = st.sidebar.slider(f'{column}', int(data[column].min()), int(data[column].max()), int(data[column].mean()))
    return pd.DataFrame(features, index=[0])

user_input = user_input_features()

st.subheader('Data Fitur:')
st.write(user_input)

# Pilih hanya fitur yang digunakan saat melatih model
selected_features = user_input[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory',
                                'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w',
                                'talk_time', 'three_g', 'touch_screen', 'wifi']]

# Prediksi harga dengan model Decision Tree
decision_tree_prediction = decision_tree_model.predict(selected_features.values)

# Konversi ke format mata uang Rupiah
formatted_decision_tree_prediction = f"IDR {float(decision_tree_prediction[0]):,.2f}"

# Tampilkan hasil prediksi Decision Tree
st.subheader('Hasil Prediksi Harga Telepon (Decision Tree):')

# Tampilkan kelas yang diprediksi dalam format mata uang Rupiah
st.write(f'Harga yang Diprediksi: {formatted_decision_tree_prediction}')

# Informasi tentang dataset
if st.checkbox("Detail Kumpulan Dataset Fitur Telepon"):
    st.subheader('Detail Dataset Fitur:')
    st.write(data)
    
    fig = px.pie(data, names='price_range', title='Distribusi Kisaran Harga Telepon')
    st.plotly_chart(fig)

    st.bar_chart(data['price_range'].value_counts())

    # Line chart untuk menunjukkan tren harga terhadap ram
    st.subheader('Tren Harga vs RAM:')
    fig_line = px.line(data, x='ram', y='price_range', title='Tren Harga vs RAM')
    st.plotly_chart(fig_line)

    # Area chart untuk menunjukkan persebaran harga berdasarkan px_height dan px_width
    st.subheader('Persebaran Harga berdasarkan Px_Height dan Px_Width:')
    fig_area = px.area(data, x='px_height', y='px_width', color='price_range', title='Persebaran Harga berdasarkan Px_Height dan Px_Width')
    st.plotly_chart(fig_area)
