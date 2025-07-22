import streamlit as st
import pickle
import pandas as pd
import random

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Sentimen Steam",
    page_icon="🕵️",
    layout="wide"
)

# --- CSS KUSTOM UNTUK MENGURANGI JARAK ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# --- FUNGSI UNTUK MEMUAT SEMUA MODEL ---
@st.cache_resource
def load_models_and_vectorizer():
    with open('naive_bayes_model.pkl', 'rb') as model_file:
        model_nb = pickle.load(model_file)
    with open('knn_model.pkl', 'rb') as model_file:
        model_knn = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model_nb, model_knn, vectorizer

# --- MEMUAT MODEL ---
model_nb, model_knn, vectorizer = load_models_and_vectorizer()


# --- UTAMA APLIKASI ---
st.title("Analisis Sentimen Ulasan Aplikasi Steam 🕵️")
st.markdown("Aplikasi ini membandingkan performa model **Naive Bayes** dan **K-Nearest Neighbors (KNN)** untuk analisis sentimen.")
st.markdown("---")

# --- PENGATURAN MODEL (DI HALAMAN UTAMA) ---
st.subheader("⚙️ Pengaturan Model")
model_choice = st.radio(
    "Pilih Algoritma Klasifikasi:",
    ('Naive Bayes', 'K-Nearest Neighbors (KNN)'),
    key='model_selection',
    horizontal=True,
)
st.markdown("---")

# --- INPUT SECTION ---
st.subheader("📝 Input Ulasan Anda")

contoh_positif_list = [
    "Aplikasinya keren dan sangat membantu, banyak diskon game!",
    "aplikasinya mudah digunakan",
    "jadi mudah beli game dari hp, tidak perlu buka PC",
    "awalnya lupa password, ngurus lewat hp jadi mudah, cepat ditangani, sangat bagus",
    "sekarang gampang akses info tentang steam",
    "Download game dari kantor sangat praktis"
]
contoh_negatif_list = [
    "aplikasinya burik banget susah",
    "Tidak semua fitur desktop tersedia di mobile.",
    "Tidak bisa melihat badge atau Steam Points jelas",
    "Chat sering tidak sinkron dengan versi desktop",
    "Market lambat banget, udah kayak pakai internet 2G",
    "Chat nggak nyampe, terus malah error pas dikirim ulang",
]

col1, col2 = st.columns(2)
with col1:
    if st.button('Coba Contoh Positif', use_container_width=True):
        st.session_state.user_input = random.choice(contoh_positif_list)
with col2:
    if st.button('Coba Contoh Negatif', use_container_width=True):
        st.session_state.user_input = random.choice(contoh_negatif_list)

with st.form(key='sentiment_form'):
    user_input = st.text_area(
        "Masukkan ulasan untuk dianalisis:",
        key='user_input',
        height=150
    )
    submit_button = st.form_submit_button(
        label='Analisis Sentimen',
        use_container_width=True
        )

# --- HASIL ANALISIS (DI BAWAH INPUT) ---
if submit_button and user_input:
    st.markdown("---")
    st.subheader("📊 Hasil Analisis")

    model = model_nb if model_choice == 'Naive Bayes' else model_knn
    
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    prediction_proba = model.predict_proba(input_vector)
    
    sentiment = prediction[0]
    prob_negatif = prediction_proba[0][0]
    prob_positif = prediction_proba[0][1]
    
    if sentiment == 'positif':
        st.success(f"**Hasil Prediksi: Sentimen Positif** ✅")
    else:
        st.error(f"**Hasil Prediksi: Sentimen Negatif** ❌")
    
    st.metric(label=f"Tingkat Keyakinan ({model_choice})", value=f"{max(prob_positif, prob_negatif):.2%}")

    if model_choice == 'Naive Bayes':
        st.info(
            """
            💡 **Penjelasan Keyakinan Naive Bayes:**
            Skor ini dihitung berdasarkan **probabilitas (peluang)**. Model menghitung seberapa besar kemungkinan ulasan Anda tergolong 'positif' atau 'negatif' berdasarkan kemunculan kata-kata di dalamnya, lalu membandingkannya dengan pola kata pada data yang sudah dilatih.
            """
        )
    else: 
        st.info(
            """
            💡 **Penjelasan Keyakinan KNN:**
            Skor ini ditentukan oleh **'tetangga terdekat'**. Model mencari beberapa ulasan di data latih yang paling mirip dengan ulasan Anda. Persentase ini menunjukkan berapa banyak dari 'tetangga' tersebut yang sentimennya sama dengan hasil prediksi.
            """
        )

# --- TABS UNTUK INFORMASI TAMBAHAN ---
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📜 Tentang Aplikasi", "🧠 Cara Kerja", "📈 Performa Model"])

with tab1:
    st.subheader("Deskripsi Proyek")
    st.markdown("""
    Aplikasi ini merupakan bagian dari pengerjaan Skripsi di bidang Ilmu Komputer dengan fokus pada *Natural Language Processing* (NLP) dan *Machine Learning*.
    - **Tujuan**: Menganalisis dan membandingkan kinerja algoritma Naive Bayes dan K-Nearest Neighbors untuk klasifikasi sentimen pada ulasan aplikasi Steam.
    - **Metodologi**: CRISP-DM.
    - **Dibuat oleh**: Farhan Pratama Putra - 10121440 - Universitas Gunadarma
    """)

with tab2:
    st.subheader("Alur Kerja Proses Analisis")
    st.markdown("""
    1.  **Input Teks**: Teks ulasan yang Anda masukkan.
    2.  **TF-IDF Vectorization**: Teks diubah menjadi representasi numerik yang mengukur pentingnya sebuah kata dalam ulasan.
    3.  **Klasifikasi**: Vektor numerik tersebut dimasukkan ke dalam model machine learning yang dipilih untuk memprediksi label sentimen **Positif** atau **Negatif**.
    """)

with tab3:
    st.subheader("Tabel Perbandingan Kinerja Model")
    performance_data = {
        'Model': ['Naive Bayes', 'KNN'],
        'AAccuracy': [0.80, 0.65],
        'Precision': [0.81, 0.64],
        'Recall': [0.72, 0.65],
        'F1-Score': [0.74, 0.63]
    }
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True, hide_index=True)
    st.caption("Performa dihitung berdasarkan *Macro Average* pada data uji.")

    # --- BLOK PENJELASAN METRIK BARU
    st.markdown("---") 
    st.subheader("Penjelasan Singkat Metrik")
    st.markdown(
        """
        - **Accuracy**: Persentase seberapa sering model menebak sentimen (baik positif maupun negatif) dengan benar dari keseluruhan data.
        - **Precision**: Mengukur ketepatan prediksi. Dari semua yang ditebak sebagai 'positif', berapa persen yang benar? Begitu pula untuk 'negatif'.
        - **Recall**: Mengukur kelengkapan model. Dari semua yang seharusnya 'positif', berapa persen yang berhasil ditemukan model? Begitu pula untuk 'negatif'.
        - **F1-Score**: Nilai gabungan yang menyeimbangkan antara Presisi dan Recall, memberikan satu angka performa yang komprehensif.
        """
    )