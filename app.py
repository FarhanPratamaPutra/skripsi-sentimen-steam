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


# --- FUNGSI UNTUK MEMUAT MODEL (HANYA NAIVE BAYES) ---
@st.cache_resource
def load_model_and_vectorizer():
    with open('naive_bayes_model.pkl', 'rb') as model_file:
        model_nb = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model_nb, vectorizer

# --- MEMUAT MODEL ---
model, vectorizer = load_model_and_vectorizer()

# --- SIDEBAR DIHAPUS ---

# --- UTAMA APLIKASI ---
st.title("Analisis Sentimen Ulasan Aplikasi Steam 🕵️")
st.markdown("---")

# --- TATA LETAK KOLOM ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📝 Input Ulasan Anda")

    contoh_positif_list = [
        "Aplikasinya keren dan sangat membantu, banyak diskon game!",
        "Mantap, proses trade jadi sangat mudah dan aman lewat hp",
        "aplikasinya mudah digunakan",
        "jadi mudah beli game dari hp, tidak perlu buka PC",
        "awalnya lupa password, ngurus lewat hp jadi mudah, cepat ditangani, sangat bagus",
        "sekarang gampang akses info tentang steam",
        "Cek diskon musiman sekarang gampang banget, tinggal buka aplikasi",
        "Notifikasi flash sale langsung ke ponsel",
        "Download game dari kantor sangat praktis",
        "Sangat membantu saat Steam Summer Sale."
    ]
    contoh_negatif_list = [
        "aplikasinya burik banget susah",
        "Autentikasi kadang gagal tanpa alasan jelas",
        "Tidak bisa lihat semua detail library",
        "Tidak semua fitur desktop tersedia di mobile.",
        "Tidak bisa melihat badge atau Steam Points jelas",
        "Chat sering tidak sinkron dengan versi desktop",
        "Ganti akun ribet, harus hapus data aplikasi",
        "Market lambat banget, udah kayak pakai internet 2G",
        "Chat nggak nyampe, terus malah error pas dikirim ulang",
        "Sering error waktu buka halaman komunitas atau berita"
    ]

    if st.button('Coba Contoh Positif', use_container_width=True):
        st.session_state.user_input = random.choice(contoh_positif_list)
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

with col2:
    st.subheader("📊 Hasil Analisis")

    if submit_button and user_input:
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
        
        st.metric(label="Tingkat Keyakinan", value=f"{max(prob_positif, prob_negatif):.2%}")

    else:
        st.info("Silakan masukkan ulasan dan klik tombol analisis untuk melihat hasilnya.")

# --- TABS UNTUK INFORMASI TAMBAHAN ---
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📜 Tentang Aplikasi", "🧠 Cara Kerja", "📈 Performa Model"])

with tab1:
    st.subheader("Deskripsi Proyek")
    st.markdown("""
    Aplikasi ini merupakan bagian dari pengerjaan Skripsi di bidang Ilmu Komputer dengan fokus pada *Natural Language Processing* (NLP) dan *Machine Learning*.
    - **Tujuan**: Mengimplementasikan model Naive Bayes untuk klasifikasi sentimen pada ulasan aplikasi Steam.
    - **Metodologi**: CRISP-DM.
    - **Dibuat oleh**: [Nama Anda] - [NIM Anda] - [Universitas Anda]
    """)

with tab2:
    st.subheader("Alur Kerja Proses Analisis")
    st.markdown("""
    1.  **Input Teks**: Teks ulasan yang Anda masukkan.
    2.  **TF-IDF Vectorization**: Teks diubah menjadi representasi numerik yang mengukur pentingnya sebuah kata dalam ulasan.
    3.  **Klasifikasi**: Vektor numerik tersebut dimasukkan ke dalam model Naive Bayes yang telah dilatih untuk memprediksi label sentimen **Positif** atau **Negatif**.
    """)

with tab3:
    st.subheader("Tabel Kinerja Model Naive Bayes")
    performance_data = {
        'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
        'Nilai': [0.80, 0.81, 0.72, 0.74]
    }
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True, hide_index=True)
    st.caption("Performa dihitung berdasarkan *Macro Average* pada data uji.")