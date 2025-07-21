import streamlit as st
import pickle
import pandas as pd
import random # <<< BARU: Import library random

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
        .st-emotion-cache-1y4p8pa {
            padding-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)


# --- FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_models_and_vectorizer():
    with open('naive_bayes_model.pkl', 'rb') as model_file:
        model_nb = pickle.load(model_file)
    with open('knn_model.pkl', 'rb') as model_file:
        model_knn = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model_nb, model_knn, vectorizer

# --- FUNGSI load_review_data() DIHAPUS --- # <<< DIHAPUS

# --- MEMUAT MODEL ---
model_nb, model_knn, vectorizer = load_models_and_vectorizer()
# --- Baris untuk memuat data review dihapus ---

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    model_choice = st.sidebar.radio(
        "Pilih Algoritma Klasifikasi:",
        ('Naive Bayes', 'K-Nearest Neighbors (KNN)'),
        key='model_selection'
    )

# --- UTAMA APLIKASI ---
st.title("Analisis Sentimen Ulasan Aplikasi Steam 🕵️")
st.markdown("---")

# --- TATA LETAK KOLOM ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📝 Input Ulasan Anda")

    # --- DAFTAR KALIMAT CONTOH (HARDCODED) --- # <<< BARU
    contoh_positif_list = [
        "Aplikasinya keren dan sangat membantu, banyak diskon game!",
        "Terima kasih steam, akun saya yang dihack akhirnya bisa kembali.",
        "Aplikasi ini bagus untuk memantau akun dan membeli game saat ada diskon.",
        "Mantap, proses trade jadi sangat mudah dan aman lewat hp."
    ]
    contoh_negatif_list = [
        "Login susah banget, sering error dan lambat. Kecewa.",
        "Ribet banget, bikin akun saja gagal terus karena verifikasi email.",
        "Captcha-nya aneh, sudah diisi benar masih saja salah. Sulit login.",
        "Aplikasi ini boros kuota dan sering macet saat dibuka."
    ]

    # --- LOGIKA TOMBOL CONTOH DIUBAH --- # <<< DIUBAH
    if st.button('Coba Contoh Positif', use_container_width=True):
        st.session_state.user_input = random.choice(contoh_positif_list)
    if st.button('Coba Contoh Negatif', use_container_width=True):
        st.session_state.user_input = random.choice(contoh_negatif_list)

    # --- FORM INPUT ---
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

    else:
        st.info("Silakan masukkan ulasan dan klik tombol analisis untuk melihat hasilnya.")