import streamlit as st
import pickle
import pandas as pd

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

# --- FUNGSI BARU UNTUK MEMUAT DATA ULASAN --- # <<< BARU
@st.cache_data
def load_review_data():
    df = pd.read_csv('labeled_data.csv')
    # Pastikan tidak ada baris kosong yang menyebabkan error
    df.dropna(subset=['stemming', 'label'], inplace=True)
    positive_reviews = df[df['label'] == 'positif']
    negative_reviews = df[df['label'] == 'negatif']
    return positive_reviews, negative_reviews

# --- MEMUAT MODEL DAN DATA ---
model_nb, model_knn, vectorizer = load_models_and_vectorizer()
positive_reviews, negative_reviews = load_review_data() # <<< BARU

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

    # --- LOGIKA TOMBOL CONTOH DIUBAH --- # <<< DIUBAH
    if st.button('Coba Contoh Positif', use_container_width=True):
        # Ambil 1 sampel acak dari data positif
        st.session_state.user_input = positive_reviews['stemming'].sample(1).iloc[0]
    if st.button('Coba Contoh Negatif', use_container_width=True):
        # Ambil 1 sampel acak dari data negatif
        st.session_state.user_input = negative_reviews['stemming'].sample(1).iloc[0]

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

# --- TABS UNTUK INFORMASI TAMBAHAN ---
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📜 Tentang Aplikasi", "🧠 Cara Kerja", "📈 Performa Model"])

with tab1:
    st.subheader("Deskripsi Proyek")
    st.markdown("""
    Aplikasi ini merupakan bagian dari pengerjaan Skripsi di bidang Ilmu Komputer dengan fokus pada *Natural Language Processing* (NLP) dan *Machine Learning*.
    - **Tujuan**: Menganalisis dan membandingkan kinerja algoritma Naive Bayes dan K-Nearest Neighbors untuk klasifikasi sentimen pada ulasan aplikasi Steam.
    - **Metodologi**: CRISP-DM.
    - **Dibuat oleh**: [Nama Anda] - [NIM Anda] - [Universitas Anda]
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
        'Akurasi': [0.80, 0.65],
        'Presisi': [0.81, 0.64],
        'Recall': [0.72, 0.65],
        'F1-Score': [0.74, 0.63]
    }
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True, hide_index=True)
    st.caption("Performa dihitung berdasarkan *Macro Average* pada data uji.")