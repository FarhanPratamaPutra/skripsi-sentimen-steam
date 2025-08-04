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

# --- CSS UNTUK MENGURANGI JARAK ---
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
    # Memuat model Naive Bayes yang sudah dilatih dengan SMOTE
    with open('naive_bayes_model.pkl', 'rb') as model_file:
        model_nb = pickle.load(model_file)
    # Memuat model Logistic Regression yang sudah dilatih dengan SMOTE
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model_lr = pickle.load(model_file)
    # Memuat TF-IDF Vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model_nb, model_lr, vectorizer

# --- MEMUAT MODEL ---
model_nb, model_lr, vectorizer = load_models_and_vectorizer()


# --- UTAMA APLIKASI ---
st.title("Analisis Sentimen Ulasan Aplikasi Steam 🕵️")
st.markdown("Aplikasi ini membandingkan performa model **Naive Bayes** dan **Logistic Regression** untuk analisis sentimen.")
st.markdown("---")

# --- PENGATURAN MODEL (DI HALAMAN UTAMA) ---
st.subheader("⚙️ Pengaturan Model")
model_choice = st.radio(
    "Pilih Algoritma Klasifikasi:",
    ('Naive Bayes', 'Logistic Regression'),
    key='model_selection',
    horizontal=True,
)
st.markdown("---")

# --- INPUT SECTION ---
st.subheader("📝 Input Ulasan Anda")

contoh_positif_list = [
    "Aplikasinya keren dan sangat membantu, banyak diskon game!",
    "Navigasi aplikasinya lancar dan responsif.",
    "Saya suka fitur wishlist, jadi tahu kapan ada diskon.",
    "Notifikasi diskonnya selalu tepat waktu, mantap",
    "Desain aplikasinya modern dan enak dilihat"
]
contoh_negatif_list = [
    "aplikasinya burik banget susah",
    "Market lambat banget, udah kayak pakai internet 2G",
    "Chat sering tidak sinkron dengan versi desktop sangat menganggu sekali",
    "Aplikasi sering force close tiba-tiba, aneh banget",
    "Fitur pencariannya error terus, tidak akurat"
]
contoh_netral_list = [
    "beli game dari hp, tidak perlu buka PC.",
    "Sangat memudahkan untuk top-up wallet Steam.",
    "Hanya digunakan untuk cek katalog game.",
    "Baru pertama kali menggunakan aplikasi ini",
    "Pengguna dapat mengelola pengaturan keluarga (Family view)",
    "Hanya ingin tahu isi dari aplikasinya"
]

col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Coba Contoh Positif', use_container_width=True):
        st.session_state.user_input = random.choice(contoh_positif_list)
with col2:
    if st.button('Coba Contoh Negatif', use_container_width=True):
        st.session_state.user_input = random.choice(contoh_negatif_list)
with col3:
    if st.button('Coba Contoh Netral', use_container_width=True):
        st.session_state.user_input = random.choice(contoh_netral_list)

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

    model = model_nb if model_choice == 'Naive Bayes' else model_lr
    
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    prediction_proba = model.predict_proba(input_vector)
    
    sentiment = prediction[0]
    
    # Mengambil probabilitas secara dinamis
    class_labels = model.classes_
    prob_dict = {label: prob for label, prob in zip(class_labels, prediction_proba[0])}

    prob_positif = prob_dict.get('positive', 0)
    prob_negatif = prob_dict.get('negative', 0)
    prob_netral = prob_dict.get('neutral', 0)
    
    if sentiment == 'positive':
        st.success(f"**Hasil Prediksi: Sentimen Positif** ✅")
    elif sentiment == 'negative':
        st.error(f"**Hasil Prediksi: Sentimen Negatif** ❌")
    else: # sentiment == 'netral'
        st.warning(f"**Hasil Prediksi: Sentimen Netral** 😐")
    
    st.metric(label=f"Tingkat Keyakinan ({model_choice})", value=f"{max(prob_positif, prob_negatif, prob_netral):.2%}")

    st.markdown("---") 
    if model_choice == 'Naive Bayes':
        st.info(
            """
            💡 **Penjelasan Keyakinan Naive Bayes:**
            Skor ini dihitung berdasarkan **probabilitas (peluang)**. Model menghitung seberapa besar kemungkinan ulasan Anda tergolong 'positif', 'negatif', atau 'netral' berdasarkan kemunculan kata-kata di dalamnya.
            """
        )
    else: 
        st.info(
            """
            💡 **Penjelasan Keyakinan Logistic Regression:**
            Skor ini dihasilkan dari **fungsi softmax** yang mengubah output menjadi probabilitas untuk setiap kelas (positif, negatif, netral). Model mempelajari bobot untuk setiap kata dan menghitung skor total untuk setiap kelas.
            """
        )

# --- TABS UNTUK INFORMASI TAMBAHAN ---
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📜 Tentang Aplikasi", "🧠 Cara Kerja", "📈 Performa Model"])

with tab1:
    st.subheader("Deskripsi Proyek")
    st.markdown("""
    Aplikasi ini merupakan bagian dari pengerjaan Skripsi di bidang Ilmu Komputer dengan fokus pada *Natural Language Processing* (NLP) dan *Machine Learning*.
    - **Tujuan**: Menganalisis dan membandingkan kinerja algoritma Naive Bayes dan Logistic Regression untuk klasifikasi sentimen pada ulasan aplikasi Steam.
    - **Metodologi**: CRISP-DM.
    - **Dibuat oleh**: Farhan Pratama Putra - 10121440 - Universitas Gunadarma
    """)

with tab2:
    st.subheader("Alur Kerja Proses Analisis")
    st.markdown("""
    1.  **Input Teks**: Teks ulasan yang Anda masukkan.
    2.  **TF-IDF Vectorization**: Teks diubah menjadi representasi numerik.
    3.  **Klasifikasi**: Vektor numerik dimasukkan ke model untuk memprediksi label sentimen **Positif**, **Negatif**, atau **Netral**.
    """)

with tab3:
    st.subheader("Tabel Perbandingan Kinerja Model")
    # --- PENTING: Ganti nilai performa di bawah ini ---
    # --- sesuai dengan hasil baru dari notebook Anda setelah SMOTE ---
    performance_data = {
        'Model': ['Naive Bayes', 'Logistic Regression'],
        'Accuracy': [0.71, 0.75], 
        'Precision': [0.69, 0.74],
        'Recall': [0.70, 0.76],
        'F1-Score': [0.70, 0.75]
    }
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True, hide_index=True)
    st.caption("Performa dihitung berdasarkan *Macro Average* pada data uji.")
    
    st.markdown("---") 
    st.subheader("Penjelasan Singkat Metrik")
    st.markdown(
        """
        - **Akurasi**: Persentase seberapa sering model menebak sentimen (baik positif, negatif, maupun netral) dengan benar dari keseluruhan data.
        - **Presisi**: Mengukur ketepatan prediksi. Dari semua yang ditebak sebagai 'positif', berapa persen yang benar? Begitu pula untuk kelas lainnya.
        - **Recall**s: Mengukur kelengkapan model. Dari semua yang seharusnya 'positif', berapa persen yang berhasil ditemukan model? Begitu pula untuk kelas lainnya.
        - **F1-Score**: Nilai gabungan yang menyeimbangkan antara Presisi dan Recall, memberikan satu angka performa yang komprehensif.
        """
    )
