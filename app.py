import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    layout="wide"
)

# --- KONFIGURASI HALAMAN --- # <-- BARU
st.set_page_config(
    page_title="Analisis Sentimen Steam",
    page_icon="🕵️",
    layout="wide"
)

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

# --- MEMUAT MODEL ---
model_nb, model_knn, vectorizer = load_models_and_vectorizer()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Pilih Model Klasifikasi")
    model_choice = st.radio(
        "Algoritma:",
        ('Naive Bayes', 'K-Nearest Neighbors (KNN)'),
        key='model_selection'
    )
    
    # --- MENAMPILKAN PERFORMA MODEL DI SIDEBAR --- # <-- BARU
    st.markdown("---")
    st.header("Performa Model")
    
    # Data performa dari notebook Anda
    performance_data = {
        'Model': ['Naive Bayes', 'KNN'],
        'Akurasi': [0.80, 0.65],
        'Presisi': [0.81, 0.64],
        'Recall': [0.72, 0.65],
        'F1-Score': [0.74, 0.63]
    }
    df_performance = pd.DataFrame(performance_data)
    
    st.dataframe(df_performance, hide_index=True)
    st.info("Performa dihitung berdasarkan Macro Avg pada data uji.")
    

# --- UTAMA APLIKASI ---
st.title("Analisis Sentimen Ulasan Aplikasi Steam 🕵️")
st.markdown("Aplikasi ini membandingkan performa model **Naive Bayes** dan **K-Nearest Neighbors (KNN)** untuk analisis sentimen ulasan.")
st.markdown("---")


# --- TATA LETAK KOLOM --- # <-- BARU
col1, col2 = st.columns([2, 2])

with col1:
    st.subheader("📝 Input Ulasan")

    # --- CONTOH ULASAN --- # <-- BARU
    st.write("Coba dengan contoh ulasan di bawah ini:")
    example_buttons = st.container()
    if example_buttons.button('Contoh Positif', key='positif_example'):
        st.session_state.user_input = "Aplikasinya keren dan sangat membantu, banyak diskon game!"
    if example_buttons.button('Contoh Negatif', key='negatif_example'):
        st.session_state.user_input = "Login susah banget, sering error dan lambat. Kecewa."

    # --- FORM INPUT ---
    with st.form(key='sentiment_form'):
        user_input = st.text_area(
            "Masukkan ulasan Anda di sini:",
            key='user_input'
        )
        submit_button = st.form_submit_button(label='Analisis Sentimen')

with col2:
    st.subheader("📊 Hasil Analisis")

    # --- LOGIKA PREDIKSI & TAMPILAN HASIL ---
    if submit_button and user_input:
        # Pilih model berdasarkan pilihan di sidebar
        model = model_nb if model_choice == 'Naive Bayes' else model_knn
        
        # Prediksi
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        prediction_proba = model.predict_proba(input_vector)
        
        sentiment = prediction[0]
        prob_negatif = prediction_proba[0][0]
        prob_positif = prediction_proba[0][1]
        
        # --- TAMPILAN HASIL BARU --- # <-- BARU
        if sentiment == 'positif':
            st.success(f"**Hasil Prediksi: Sentimen Positif** ✅")
        else:
            st.error(f"**Hasil Prediksi: Sentimen Negatif** ❌")
        
        # Menampilkan probabilitas dengan st.metric
        st.metric(label="Tingkat Keyakinan", value=f"{max(prob_positif, prob_negatif):.2%}")

        # Menampilkan probabilitas dalam bentuk bar chart
        st.write("Probabilitas:")
        prob_df = pd.DataFrame({
            'Sentimen': ['Positif', 'Negatif'],
            'Probabilitas': [prob_positif, prob_negatif]
        })
        st.bar_chart(prob_df.set_index('Sentimen'), height=250)

    else:
        st.info("Silakan masukkan ulasan dan klik tombol analisis untuk melihat hasilnya.")