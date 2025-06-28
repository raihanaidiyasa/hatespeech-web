import streamlit as st
import pickle

# Menggunakan cache untuk memuat model dan vectorizer agar lebih cepat
# dan tidak perlu di-load ulang setiap kali ada interaksi dari user.
# @st.cache_data untuk versi Streamlit yang lebih lama
@st.cache_resource
def load_model_and_vectorizer():
    """
    Fungsi untuk memuat model dan vectorizer dari file pickle.
    """
    try:
        with open('model_klasifikasi.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Pastikan file 'model_klasifikasi.pkl' dan 'vectorizer.pkl' ada di folder yang sama.")
        return None, None

# Muat model dan vectorizer saat aplikasi pertama kali dijalankan
model, vectorizer = load_model_and_vectorizer()

# --- Tampilan Utama Aplikasi Streamlit ---

st.set_page_config(page_title="Klasifikasi Ujaran Kebencian", page_icon="⚖️", layout="centered")

st.title("⚖️ Klasifikasi Ujaran Kebencian")
st.write(
    "Selamat datang di aplikasi klasifikasi komentar. "
    "Masukkan sebuah komentar di bawah ini untuk menganalisis apakah komentar tersebut mengandung ujaran kebencian positif atau negatif."
)

# Buat area teks untuk input dari user
comment_text = st.text_area("Masukkan Komentar:", "", height=150, placeholder="Tulis komentar Anda di sini...")

# Buat tombol untuk memulai klasifikasi
if st.button("Klasifikasi Sekarang", type="primary"):
    if model is not None and vectorizer is not None:
        if comment_text:
            # Tampilkan spinner saat model sedang bekerja
            with st.spinner('Menganalisis komentar...'):
                # 1. Ubah teks input menggunakan vectorizer
                vectorized_text = vectorizer.transform([comment_text])
                
                # 2. Lakukan prediksi dengan model
                prediction = model.predict(vectorized_text)
                
                # 3. Ambil hasil prediksi
                result = prediction[0]
                
                # 4. Tentukan label dan tampilkan hasil
                # **PENTING**: Sesuaikan label ini dengan mapping di notebook Anda (misal: 1=Positif, 0=Negatif)
                if result == 1:
                    hasil_klasifikasi = 'Positif Ujaran Kebencian'
                    st.error(f'**Hasil Analisis:** {hasil_klasifikasi}')
                else:
                    hasil_klasifikasi = 'Negatif Ujaran Kebencian'
                    st.success(f'**Hasil Analisis:** {hasil_klasifikasi}')
            
            # Tampilkan juga komentar aslinya
            st.write("**Komentar yang Dianalisis:**")
            st.info(f'"{comment_text}"')
            
        else:
            # Tampilkan peringatan jika user tidak memasukkan apa-apa
            st.warning("Mohon masukkan sebuah komentar terlebih dahulu.")