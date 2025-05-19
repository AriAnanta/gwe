import streamlit as st
import pickle
import numpy as np
import warnings

# Abaikan peringatan versi scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

# Load model dan scaler
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_insight_rekomendasi(prediction, bmi, lifestyle_data):
    insight = []
    rekomendasi = []
    
    # Gunakan nilai BMI yang diberikan untuk insight dan rekomendasi
    # Kategori BMI standar WHO
    bmi_categories = {
        'underweight': (0, 18.5),
        'normal': (18.5, 25),
        'overweight': (25, 30),
        'obese': (30, float('inf'))
    }
    
    # Tentukan kategori BMI aktual
    bmi_category = ''
    for category, (lower, upper) in bmi_categories.items():
        if lower <= bmi < upper:
            bmi_category = category
            break
    
    # Insight dan rekomendasi berdasarkan kategori obesitas
    if prediction == "Insufficient Weight":
        insight.extend([
            "Berat badan Anda berada di bawah normal",
            f"BMI Anda adalah {bmi:.1f} (Normal: 18.5-24.9)",
            "Kekurangan berat badan dapat mempengaruhi sistem kekebalan tubuh dan kesehatan tulang",
            f"Pada usia {lifestyle_data['Age']} tahun, penting untuk memiliki berat badan yang cukup"
        ])
        rekomendasi.extend([
            "Tingkatkan asupan kalori harian dengan makanan bergizi sekitar 300-500 kalori lebih banyak per hari",
            "Konsumsi protein berkualitas tinggi (0.8-1g per kg berat badan)",
            "Lakukan latihan kekuatan 2-3 kali seminggu untuk membangun massa otot"
        ])
    
    elif prediction == "Normal Weight":
        insight.extend([
            "Berat badan Anda ideal",
            f"BMI Anda adalah {bmi:.1f} (Normal: 18.5-24.9)",
            "Pertahankan pola hidup sehat Anda",
            f"Pada usia {lifestyle_data['Age']} tahun, berat badan ideal membantu mencegah berbagai penyakit kronis"
        ])
        rekomendasi.extend([
            "Jaga pola makan seimbang dengan porsi yang tepat",
            "Tetap aktif dengan olahraga rutin minimal 150 menit per minggu",
            "Pertahankan konsumsi air yang cukup (minimal 8 gelas per hari)"
        ])
    
    elif "Overweight" in prediction or "Obesity" in prediction:
        # Tentukan tingkat kelebihan berat badan
        severity = ""
        risk_level = ""
        if "Overweight Level I" in prediction:
            severity = "ringan"
            risk_level = "meningkat"
        elif "Overweight Level II" in prediction:
            severity = "sedang"
            risk_level = "tinggi"
        elif "Obesity Type I" in prediction:
            severity = "tinggi"
            risk_level = "sangat tinggi"
        elif "Obesity Type II" in prediction or "Obesity Type III" in prediction:
            severity = "sangat tinggi"
            risk_level = "ekstrem"
        
        insight.extend([
            f"BMI Anda adalah {bmi:.1f} (Normal: 18.5-24.9)",
            f"Anda memiliki kelebihan berat badan tingkat {severity}",
            f"Risiko penyakit kardiovaskular, diabetes, dan masalah kesehatan lainnya {risk_level}",
            "Pola hidup dan pola makan sangat mempengaruhi berat badan Anda"
        ])
        
        # Rekomendasi berdasarkan tingkat keparahan
        if severity in ["ringan", "sedang"]:
            rekomendasi.extend([
                "Kurangi asupan kalori sekitar 300-500 kalori per hari",
                "Tingkatkan aktivitas fisik menjadi 150-300 menit per minggu",
                "Batasi konsumsi makanan olahan dan tinggi gula"
            ])
        else:
            rekomendasi.extend([
                "Kurangi asupan kalori sekitar 500-750 kalori per hari",
                "Tingkatkan aktivitas fisik secara bertahap hingga 300 menit per minggu",
                "Konsultasikan dengan ahli gizi untuk program penurunan berat badan yang aman",
                "Pertimbangkan untuk berkonsultasi dengan dokter untuk evaluasi kesehatan menyeluruh"
            ])
    
    # Insight dan rekomendasi berdasarkan gaya hidup
    if float(lifestyle_data['FCVC']) < 2:
        insight.append("Konsumsi sayur dan buah Anda masih kurang dari rekomendasi harian")
        rekomendasi.append("Tingkatkan konsumsi sayur dan buah menjadi 5 porsi per hari")
    
    if float(lifestyle_data['CH2O']) < 2:
        insight.append("Konsumsi air Anda kurang dari kebutuhan optimal")
        rekomendasi.append(f"Tingkatkan konsumsi air putih menjadi minimal {round(float(lifestyle_data['Weight'])*0.033, 1)}L per hari")
    
    if float(lifestyle_data['FAF']) < 2:
        insight.append("Tingkat aktivitas fisik Anda rendah")
        rekomendasi.append("Tingkatkan frekuensi aktivitas fisik menjadi minimal 3 hari per minggu")
    
    if lifestyle_data['SMOKE'] == 'yes':
        insight.append("Merokok dapat mempengaruhi metabolisme tubuh dan meningkatkan risiko penyakit")
        rekomendasi.append("Pertimbangkan untuk berhenti merokok dan cari dukungan profesional jika diperlukan")
    
    if float(lifestyle_data['TUE']) > 2:
        insight.append(f"Penggunaan perangkat elektronik selama {lifestyle_data['TUE']} jam per hari dapat mengurangi aktivitas fisik")
        rekomendasi.append("Kurangi waktu penggunaan teknologi dan ganti dengan aktivitas fisik atau hobi aktif")
    
    if lifestyle_data['FAVC'] == 'yes':
        insight.append("Konsumsi makanan tinggi kalori secara berlebihan berkontribusi pada penambahan berat badan")
        rekomendasi.append("Batasi makanan tinggi kalori dan ganti dengan alternatif yang lebih sehat")
    
    if lifestyle_data['CALC'] == 'Always':
        insight.append("Konsumsi alkohol berlebihan dapat menambah kalori kosong dan mengganggu metabolisme")
        rekomendasi.append("Kurangi konsumsi alkohol dan batasi pada jumlah moderat (maksimal 1-2 gelas per hari)")
    
    return insight, rekomendasi

def main():
    st.title('Prediksi Tingkat Obesitas')
    
    # Form input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'], format_func=lambda x: 'Laki-laki' if x == 'Male' else 'Perempuan')
            age = st.number_input('Usia', min_value=0, step=1)
            height = st.number_input('Tinggi Badan (meter)', min_value=0.0, step=0.01, format="%.2f")
            weight = st.number_input('Berat Badan (kg)', min_value=0.0, step=0.1)
            family_history = st.selectbox('Apakah ada anggota keluarga yang mengalami obesitas?', ['yes', 'no'], format_func=lambda x: 'Ya' if x == 'yes' else 'Tidak')
            favc = st.selectbox('Apakah Anda sering mengonsumsi makanan tinggi kalori?', ['yes', 'no'], format_func=lambda x: 'Ya' if x == 'yes' else 'Tidak')
            fcvc = st.number_input('Seberapa sering Anda mengonsumsi sayuran? (1-3)', min_value=1, max_value=3, step=1)
            ncp = st.number_input('Berapa kali makan dalam sehari?', min_value=1, max_value=4, step=1)
        
        with col2:
            caec = st.selectbox('Apakah Anda makan lagi di antara waktu makan anda?', 
                              ['Sometimes', 'Always', 'Never'],
                              format_func=lambda x: 'Kadang-kadang' if x == 'Sometimes' else ('Selalu' if x == 'Always' else 'Tidak Pernah'))
            smoke = st.selectbox('Apakah Anda merokok?', ['yes', 'no'], format_func=lambda x: 'Ya' if x == 'yes' else 'Tidak')
            ch2o = st.number_input('Berapa liter air yang Anda minum per hari?', min_value=1.0, max_value=3.0, step=0.1)
            scc = st.selectbox('Apakah Anda memantau kalori harian?', ['yes', 'no'], format_func=lambda x: 'Ya' if x == 'yes' else 'Tidak')
            faf = st.number_input('Berapa hari dalam seminggu Anda berolahraga?', min_value=0, max_value=7, step=1)
            tue = st.number_input('Berapa jam per hari Anda menggunakan perangkat elektronik?', min_value=0, max_value=24, step=1)
            calc = st.selectbox('Seberapa sering Anda mengonsumsi alkohol?',
                              ['Sometimes', 'Always', 'Never'],
                              format_func=lambda x: 'Kadang-kadang' if x == 'Sometimes' else ('Selalu' if x == 'Always' else 'Tidak Pernah'))
            mtrans = st.selectbox('Transportasi yang biasa Anda gunakan?',
                                ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'],
                                format_func=lambda x: {
                                    'Public_Transportation': 'Transportasi Umum',
                                    'Walking': 'Jalan Kaki',
                                    'Automobile': 'Mobil',
                                    'Motorbike': 'Motor',
                                    'Bike': 'Sepeda'
                                }[x])

        submitted = st.form_submit_button("Prediksi Obesitas")

    if submitted:
        try:
            # Kumpulkan data
            data = {
                'Gender': gender,
                'Age': age,
                'Height': height,
                'Weight': weight,
                'family_history': family_history,
                'FAVC': favc,
                'FCVC': fcvc,
                'NCP': ncp,
                'CAEC': caec,
                'SMOKE': smoke,
                'CH2O': ch2o,
                'SCC': scc,
                'FAF': faf,
                'TUE': tue,
                'CALC': calc,
                'MTRANS': mtrans
            }

            # Ekstrak fitur
            features = [
                float(data['Gender'] == 'Male'),
                float(data['Age']),
                float(data['Height']),
                float(data['Weight']),
                float(data['family_history'] == 'yes'),
                float(data['FAVC'] == 'yes'),
                float(data['FCVC']),
                float(data['NCP']),
                float(data['CAEC'] == 'Always'),
                float(data['SMOKE'] == 'yes'),
                float(data['CH2O']),
                float(data['SCC'] == 'yes'),
                float(data['FAF']),
                float(data['TUE']),
                float(data['CALC'] == 'Always'),
                float(data['MTRANS'] == 'Public_Transportation')
            ]

            # Definisikan nama fitur untuk mengatasi peringatan
            feature_names = [
                'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC',
                'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
            ]
            
            # Reshape dan transformasi fitur
            features = np.array(features).reshape(1, -1)
            # Gunakan parameter feature_names_in untuk mengatasi peringatan
            features_scaled = scaler.transform(features)

            # Prediksi
            prediction = model.predict(features_scaled)

            # Mapping hasil prediksi
            obesity_levels = {
                0: "Insufficient Weight",
                1: "Normal Weight", 
                2: "Overweight Level I",
                3: "Overweight Level II",
                4: "Obesity Type I",
                5: "Obesity Type II",
                6: "Obesity Type III"
            }

            prediksi = obesity_levels[prediction[0]]
            bmi = float(data['Weight']) / (float(data['Height']) * float(data['Height']))

            # Dapatkan insight dan rekomendasi dengan BMI yang sudah dihitung
            insight, rekomendasi = get_insight_rekomendasi(prediksi, bmi, data)

            # Tampilkan hasil
            st.success(f"Prediksi Kategori: {prediksi}")
            st.info(f"BMI: {bmi:.4f}")

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Insight")
                for ins in insight:
                    st.write("•", ins)

            with col2:
                st.subheader("Rekomendasi")
                for rek in rekomendasi:
                    st.write("•", rek)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")

if __name__ == '__main__':
    main()