import streamlit as st
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load model dan scaler
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_insight_rekomendasi(prediction, bmi, lifestyle_data):
    insight = []
    rekomendasi = []
    
    # Kategori BMI standar WHO dengan detail lebih spesifik
    bmi_categories = {
        'severely_underweight': (0, 16),
        'moderately_underweight': (16, 17),
        'mildly_underweight': (17, 18.5),
        'normal': (18.5, 25),
        'overweight': (25, 30),
        'obese_class_1': (30, 35),
        'obese_class_2': (35, 40),
        'obese_class_3': (40, float('inf'))
    }
    
    # Tentukan kategori BMI aktual
    bmi_category = ''
    for category, (lower, upper) in bmi_categories.items():
        if lower <= bmi < upper:
            bmi_category = category
            break
    
    # Tambahkan insight umum berdasarkan data pengguna
    gender_term = "Pria" if lifestyle_data['Gender'] == "Male" else "Wanita"
    age = int(lifestyle_data['Age'])
    age_group = ""
    
    if age < 18:
        age_group = "remaja"
    elif age < 30:
        age_group = "dewasa muda"
    elif age < 50:
        age_group = "dewasa"
    else:
        age_group = "lansia"
    
    # Insight dan rekomendasi berdasarkan kategori obesitas
    if prediction == "Insufficient Weight":
        insight.extend([
            f"Sebagai {gender_term} {age_group} dengan tinggi {lifestyle_data['Height']}m, berat badan Anda ({lifestyle_data['Weight']}kg) berada di bawah normal",
            f"BMI Anda adalah {bmi:.1f} (Kategori: {bmi_category.replace('_', ' ').title()}, Normal: 18.5-24.9)",
            "Kekurangan berat badan dapat mempengaruhi sistem kekebalan tubuh, kesehatan tulang, dan fungsi hormonal",
            f"Pada usia {lifestyle_data['Age']} tahun, penting untuk memiliki berat badan yang cukup untuk mendukung metabolisme dan kesehatan optimal"
        ])
        
        if lifestyle_data['family_history'] == 'yes':
            insight.append("Meskipun ada riwayat obesitas dalam keluarga, faktor genetik tidak selalu menentukan berat badan Anda")
        
        rekomendasi.extend([
            f"Tingkatkan asupan kalori harian dengan makanan bergizi sekitar 300-500 kalori lebih banyak per hari (target: {round(float(lifestyle_data['Weight'])*35)} kalori/hari)",
            f"Konsumsi protein berkualitas tinggi (target: {round(float(lifestyle_data['Weight'])*1.2, 1)}g per hari)",
            "Lakukan latihan kekuatan 2-3 kali seminggu untuk membangun massa otot dan meningkatkan kepadatan tulang",
            "Konsumsi makanan kaya nutrisi seperti kacang-kacangan, alpukat, dan minyak zaitun untuk kalori sehat"
        ])
    
    elif prediction == "Normal Weight":
        insight.extend([
            f"Sebagai {gender_term} {age_group} dengan tinggi {lifestyle_data['Height']}m, berat badan Anda ({lifestyle_data['Weight']}kg) berada dalam rentang ideal",
            f"BMI Anda adalah {bmi:.1f} (Kategori: Normal, Rentang ideal: 18.5-24.9)",
            "Berat badan ideal mengurangi risiko berbagai penyakit kronis seperti diabetes, hipertensi, dan penyakit jantung",
            f"Pada usia {lifestyle_data['Age']} tahun, mempertahankan berat badan ideal sangat penting untuk kesehatan jangka panjang"
        ])
        
        if lifestyle_data['family_history'] == 'yes':
            insight.append("Meskipun ada riwayat obesitas dalam keluarga, pola hidup sehat Anda berhasil menjaga berat badan ideal")
            rekomendasi.append("Tetap waspada dan pertahankan pola hidup sehat mengingat adanya faktor risiko genetik")
        
        rekomendasi.extend([
            f"Jaga pola makan seimbang dengan porsi yang tepat (target kalori: {round(float(lifestyle_data['Weight'])*30)} kalori/hari)",
            "Tetap aktif dengan olahraga rutin minimal 150 menit per minggu dengan intensitas sedang",
            f"Pertahankan konsumsi air yang cukup (minimal {round(float(lifestyle_data['Weight'])*0.033, 1)}L atau 8 gelas per hari)",
            "Lakukan pemeriksaan kesehatan rutin untuk memantau indikator kesehatan lainnya"
        ])
    
    elif "Overweight" in prediction or "Obesity" in prediction:
        severity = ""
        risk_level = ""
        calorie_reduction = 0
        exercise_minutes = 0
        
        if "Overweight Level I" in prediction:
            severity = "ringan"
            risk_level = "meningkat"
            calorie_reduction = 300
            exercise_minutes = 150
        elif "Overweight Level II" in prediction:
            severity = "sedang"
            risk_level = "tinggi"
            calorie_reduction = 400
            exercise_minutes = 200
        elif "Obesity Type I" in prediction:
            severity = "tinggi"
            risk_level = "sangat tinggi"
            calorie_reduction = 500
            exercise_minutes = 250
        elif "Obesity Type II" in prediction:
            severity = "sangat tinggi"
            risk_level = "ekstrem"
            calorie_reduction = 600
            exercise_minutes = 300
        elif "Obesity Type III" in prediction:
            severity = "ekstrem"
            risk_level = "sangat ekstrem"
            calorie_reduction = 750
            exercise_minutes = 300
        
        insight.extend([
            f"Sebagai {gender_term} {age_group} dengan tinggi {lifestyle_data['Height']}m, berat badan Anda ({lifestyle_data['Weight']}kg) berada di atas normal",
            f"BMI Anda adalah {bmi:.1f} (Kategori: {bmi_category.replace('_', ' ').title()}, Normal: 18.5-24.9)",
            f"Anda memiliki kelebihan berat badan tingkat {severity} yang perlu ditangani",
            f"Risiko penyakit kardiovaskular, diabetes, dan masalah kesehatan lainnya {risk_level}"
        ])
        
        if lifestyle_data['family_history'] == 'yes':
            insight.append("Riwayat obesitas dalam keluarga meningkatkan risiko Anda, namun pola hidup tetap menjadi faktor penentu utama")
        
        # Target berat ideal
        ideal_weight_low = round((18.5 * float(lifestyle_data['Height']) * float(lifestyle_data['Height'])), 1)
        ideal_weight_high = round((24.9 * float(lifestyle_data['Height']) * float(lifestyle_data['Height'])), 1)
        weight_to_lose = round(float(lifestyle_data['Weight']) - ideal_weight_high, 1)
        
        if weight_to_lose > 0:
            insight.append(f"Untuk mencapai berat badan ideal, Anda perlu menurunkan sekitar {weight_to_lose}kg (target: {ideal_weight_low}-{ideal_weight_high}kg)")
        
        # Rekomendasi berdasarkan tingkat keparahan
        if severity in ["ringan", "sedang"]:
            rekomendasi.extend([
                f"Kurangi asupan kalori sekitar {calorie_reduction} kalori per hari (target: {round(float(lifestyle_data['Weight'])*22)} kalori/hari)",
                f"Tingkatkan aktivitas fisik menjadi {exercise_minutes} menit per minggu dengan kombinasi kardio dan latihan kekuatan",
                "Batasi konsumsi makanan olahan, tinggi gula, dan karbohidrat sederhana",
                "Prioritaskan protein tanpa lemak, sayuran, dan lemak sehat dalam pola makan Anda"
            ])
        else:
            rekomendasi.extend([
                f"Kurangi asupan kalori sekitar {calorie_reduction} kalori per hari (target: {round(float(lifestyle_data['Weight'])*20)} kalori/hari)",
                f"Tingkatkan aktivitas fisik secara bertahap hingga {exercise_minutes} menit per minggu dengan intensitas yang sesuai kondisi Anda",
                "Konsultasikan dengan ahli gizi untuk program penurunan berat badan yang aman dan efektif",
                "Pertimbangkan untuk berkonsultasi dengan dokter untuk evaluasi kesehatan menyeluruh dan penanganan risiko penyakit terkait",
                "Tetapkan target penurunan berat badan yang realistis (0.5-1kg per minggu) untuk hasil yang berkelanjutan"
            ])
    
    # Insight dan rekomendasi berdasarkan gaya hidup
    if float(lifestyle_data['FCVC']) < 2:
        insight.append("Konsumsi sayur dan buah Anda masih kurang dari rekomendasi harian (skor: {:.1f}/3)".format(float(lifestyle_data['FCVC'])))
        rekomendasi.append("Tingkatkan konsumsi sayur dan buah menjadi minimal 5 porsi per hari (3 porsi sayur, 2 porsi buah)")
    
    if float(lifestyle_data['CH2O']) < 2:
        insight.append("Konsumsi air Anda ({:.1f}L/hari) kurang dari kebutuhan optimal".format(float(lifestyle_data['CH2O'])))
        rekomendasi.append(f"Tingkatkan konsumsi air putih menjadi minimal {round(float(lifestyle_data['Weight'])*0.033, 1)}L per hari (sekitar {round(float(lifestyle_data['Weight'])*0.033/0.25)} gelas)")
    
    if float(lifestyle_data['FAF']) < 2:
        insight.append("Tingkat aktivitas fisik Anda rendah ({} hari/minggu)".format(int(float(lifestyle_data['FAF']))))
        rekomendasi.append("Tingkatkan frekuensi aktivitas fisik menjadi minimal 3-5 hari per minggu dengan durasi 30-60 menit per sesi")
    elif float(lifestyle_data['FAF']) >= 2 and float(lifestyle_data['FAF']) < 4:
        insight.append("Tingkat aktivitas fisik Anda cukup baik ({} hari/minggu), namun masih bisa ditingkatkan".format(int(float(lifestyle_data['FAF']))))
    elif float(lifestyle_data['FAF']) >= 4:
        insight.append("Tingkat aktivitas fisik Anda sangat baik ({} hari/minggu), pertahankan kebiasaan ini".format(int(float(lifestyle_data['FAF']))))
    
    if lifestyle_data['SMOKE'] == 'yes':
        insight.append("Merokok dapat mempengaruhi metabolisme tubuh, meningkatkan risiko penyakit, dan mengurangi kapasitas paru-paru untuk aktivitas fisik")
        rekomendasi.append("Pertimbangkan untuk berhenti merokok dan cari dukungan profesional jika diperlukan (konsultasi dengan dokter atau program berhenti merokok)")
    
    if float(lifestyle_data['TUE']) > 2:
        insight.append(f"Penggunaan perangkat elektronik selama {lifestyle_data['TUE']} jam per hari dapat mengurangi aktivitas fisik dan meningkatkan perilaku sedentari")
        rekomendasi.append("Kurangi waktu penggunaan teknologi dan ganti dengan aktivitas fisik atau hobi aktif (target: maksimal 2 jam screen time di luar pekerjaan)")
    
    if lifestyle_data['FAVC'] == 'yes':
        insight.append("Konsumsi makanan tinggi kalori secara berlebihan berkontribusi signifikan pada penambahan berat badan dan risiko penyakit metabolik")
        rekomendasi.append("Batasi makanan tinggi kalori dan ganti dengan alternatif yang lebih sehat (buah sebagai camilan, hindari makanan cepat saji)")
    
    if lifestyle_data['CALC'] == 'Always':
        insight.append("Konsumsi alkohol berlebihan dapat menambah kalori kosong, mengganggu metabolisme, dan merusak fungsi hati")
        rekomendasi.append("Kurangi konsumsi alkohol dan batasi pada jumlah moderat (maksimal 1 gelas/hari untuk wanita, 2 gelas/hari untuk pria)")
    
    if int(lifestyle_data['NCP']) > 3:
        insight.append(f"Frekuensi makan {lifestyle_data['NCP']} kali sehari dapat menyebabkan konsumsi kalori berlebih jika porsi tidak dikontrol")
        rekomendasi.append("Perhatikan ukuran porsi makan dan pastikan setiap makanan mengandung nutrisi seimbang")
    elif int(lifestyle_data['NCP']) < 3:
        insight.append(f"Frekuensi makan hanya {lifestyle_data['NCP']} kali sehari dapat menyebabkan makan berlebihan saat lapar")
        rekomendasi.append("Pertimbangkan untuk makan dalam porsi lebih kecil namun lebih sering (3 kali makan utama dan 2 kali camilan sehat)")
    
    if lifestyle_data['SCC'] == 'no' and ("Overweight" in prediction or "Obesity" in prediction):
        insight.append("Tidak memantau asupan kalori dapat menyulitkan kontrol berat badan")
        rekomendasi.append("Mulai catat asupan makanan dan kalori harian menggunakan aplikasi atau jurnal makanan")
    
    if lifestyle_data['MTRANS'] in ['Automobile', 'Public_Transportation'] and float(lifestyle_data['FAF']) < 3:
        insight.append(f"Penggunaan {lifestyle_data['MTRANS']} sebagai transportasi utama mengurangi aktivitas fisik harian Anda")
        rekomendasi.append("Pertimbangkan untuk berjalan kaki atau bersepeda untuk jarak dekat, atau turun beberapa halte lebih awal dan jalan kaki")
    
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
            st.info(f"BMI: {bmi:.1f}")

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