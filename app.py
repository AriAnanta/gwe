from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan scaler
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_insight_rekomendasi(prediction, bmi, lifestyle_data):
    insight = []
    rekomendasi = []
    
    # Hitung BMI 
    height_m = float(lifestyle_data['Height'])
    weight_kg = float(lifestyle_data['Weight'])
    bmi = weight_kg / (height_m * height_m)
    
    # Insight dan rekomendasi berdasarkan kategori obesitas
    if prediction == "Insufficient Weight":
        insight.extend([
            "Berat badan Anda berada di bawah normal",
            f"BMI Anda adalah {bmi:.1f} (Normal: 18.5-24.9)",
            "Kekurangan berat badan dapat mempengaruhi sistem kekebalan tubuh"
        ])
        rekomendasi.extend([
            "Tingkatkan asupan kalori harian dengan makanan bergizi",
            "Konsumsi protein berkualitas tinggi",
            "Lakukan latihan kekuatan untuk membangun massa otot"
        ])
    
    elif prediction == "Normal Weight":
        insight.extend([
            "Berat badan anda ideal",
            f"BMI Anda adalah {bmi:.1f} (Normal: 18.5-24.9)",
            "Pertahankan pola hidup sehat Anda"
        ])
        rekomendasi.extend([
            "Jaga pola makan seimbang",
            "Tetap aktif dengan olahraga rutin",
            "Pertahankan konsumsi air yang cukup"
        ])
    
    elif "Overweight" in prediction or "Obesity" in prediction:
        insight.extend([
            f"BMI Anda adalah {bmi:.1f} (Normal: 18.5-24.9)",
            "Kelebihan berat badan dapat meningkatkan risiko penyakit kardiovaskular",
            "Pola hidup dan pola makan mempengaruhi berat badan Anda"
        ])
        rekomendasi.extend([
            "Kurangi konsumsi makanan tinggi kalori dan lemak",
            "Tingkatkan aktivitas fisik minimal 30 menit per hari",
            "Konsultasikan dengan ahli gizi untuk program penurunan berat badan"
        ])
    
    # insight berdasarkan gaya hidup
    if float(lifestyle_data['FCVC']) < 2:
        rekomendasi.append("Tingkatkan konsumsi sayur dan buah")
    
    if float(lifestyle_data['CH2O']) < 2:
        rekomendasi.append("Tingkatkan konsumsi air putih (minimal 2L per hari)")
    
    if float(lifestyle_data['FAF']) < 2:
        rekomendasi.append("Tingkatkan frekuensi aktivitas fisik")
    
    if lifestyle_data['SMOKE'] == 'yes':
        insight.append("Merokok dapat mempengaruhi metabolisme tubuh")
        rekomendasi.append("Pertimbangkan untuk berhenti merokok")
    
    if float(lifestyle_data['TUE']) > 1:
        rekomendasi.append("Kurangi waktu penggunaan teknologi dan tingkatkan aktivitas fisik")
    
    return insight, rekomendasi

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            # Handle API request
            data = request.get_json()
        else:
            # Handle form submission
            data = {
                'Gender': request.form.get('Gender'),
                'Age': request.form.get('Age'),
                'Height': request.form.get('Height'),
                'Weight': request.form.get('Weight'),
                'family_history': request.form.get('family_history'),
                'FAVC': request.form.get('FAVC'),
                'FCVC': request.form.get('FCVC'),
                'NCP': request.form.get('NCP'),
                'CAEC': request.form.get('CAEC'),
                'SMOKE': request.form.get('SMOKE'),
                'CH2O': request.form.get('CH2O'),
                'SCC': request.form.get('SCC'),
                'FAF': request.form.get('FAF'),
                'TUE': request.form.get('TUE'),
                'CALC': request.form.get('CALC'),
                'MTRANS': request.form.get('MTRANS')
            }
        
        # Ekstrak fitur dari data
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
        
        # Reshape dan transformasi fitur
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Prediksi
        prediction = model.predict(features_scaled)
        
        # Mapping hasil prediksi ke label
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
        
        # Hitung BMI
        bmi = float(data['Weight']) / (float(data['Height']) * float(data['Height']))
        
        # Dapatkan insight dan rekomendasi
        insight, rekomendasi = get_insight_rekomendasi(prediksi, bmi, data)
        
        result = {
            "prediksi": prediksi,
            "bmi": round(bmi, 1),
            "insight": insight,
            "rekomendasi": rekomendasi
        }
        
        if request.is_json:
            return jsonify(result)
        else:
            return render_template('result.html', 
                                prediksi=result["prediksi"],
                                bmi=result["bmi"],
                                insights=result["insight"],
                                rekomendasis=result["rekomendasi"])
    
    except Exception as e:
        error_message = str(e)
        if request.is_json:
            return jsonify({"error": error_message}), 400
        else:
            return render_template('error.html', error=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 