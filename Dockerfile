# Gunakan image Python base pake yg slim biar ringan
FROM python:3.11-slim

# Set working directory untuk containernya
WORKDIR /app

# Copy file requirements dan install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy hanya file yang diperlukan
COPY ObesityDetection.py .
COPY logistic_model.pkl .
COPY scaler.pkl .

# Expose port untuk Streamlit (default 8501)
EXPOSE 8501

# Command untuk menjalankan aplikasi Streamlit
CMD ["streamlit", "run", "ObesityDetection.py", "--server.port=8501", "--server.address=0.0.0.0"] 