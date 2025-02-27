FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install scikit-learn==1.3.2  # Install the correct version of scikit-learn

# Copy the model file
COPY model.joblib /app/model.joblib

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
