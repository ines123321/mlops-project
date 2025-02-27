# Utilise une image de base Python
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /slimane-ines-4DS5-ml_project

# Installer les dépendances nécessaires
COPY requirements.txt .
RUN pip install -r requirements.txt

# Installer MLflow
RUN pip install mlflow

# Exposer le port utilisé par MLflow UI
EXPOSE 5000

# Lancer MLflow UI
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]

