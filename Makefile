# Nom des fichiers et répertoires
DATA_FILE = merged_churn.csv
PROCESSED_DATA_FILE = processed_data.csv
PCA_OUTPUT = pca_output.pkl
MODEL_FILE = random_forest_model.pkl
EVALUATION_REPORT = evaluation_report.txt

# Commandes pour installer les dépendances (optionnel)
install:
	pip install -r requirements.txt

# Étape 1 : Charger les données
load_data:
	python3 main.py --load

# Étape 2 : Préparer les données (standardisation)
prepare_data: load_data
	python3 main.py --prepare

# Étape 3 : Appliquer PCA
pca: prepare_data
	python3 main.py --pca

# Étape 4 : KMeans Clustering
kmeans: pca
	python3 main.py --kmeans

# Étape 5 : Clustering Hiérarchique
hierarchical: pca
	python3 main.py --hierarchical

# Étape 6 : Entraîner le modèle Random Forest
train: prepare_data pca kmeans hierarchical
	python3 main.py --train

# Étape 7 : Évaluer le modèle
evaluate: train
	python3 main.py --evaluate

# Étape 8 : Pipeline complet
all: train evaluate

# Start the FastAPI server
start-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000
#  Test the API
test-api:
	curl -X 'POST' \
	'http://127.0.0.1:8000/predict/' \
	-H 'Content-Type: application/json' \
	-d '{
	"features": [0, 132, 410, 1, 0, 30, 280.5, 115, 48.20, 202.3, 101, 18.25, 250.3, 95, 12.50, 9.5]
	}'
run_mlflow:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &
# Nettoyer les fichiers générés
clean:
	rm -f $(PROCESSED_DATA_FILE) $(PCA_OUTPUT) $(MODEL_FILE) $(EVALUATION_REPORT)
# Construire l'image Docker
docker-build:
	docker build -t $(DOCKER_USERNAME)/$(IMAGE_NAME):$(TAG) .

# Pousser l'image Docker sur Docker Hub
docker-push: docker-build
	docker push $(DOCKER_USERNAME)/$(IMAGE_NAME):$(TAG)

# Lancer le conteneur Docker
docker-run: docker-build
	docker run -p 8000:8000 $(DOCKER_USERNAME)/$(IMAGE_NAME):$(TAG)

# Se connecter à Docker Hub
docker-login:
	docker login

# Nettoyer les conteneurs et images Docker
docker-clean:
	docker stop $$(docker ps -aq)  # Arrêter tous les conteneurs
	docker rm $$(docker ps -aq)    # Supprimer tous les conteneurs
	docker rmi $$(docker images -q) # Supprimer toutes les images
