##Créer un environnement virtuel:
python -m venv venv
##Activer l'environnement virtuel
venv\Scripts\activate
##Installer les dépendances
pip install -r mlops/requirements.txt
##Installer les dépendances simplifiées
pip install -r requirements-simple.txt
##Prétraitement des données
python -m mlops.preprocessing.preprocess_csv --input mlops/data/raw/reviews.csv --output mlops/data/processed/reviews_processed.csv --lowercase --remove-punctuation
##Entraînement du modèle
python -m mlops.main train --data mlops/data/processed/reviews_processed.csv --model-output models/svm_model.pkl --model-type svm
##Évaluation du modèle
python -m mlops.evaluation.evaluate --model models/svm_model.pkl --data mlops/data/processed/reviews_processed.csv --output metrics/evaluation_metrics.json
##Démarrage de l'API
python -m mlops.main service --model models/svm_model.pkl --port 8000
##Déploiement avec Docker
docker-compose -f mlops/docker/docker-compose.yml up -d
docker-compose -f mlops/docker/docker-compose.yml up -d --build
##Démarrez l'API
python -m mlops.main service --model models/svm_model.pkl --port 8000

##Ouvrez le fichier HTML** dans votre navigateur:
Double-cliquez sur `interface_moderne.html` ou ouvrez-le avec votre navigateur

##Testez l'inscription

- Allez dans l'onglet "Inscription"
- Remplissez les champs et cliquez sur "S'inscrire"
- Vous devriez voir un message de succès

##Testez la connexion
##Testez la classification
