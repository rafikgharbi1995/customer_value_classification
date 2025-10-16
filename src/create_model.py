import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from feature_engineering import calculate_rfm_features, create_customer_labels
from data_cleaning import load_and_clean_data


def create_and_save_model():
    """Crée et sauvegarde le modèle manuellement"""
    print(" Création du modèle...")

    # Chargement des données
    df = load_and_clean_data('data/Online Retail.xlsx')
    rfm_df = calculate_rfm_features(df)
    final_df = create_customer_labels(rfm_df)

    # Préparation des données
    X = final_df[['Recency', 'Frequency', 'Monetary']]
    y = final_df['Value_Class']

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Création du dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)

    # Sauvegarde
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': ['Recency', 'Frequency', 'Monetary']
    }

    joblib.dump(model_data, 'models/final_rf_model.pkl')
    print(" Modèle sauvegardé : models/final_rf_model.pkl")

    # Vérification
    if os.path.exists('models/final_rf_model.pkl'):
        print(" Fichier .pkl créé avec succès !")
    else:
        print(" Erreur : le fichier n'a pas été créé")


if __name__ == "__main__":
    create_and_save_model()