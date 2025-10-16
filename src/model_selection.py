import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


def prepare_model_data(rfm_df):
    """
    Prépare les données pour la modélisation
    """
    print(" Préparation des données pour la modélisation...")

    # Features et target
    X = rfm_df[['Recency', 'Frequency', 'Monetary']]
    y = rfm_df['Value_Class']

    # Normalisation des features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f" Données d'entraînement: {X_train.shape}")
    print(f" Données de test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler


def evaluate_models(X_train, X_test, y_train, y_test):
    """
    Évalue plusieurs algorithmes de classification
    """
    print("\n ÉVALUATION DES ALGORITHMES DE CLASSIFICATION")
    print("=" * 50)

    # Définition des modèles à tester
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    results = {}

    for name, model in models.items():
        print(f"\n Entraînement de {name}...")

        # Entraînement
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Évaluation
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }

        print(f" {name} - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Rapport détaillé
        print(f" Rapport de classification pour {name}:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Basse Valeur', 'Moyenne Valeur', 'Haute Valeur']))

    return results


def select_best_model(results):
    """
    Sélectionne le meilleur modèle basé sur l'accuracy
    """
    print("\n SÉLECTION DU MEILLEUR MODÈLE")
    print("=" * 40)

    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']

    print(f"  Meilleur modèle: {best_model_name}")
    print(f" Accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")

    # Comparaison des performances
    print("\n COMPARAISON DES MODÈLES:")
    for name, result in results.items():
        print(f"   {name}: {result['accuracy']:.4f} ({result['accuracy'] * 100:.2f}%)")

    return best_model_name, results[best_model_name]


def feature_importance_analysis(best_model, feature_names):
    """
    Analyse l'importance des features pour le meilleur modèle
    """
    print("\n ANALYSE DE L'IMPORTANCE DES FEATURES")

    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print(" Importance des features (Random Forest/XGBoost):")
        for _, row in importance_df.iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")

    elif hasattr(best_model, 'coef_'):
        print(" Coefficients (Régression Logistique):")
        for i, feature in enumerate(feature_names):
            print(f"   {feature}: {best_model.coef_[0][i]:.4f}")


if __name__ == "__main__":
    from feature_engineering import calculate_rfm_features, create_customer_labels
    from data_cleaning import load_and_clean_data

    print(" PHASE 2 - SÉLECTION DU MODÈLE")

    # Chargement et préparation des données
    df = load_and_clean_data('data/Online Retail.xlsx')
    rfm_df = calculate_rfm_features(df)
    final_df = create_customer_labels(rfm_df)

    # Préparation pour la modélisation
    X_train, X_test, y_train, y_test, scaler = prepare_model_data(final_df)

    # Évaluation des modèles
    results = evaluate_models(X_train, X_test, y_train, y_test)

    # Sélection du meilleur modèle
    best_model_name, best_result = select_best_model(results)

    # Analyse des features
    feature_importance_analysis(best_result['model'], ['Recency', 'Frequency', 'Monetary'])

    print("\n RECOMMANDATION FINALE:")
    print(f"• Modèle sélectionné: {best_model_name}")
    print("• Justification: Meilleure performance (accuracy) sur les données de test")
    print("• Les 3 features RFM sont pertinentes pour la classification")
    print("• Prêt pour la Phase 3: Optimisation et entraînement final")