import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle et génère des visualisations
    """
    # Prédictions
    y_pred = model.predict(X_test)

    # Rapport de classification
    print(" RAPPORT DE CLASSIFICATION:")
    print(classification_report(y_test, y_pred, target_names=['Bas', 'Moyen', 'Haut']))

    # Matrice de confusion
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bas', 'Moyen', 'Haut'],
                yticklabels=['Bas', 'Moyen', 'Haut'])
    plt.title('Matrice de Confusion')
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.show()

    # Importance des features
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Importance des Features')
        plt.show()


if __name__ == "__main__":
    # Charger le modèle et les données de test
    model = joblib.load('../models/best_model.pkl')

    from model_training import prepare_data
    from feature_engineering import create_customer_labels, calculate_rfm_features
    from data_cleaning import load_and_clean_data

    df = load_and_clean_data('../data/Online Retail.xlsx')
    rfm_df = calculate_rfm_features(df)
    final_df = create_customer_labels(rfm_df)

    X_train, X_test, y_train, y_test = prepare_data(final_df)
    evaluate_model(model, X_test, y_test)