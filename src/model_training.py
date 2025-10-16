import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def advanced_model_training(rfm_df):
    """
    Phase 3 - Entraînement avancé et optimisation du modèle
    """
    print(" PHASE 3 - ENTRAÎNEMENT ET OPTIMISATION DU MODÈLE")
    print("=" * 60)

    # 1. Préparation des données
    print("\n 1. PRÉPARATION DES DONNÉES POUR L'ENTRAÎNEMENT")
    X = rfm_df[['Recency', 'Frequency', 'Monetary']]
    y = rfm_df['Value_Class']

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f" Données d'entraînement: {X_train.shape}")
    print(f" Données de test: {X_test.shape}")

    # 2. Entraînement du modèle initial
    print("\n 2. ENTRAÎNEMENT DU MODÈLE RANDOM FOREST")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Prédictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    # 3. Évaluation détaillée
    print("\n 3. ÉVALUATION DÉTAILLÉE DU MODÈLE")

    # Accuracy train vs test
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f" Accuracy entraînement: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f" Accuracy test: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Rapport de classification détaillé
    print(f"\n RAPPORT DE CLASSIFICATION (TEST):")
    print(classification_report(y_test, y_pred_test,
                                target_names=['Basse Valeur', 'Moyenne Valeur', 'Haute Valeur']))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Basse', 'Moyenne', 'Haute'],
                yticklabels=['Basse', 'Moyenne', 'Haute'])
    plt.title('Matrice de Confusion - Random Forest')
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Étiquette prédite')
    plt.show()

    # 4. Détection surajustement/sous-ajustement
    print("\n 4. ANALYSE SURAJUSTEMENT/SOUS-AJUSTEMENT")

    # Validation croisée
    cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5)
    print(f" Validation croisée (5 folds): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Courbe d'apprentissage
    plot_learning_curve(rf_model, X_scaled, y)

    # Analyse du gap train/test
    accuracy_gap = train_accuracy - test_accuracy
    if accuracy_gap > 0.1:
        print(f"  ALERTE: Écart important train/test: {accuracy_gap:.4f} (risque de surajustement)")
    elif accuracy_gap < 0.01:
        print(f" Bon équilibre: Écart train/test négligeable: {accuracy_gap:.4f}")
    else:
        print(f" Écart train/test acceptable: {accuracy_gap:.4f}")

    return rf_model, scaler, X_train, X_test, y_train, y_test, train_accuracy, test_accuracy


def plot_learning_curve(model, X, y):
    """
    Trace la courbe d'apprentissage pour détecter surajustement/sous-ajustement
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Score entraînement')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Score validation')
    plt.title('Courbe d\'Apprentissage')
    plt.xlabel('Taille de l\'ensemble d\'entraînement')
    plt.ylabel('Score Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def hyperparameter_optimization(X_train, y_train):
    """
    Optimisation des hyperparamètres avec GridSearch
    """
    print("\n  5. OPTIMISATION DES HYPERPARAMÈTRES")

    # Définition des hyperparamètres à tester
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f" Meilleurs paramètres: {grid_search.best_params_}")
    print(f" Meilleur score (validation): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def save_final_model(model, scaler, filename='../models/final_rf_model.pkl'):
    """
    Sauvegarde le modèle final et le scaler
    """
    import os
    os.makedirs('../models', exist_ok=True)

    # Sauvegarde du modèle et du scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': ['Recency', 'Frequency', 'Monetary']
    }

    joblib.dump(model_data, filename)
    print(f" Modèle final sauvegardé: {filename}")


if __name__ == "__main__":
    from feature_engineering import calculate_rfm_features, create_customer_labels
    from data_cleaning import load_and_clean_data

    # Chargement des données
    df = load_and_clean_data('data/Online Retail.xlsx')
    rfm_df = calculate_rfm_features(df)
    final_df = create_customer_labels(rfm_df)

    # Phase 3 complète
    rf_model, scaler, X_train, X_test, y_train, y_test, train_acc, test_acc = advanced_model_training(final_df)

    # Optimisation des hyperparamètres
    best_model = hyperparameter_optimization(X_train, y_train)

    # Évaluation du modèle optimisé
    best_test_accuracy = best_model.score(X_test, y_test)
    print(f"\n PERFORMANCE MODÈLE OPTIMISÉ:")
    print(f" Accuracy test (optimisé): {best_test_accuracy:.4f} ({best_test_accuracy * 100:.2f}%)")

    # Sauvegarde du modèle final
    save_final_model(best_model, scaler)

    print("\n PHASE 3 TERMINÉE AVEC SUCCÈS!")
    print("• Modèle entraîné et optimisé")
    print("• Évaluation complète effectuée")
    print("• Modèle sauvegardé pour le déploiement")