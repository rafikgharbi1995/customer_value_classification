import pandas as pd
import numpy as np
from datetime import datetime


def calculate_rfm_features(df):
    """
    Calcule les features RFM pour chaque client
    """
    print(" Calcul des métriques RFM...")

    # Date de référence (1 jour après la dernière commande)
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Agrégation par client
    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Récence
        'InvoiceNo': 'nunique',  # Fréquence
        'TotalAmount': 'sum'  # Montant
    }).reset_index()

    # Renommage des colonnes
    rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    print(f" RFM calculé pour {len(rfm_df)} clients")
    return rfm_df


def create_customer_labels(rfm_df):
    """
    Crée les labels de valeur client (0=Bas, 1=Moyen, 2=Haut)
    basé sur le score Monetary
    """
    print(" Création des labels de valeur...")

    # Segmentation basée sur les percentiles Monetary
    rfm_df['Value_Class'] = 0  # Par défaut Basse valeur

    # Clients Haute valeur (top 20%)
    high_value_threshold = rfm_df['Monetary'].quantile(0.8)
    high_value_mask = (rfm_df['Monetary'] >= high_value_threshold)
    rfm_df.loc[high_value_mask, 'Value_Class'] = 2

    # Clients Moyenne valeur (40% suivants)
    medium_value_threshold = rfm_df['Monetary'].quantile(0.4)
    medium_value_mask = (rfm_df['Monetary'] >= medium_value_threshold) & ~high_value_mask
    rfm_df.loc[medium_value_mask, 'Value_Class'] = 1

    # Statistiques des classes
    class_distribution = rfm_df['Value_Class'].value_counts().sort_index()
    print(" Distribution des classes de valeur:")
    for class_id, count in class_distribution.items():
        classes = {0: 'Basse', 1: 'Moyenne', 2: 'Haute'}
        percentage = (count / len(rfm_df)) * 100
        print(f"   {classes[class_id]} valeur: {count} clients ({percentage:.1f}%)")

    return rfm_df


if __name__ == "__main__":
    from data_cleaning import load_and_clean_data

    # Pipeline complète
    print(" DÉMARRAGE PHASE 2 - FEATURE ENGINEERING")
    df = load_and_clean_data('data/Online Retail.xlsx')
    rfm_df = calculate_rfm_features(df)
    final_df = create_customer_labels(rfm_df)

    print("\n Aperçu du dataset RFM final:")
    print(final_df.head())
    print(f"\n Statistiques RFM:")
    print(final_df[['Recency', 'Frequency', 'Monetary']].describe())