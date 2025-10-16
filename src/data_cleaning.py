import pandas as pd
import numpy as np


def load_and_clean_data(file_path):
    """
    Charge et nettoie le dataset Online Retail
    """
    # Chargement des données
    df = pd.read_excel(file_path)

    print(" Dimensions initiales:", df.shape)

    # Suppression des lignes avec CustomerID manquant
    df = df.dropna(subset=['CustomerID'])

    # Conversion CustomerID en entier
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Suppression des quantités négatives (retours)
    df = df[df['Quantity'] > 0]

    # Suppression des prix négatifs ou nuls
    df = df[df['UnitPrice'] > 0]

    # Calcul du montant total par ligne
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    print(" Nettoyage terminé - Dimensions finales:", df.shape)
    return df


if __name__ == "__main__":
    df = load_and_clean_data('C:/Users/helpdesk10/Desktop/customer_value_classification/data/Online Retail.xlsx')
    print(df.head())