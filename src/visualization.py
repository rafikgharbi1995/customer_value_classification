import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from feature_engineering import calculate_rfm_features, create_customer_labels
from data_cleaning import load_and_clean_data

# Configuration des styles
plt.style.use('default')
sns.set_palette("husl")


def plot_rfm_distributions(rfm_df):
    """
    Visualise les distributions des métriques RFM
    """
    print(" Création des visualisations RFM...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Distribution de la Récence
    axes[0, 0].hist(rfm_df['Recency'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution de la Récence (jours)')
    axes[0, 0].set_xlabel('Jours depuis dernier achat')
    axes[0, 0].set_ylabel('Nombre de clients')

    # Distribution de la Fréquence
    axes[0, 1].hist(rfm_df['Frequency'], bins=50, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Distribution de la Fréquence')
    axes[0, 1].set_xlabel('Nombre de commandes')
    axes[0, 1].set_ylabel('Nombre de clients')
    axes[0, 1].set_xlim(0, 50)  # Zoom sur la majorité des données

    # Distribution du Montant
    axes[1, 0].hist(rfm_df['Monetary'], bins=50, alpha=0.7, color='salmon')
    axes[1, 0].set_title('Distribution du Montant (€)')
    axes[1, 0].set_xlabel('Chiffre d\'affaires total (€)')
    axes[1, 0].set_ylabel('Nombre de clients')
    axes[1, 0].set_xlim(0, 10000)  # Zoom sur la majorité des données

    # Heatmap de corrélation RFM
    correlation_matrix = rfm_df[['Recency', 'Frequency', 'Monetary']].corr()
    im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_title('Corrélations entre métriques RFM')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels(['Récence', 'Fréquence', 'Montant'])
    axes[1, 1].set_yticklabels(['Récence', 'Fréquence', 'Montant'])

    # Ajouter les valeurs de corrélation
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                            ha='center', va='center', color='white', fontweight='bold')

    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()

    return correlation_matrix


def plot_class_analysis(rfm_df):
    """
    Analyse visuelle des classes de valeur
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Distribution des classes
    class_counts = rfm_df['Value_Class'].value_counts().sort_index()
    colors = ['red', 'orange', 'green']
    labels = ['Basse Valeur', 'Moyenne Valeur', 'Haute Valeur']

    axes[0, 0].pie(class_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Répartition des Clients par Classe de Valeur')

    # Boxplot Récence par classe
    sns.boxplot(data=rfm_df, x='Value_Class', y='Recency', ax=axes[0, 1], palette=colors)
    axes[0, 1].set_title('Récence par Classe de Valeur')
    axes[0, 1].set_xlabel('Classe de Valeur')
    axes[0, 1].set_ylabel('Jours depuis dernier achat')
    axes[0, 1].set_xticklabels(labels)

    # Boxplot Fréquence par classe
    sns.boxplot(data=rfm_df, x='Value_Class', y='Frequency', ax=axes[1, 0], palette=colors)
    axes[1, 0].set_title('Fréquence par Classe de Valeur')
    axes[1, 0].set_xlabel('Classe de Valeur')
    axes[1, 0].set_ylabel('Nombre de commandes')
    axes[1, 0].set_xticklabels(labels)

    # Boxplot Montant par classe
    sns.boxplot(data=rfm_df, x='Value_Class', y='Monetary', ax=axes[1, 1], palette=colors)
    axes[1, 1].set_title('Montant par Classe de Valeur')
    axes[1, 1].set_xlabel('Classe de Valeur')
    axes[1, 1].set_ylabel('Chiffre d\'affaires (€)')
    axes[1, 1].set_xticklabels(labels)

    plt.tight_layout()
    plt.show()


def plot_scatter_analysis(rfm_df):
    """
    Analyse par scatter plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter Plot Fréquence vs Montant
    scatter = axes[0].scatter(rfm_df['Frequency'], rfm_df['Monetary'],
                              c=rfm_df['Value_Class'], cmap='viridis', alpha=0.6)
    axes[0].set_title('Fréquence vs Montant par Classe de Valeur')
    axes[0].set_xlabel('Fréquence (nombre commandes)')
    axes[0].set_ylabel('Montant (€)')
    axes[0].grid(True, alpha=0.3)

    # Scatter Plot Récence vs Montant
    axes[1].scatter(rfm_df['Recency'], rfm_df['Monetary'],
                    c=rfm_df['Value_Class'], cmap='viridis', alpha=0.6)
    axes[1].set_title('Récence vs Montant par Classe de Valeur')
    axes[1].set_xlabel('Récence (jours)')
    axes[1].set_ylabel('Montant (€)')
    axes[1].grid(True, alpha=0.3)

    # Légende
    legend_labels = ['Basse Valeur', 'Moyenne Valeur', 'Haute Valeur']
    for i, label in enumerate(legend_labels):
        axes[0].scatter([], [], c=[plt.cm.viridis(i / 2)], label=label, alpha=0.6)
    axes[0].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(" PHASE 2 - VISUALISATION DES DONNÉES")

    # Chargement des données
    df = load_and_clean_data('data/Online Retail.xlsx')
    rfm_df = calculate_rfm_features(df)
    final_df = create_customer_labels(rfm_df)

    # Visualisations
    correlation_matrix = plot_rfm_distributions(final_df)
    plot_class_analysis(final_df)
    plot_scatter_analysis(final_df)

    print("\n ANALYSE DES CORRÉLATIONS RFM:")
    print(correlation_matrix)

    print("\n CONCLUSIONS VISUELLES:")
    print("• Récence: Plus un client est récent, plus il a de valeur")
    print("• Fréquence: Les clients à haute valeur achètent plus fréquemment")
    print("• Montant: Forte corrélation avec la valeur client")
    print("• Les 3 métriques RFM sont complémentaires pour la classification")