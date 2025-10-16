import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger le modèle
@st.cache_resource
def load_model():
    model_data = joblib.load('models/final_rf_model.pkl')
    return model_data['model'], model_data['scaler']

# Configuration de la page
st.set_page_config(
    page_title="Classification Clients",
    page_icon="📊",
    layout="wide"
)

# Titre
st.title("🎯 Classification des Clients par Valeur")
st.markdown("Prédisez la catégorie de valeur d'un client basé sur son comportement RFM")

# Sidebar pour les inputs
st.sidebar.header("📋 Paramètres du Client")

recency = st.sidebar.number_input("Récence (jours)", min_value=0, value=30)
frequency = st.sidebar.number_input("Fréquence (nb commandes)", min_value=1, value=5)
monetary = st.sidebar.number_input("Montant total (€)", min_value=0.0, value=500.0)

# Prédiction
if st.sidebar.button("🔮 Prédire la Catégorie"):
    model, scaler = load_model()

    # Préparation des features
    features = np.array([[recency, frequency, monetary]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    # Mapping des classes
    classes = {0: '🟡 Basse Valeur', 1: '🟠 Moyenne Valeur', 2: '🟢 Haute Valeur'}

    # Affichage des résultats
    st.success(f"**Catégorie prédite : {classes[prediction]}**")

    # Probabilités
    st.subheader("📈 Probabilités par Classe")
    prob_df = pd.DataFrame({
        'Classe': ['Basse Valeur', 'Moyenne Valeur', 'Haute Valeur'],
        'Probabilité': [f"{p * 100:.1f}%" for p in proba]
    })
    st.table(prob_df)

# Section explicative
st.markdown("---")
st.subheader("ℹ️ À propos de ce projet")
st.markdown("""
Ce système classe les clients en 3 catégories basées sur leur valeur :
- **🟢 Haute Valeur** : Clients les plus rentables (top 20%)
- **🟠 Moyenne Valeur** : Clients réguliers (40% suivants)  
- **🟡 Basse Valeur** : Clients occasionnels (40% restants)

**Métriques utilisées :**
- **Récence** : Jours depuis le dernier achat
- **Fréquence** : Nombre total de commandes
- **Montant** : Chiffre d'affaires généré
""")

if __name__ == "__main__":
    pass