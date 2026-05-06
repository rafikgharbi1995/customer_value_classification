import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# ---------------------------
# Configuration
# ---------------------------
st.set_page_config(page_title="Classification Clients", page_icon="📊", layout="wide")
st.title("🎯 Classification des Clients par Valeur")
st.markdown("Prédisez la catégorie de valeur d'un client à partir de ses indicateurs RFM.")

# ---------------------------
# Chargement du modèle (avec cache et gestion d'erreur)
# ---------------------------
@st.cache_resource
def load_model():
    model_path = Path('models/final_rf_model.pkl')
    if not model_path.exists():
        st.error("❌ Modèle introuvable. Placez 'final_rf_model.pkl' dans le dossier 'models/'.")
        st.stop()
    data = joblib.load(model_path)
    return data['model'], data['scaler']

model, scaler = load_model()

# Mapping des classes
class_labels = {0: "Basse Valeur", 1: "Moyenne Valeur", 2: "Haute Valeur"}
class_emojis  = {0: "🟡", 1: "🟠", 2: "🟢"}

# ---------------------------
# Initialisation du session state
# ---------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Sidebar – Prédiction unitaire
# ---------------------------
st.sidebar.header("📋 Paramètres du Client")
with st.sidebar.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.slider("Récence (jours)", 0, 365, 30)
    with col2:
        frequency = st.slider("Fréquence (commandes)", 0, 200, 5)
    with col3:
        monetary = st.slider("Montant total (€)", 0.0, 100000.0, 500.0, step=10.0)

    submitted = st.form_submit_button("🔮 Prédire la Catégorie")
    reset = st.form_submit_button("🔄 Réinitialiser")

if reset:
    st.rerun()

# ---------------------------
# Prédiction unitaire
# ---------------------------
if submitted:
    try:
        features = np.array([[recency, frequency, monetary]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        # Sauvegarde dans l'historique
        st.session_state.history.append({
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "Predicted": class_labels[prediction],
            "Proba_Haute": proba[2],
            "Proba_Moyenne": proba[1],
            "Proba_Basse": proba[0]
        })

        # Affichage résultat principal
        st.success(f"### {class_emojis[prediction]} Catégorie prédite : **{class_labels[prediction]}**")

        # Visualisation des probabilités
        prob_df = pd.DataFrame({
            "Classe": [class_labels[2], class_labels[1], class_labels[0]],
            "Probabilité": [proba[2]*100, proba[1]*100, proba[0]*100]
        }).sort_values("Probabilité", ascending=False)

        col_chart, col_comment = st.columns([1, 1])
        with col_chart:
            fig = px.bar(
                prob_df, x="Probabilité", y="Classe", orientation='h',
                color="Classe",
                color_discrete_map={"Haute Valeur": "green", "Moyenne Valeur": "orange", "Basse Valeur": "gold"},
                title="Probabilités par classe",
                range_x=[0, 100]
            )
            fig.update_layout(showlegend=False, height=250)
            st.plotly_chart(fig, use_container_width=True)

        with col_comment:
            st.markdown("**Analyse comportementale :**")
            if prediction == 2:
                st.success("Client très récent, fréquent et/ou à fort montant. Fidélisez-le avec des offres VIP.")
            elif prediction == 1:
                st.info("Client régulier mais peut être amélioré. Relancez-le avec des promotions ciblées.")
            else:
                st.warning("Client à faible engagement. Essayez une campagne de réactivation.")

        # Affichage des métriques sous forme de cartes
        m1, m2, m3 = st.columns(3)
        m1.metric("🔁 Récence", f"{recency} jours")
        m2.metric("📦 Fréquence", f"{frequency} commandes")
        m3.metric("💰 Montant", f"{monetary:.2f} €")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

# ---------------------------
# Batch prediction (upload CSV)
# ---------------------------
st.markdown("---")
st.subheader("📁 Prédiction par lot (CSV)")
uploaded_file = st.file_uploader("Téléchargez un fichier avec les colonnes : Recency, Frequency, Monetary", type=["csv"])
if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        required_cols = {"Recency", "Frequency", "Monetary"}
        if not required_cols.issubset(df_input.columns):
            st.error(f"Le fichier doit contenir les colonnes : {', '.join(required_cols)}")
        else:
            features_batch = df_input[list(required_cols)].values
            features_scaled_batch = scaler.transform(features_batch)
            predictions = model.predict(features_scaled_batch)
            probas = model.predict_proba(features_scaled_batch)

            df_result = df_input.copy()
            df_result["Catégorie"] = [class_labels[p] for p in predictions]
            df_result["Probabilité (Haute)"]  = probas[:, 2].round(3)
            df_result["Probabilité (Moyenne)"] = probas[:, 1].round(3)
            df_result["Probabilité (Basse)"]  = probas[:, 0].round(3)

            st.dataframe(df_result, use_container_width=True)
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger les résultats", csv, "predictions.csv", "text/csv")

            # Graphique de distribution des classes
            fig_dist = px.pie(names=df_result["Catégorie"].value_counts().index,
                              values=df_result["Catégorie"].value_counts().values,
                              title="Répartition des catégories prédites")
            st.plotly_chart(fig_dist, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")

# ---------------------------
# Historique des prédictions
# ---------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕒 Historique des dernières prédictions")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df.tail(10), use_container_width=True)

# ---------------------------
# Section explicative
# ---------------------------
st.markdown("---")
st.header("ℹ️ À propos du modèle")
st.markdown("""
Ce système segmente les clients en trois niveaux de valeur :
- **🟢 Haute Valeur** : Top 20 % (récence faible, fréquence/montant élevés)
- **🟠 Moyenne Valeur** : 40 % intermédiaires
- **🟡 Basse Valeur** : 40 % restants (occasionnels ou perdus)

**Indicateurs RFM :**
- **Récence** : jours depuis la dernière commande
- **Fréquence** : nombre total de commandes
- **Montant** : chiffre d'affaires cumulé
""")
