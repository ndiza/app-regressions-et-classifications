import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# Configuration de la page
st.set_page_config(
    page_title="Plateforme Multi-Modèles ML",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# CHARGEMENT DES MODÈLES AVEC CACHE (@st.cache_resource)
# ==========================================

@st.cache_resource
def load_taxi_assets():
    model = joblib.load('modele_reg_taxi.joblib')
    encoders = joblib.load('encoders_reg_taxi.joblib')
    scaler = joblib.load('scaler_reg_taxi.joblib') 
    uniques = joblib.load('uniques_taxi.joblib') 
    return model, encoders, scaler, uniques

@st.cache_resource
def load_car_reg_assets():
    model = joblib.load('modele_reg_voiture.joblib')
    scaler = joblib.load('scaler_reg_voiture.joblib')
    encoders = joblib.load('encoders_reg_voiture.joblib')
    return model, scaler, encoders

@st.cache_resource
def load_car_class_assets():
    model = joblib.load('model_classif_voiture.joblib')
    scaler = joblib.load('scaler_classif_voiture.joblib')
    encoder = joblib.load('encoders_classif_voiture.joblib')
    uniques = joblib.load('uniques_classif_voiture.joblib')
    return model, scaler, encoder, uniques

@st.cache_resource
def load_tel_class_assets():
    model = joblib.load('model_classif_telephone.joblib')
    scaler = joblib.load('scaler_classif_telephone.joblib')
    encoder = joblib.load('encoders_classif_telephone.joblib')
    uniques = joblib.load('telephone_uniques.joblib')
    return model, scaler, encoder, uniques

# ==========================================
# BARRE LATÉRALE : SÉLECTION DU MODÈLE
# ==========================================
st.sidebar.title("🧭 Navigation")
choix_modele = st.sidebar.radio(
    "Sélectionnez un modèle :",
    (
        "🚕 Régression - Prix des Taxis",
        "🚙 Régression - Prix des Voitures (Car Dataset)",
        "🚗 Classification - Statut Voitures",
        "📱 Classification - État des Téléphones"
    )
)

st.sidebar.markdown("---")
st.sidebar.info("Chaque modèle dispose de ses onglets de prédiction simple et multiple.")

# ==========================================
# 1. SECTION : RÉGRESSION TAXIS
# ==========================================
if choix_modele == "🚕 Régression - Prix des Taxis":
    st.title("🚕 Taxi Trip Price Prediction")
    
    # Chargement des objets
    try:
        xgb_model, encoders, scaler_taxi, uniques = load_taxi_assets()
    except Exception as e:
        st.error(f"Erreur de chargement des fichiers joblib pour les Taxis : {e}")
        st.stop()

    # Fonctions de prédiction Taxis
    def Pred_func_taxi(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                       Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes):
        Time_of_Day = encoders[0].transform([Time_of_Day])[0]
        Day_of_Week = encoders[1].transform([Day_of_Week])[0]
        Traffic_Conditions = encoders[2].transform([Traffic_Conditions])[0]
        Weather = encoders[3].transform([Weather])[0]
        
        x_new = np.array([Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                          Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes], dtype=float)
        x_new = x_new.reshape(1, -1)
        y_pred = xgb_model.predict(x_new)
        return round(float(y_pred[0]), 2)

    def Pred_func_csv_taxi(df_input):
        predictions = []
        for row in df_input.iloc[:, :10].values:
            pred = Pred_func_taxi(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
            predictions.append(pred)
        df_input['Trip_Price'] = predictions
        return df_input

    # Création des deux onglets
    tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])

    with tab1:
        st.subheader("Predict the price of a taxi trip with a single input")
        with st.form('form_simple_taxi'):
            col1, col2 = st.columns(2)
            with col1:
                Trip_Distance_km = st.number_input('Trip Distance (km)', min_value=0.0, value=25.0, step=0.5)
                Time_of_Day = st.selectbox('Time of Day', options=uniques[0])
                Day_of_Week = st.selectbox('Day of Week', options=uniques[1])
                Passenger_Count = st.number_input('Passenger Count', min_value=1, max_value=4, value=2, step=1)
                Traffic_Conditions = st.selectbox('Traffic Conditions', options=uniques[2])
            with col2:
                Weather = st.selectbox('Weather', options=uniques[3])
                Base_Fare = st.number_input('Base Fare', min_value=0.0, value=3.5, step=0.1)
                Per_Km_Rate = st.number_input('Per Km Rate', min_value=0.0, value=1.2, step=0.05)
                Per_Minute_Rate = st.number_input('Per Minute Rate', min_value=0.0, value=0.3, step=0.01)
                Trip_Duration_Minutes = st.number_input('Trip Duration (minutes)', min_value=0.0, value=60.0, step=1.0)
            submit_taxi = st.form_submit_button('Predict')

        if submit_taxi:
            y_pred = Pred_func_taxi(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                                    Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes)
            st.success(f"Trip Price : {y_pred} $")

    with tab2:
        st.subheader("Predict the price of a taxi trip with multiple inputs (CSV)")
        file_taxi = st.file_uploader('Upload a csv file (Taxi)', type=['csv'], key='up_taxi')
        if file_taxi is not None:
            df_taxi = pd.read_csv(file_taxi)
            st.write("Aperçu des données chargées :")
            st.dataframe(df_taxi.head())
            if st.button('Predict Batch (Taxi)'):
                df_pred_taxi = Pred_func_csv_taxi(df_taxi)
                st.write("Résultats des prédictions :")
                st.dataframe(df_pred_taxi)
                st.download_button(label='Download predictions CSV',
                                   data=df_pred_taxi.to_csv(index=False).encode('utf-8'),
                                   file_name='taxi_predictions.csv',
                                   mime='text/csv')

# ==========================================
# 2. SECTION : RÉGRESSION VOITURES (Car Dataset)
# ==========================================
elif choix_modele == "🚙 Régression - Prix des Voitures (Car Dataset)":
    st.title("🚙 Car Price Prediction (Regression)")
    
    try:
        rf_model_car, scaler_car, encoders_car = load_car_reg_assets()
    except Exception as e:
        st.error(f"Erreur de chargement des objets pour la régression voitures : {e}")
        st.stop()

    tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])

    with tab1:
        st.subheader("Estimez le prix de vente d'un véhicule")
        with st.form('form_simple_car_reg'):
            col1, col2 = st.columns(2)
            with col1:
                kms = st.number_input('Kms Driven', min_value=500, value=40000)
                present_price = st.number_input('Present Price', min_value=0.1, value=10.0)
                fuel = st.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'CNG'])
            with col2:
                seller = st.selectbox('Seller Type', options=['Dealer', 'Individual'])
                trans = st.selectbox('Transmission', options=['Manual', 'Automatic'])
                age = st.number_input('Age (years)', min_value=1, value=5)
            submit_car_reg = st.form_submit_button('Predict Price')

        if submit_car_reg:
            try:
                # 1. Création du DataFrame avec les entrées utilisateur 
                input_data = pd.DataFrame({
                    'Kms_Driven': [kms],
                    'Present_Price': [present_price],
                    'Fuel_Type': [fuel],
                    'Seller_Type': [seller],
                    'Transmission': [trans],
                    'Age': [age]
                })
                
                # 2. Encodage des colonnes catégorielles avec les encoders sauvegardés
                cat_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
                for i, col in enumerate(cat_cols):
                    input_data[col] = encoders_car[i].transform(input_data[col])
                
                # 3. Normalisation des données avec le scaler sauvegardé
                input_scaled = scaler_car.transform(input_data)
                
                # 4. Prédiction du prix avec le Random Forest Regressor
                prediction = rf_model_car.predict(input_scaled)[0]
                
                # 5. Affichage du résultat
                st.success(f"Prix de vente estimé du véhicule : **{prediction:.2f}** (Millions / Unité de votre dataset)")
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

    with tab2:
        st.subheader("Prédiction multiple (CSV) - Voitures")
        file_car_reg = st.file_uploader('Upload CSV (Car Reg)', type=['csv'], key='up_car_reg')
        if file_car_reg is not None:
            df_car = pd.read_csv(file_car_reg)
            st.dataframe(df_car.head())

# ==========================================
# 3. SECTION : CLASSIFICATION VOITURES
# ==========================================
elif choix_modele == "🚗 Classification - Statut Voitures":
    st.title("🚗 Classification - Statut du Véhicule")
    
    try:
        model_cv, scaler_cv, encoder_cv, uniques_cv = load_car_class_assets()
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.stop()

    tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])
    
    with tab1:
        st.subheader("Prédire le statut d'une voiture")
        with st.form('form_simple_car_class'):
            col1, col2 = st.columns(2)
            with col1:
                c1_marque = st.selectbox("Marque", uniques_cv['Marque'])
                c1_annee = st.number_input("Année", min_value=1990, max_value=2026, value=2015)
                c1_boite = st.selectbox("Boîte de vitesse", uniques_cv['Boite_Vitesse'])
            with col2:
                c1_adresse = st.selectbox("Adresse", uniques_cv['Adresse'])
                c1_prix = st.number_input("Prix (FCFA)", min_value=500000, max_value=100000000, value=5000000)
            
            submit_class = st.form_submit_button('Prédire le Statut')

        if submit_class:
            try:
                # 1. DataFrame dans l'ordre exact des colonnes d'entraînement
                input_data = pd.DataFrame({
                    'Marque': [c1_marque],
                    'Année': [c1_annee],
                    'Boite_Vitesse': [c1_boite],
                    'Adresse': [c1_adresse],
                    'Prix': [c1_prix]
                })

                # 2. Encodage des colonnes catégorielles (encodeurs = dict par colonne)
                for col in ['Marque', 'Boite_Vitesse', 'Adresse']:
                    input_data[col] = encoder_cv[col].transform(input_data[col])

                # 3. Normalisation puis prédiction
                input_scaled = scaler_cv.transform(input_data)
                prediction = model_cv.predict(input_scaled)[0]
                statut = encoder_cv['Venant_Occasion'].inverse_transform([prediction])[0]

                st.success(f"Statut prédit : **{statut}**")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

    with tab2:
        st.subheader("Prédiction multiple (CSV) - Voitures")
        file_cc = st.file_uploader('Upload a csv file', type=['csv'], key='up_cc')
        if file_cc is not None:
            df_cc = pd.read_csv(file_cc)
            st.dataframe(df_cc.head())
            if st.button('Lancer la prédiction multiple (Voitures)'):
                st.info("Traitement par lot en cours...")
# ==========================================
# 4. SECTION : CLASSIFICATION TÉLÉPHONES
# ==========================================
elif choix_modele == "📱 Classification - État des Téléphones":
    st.title("📱 Classification - État du Téléphone")
    
    try:
        model_tel, scaler_tel, encoder_tel, uniques_tel = load_tel_class_assets()
    except Exception as e:
        st.error(f"Erreur de chargement des objets pour les téléphones : {e}")
        st.stop()

    tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])
    
    with tab1:
        st.subheader("Prédire l'état d'un téléphone à partir de ses caractéristiques")
        with st.form('form_simple_tel'):
            col1, col2 = st.columns(2)
            with col1:
                c2_prix = st.number_input("Prix du téléphone (FCFA)", min_value=10000, max_value=2000000, value=150000, key="c2_pr")
                c2_adresse = st.selectbox("Lieu / Adresse", uniques_tel['adresse'], key="c2_adr")
                c2_marque = st.selectbox("Marque du téléphone", uniques_tel['marque'], key="c2_mq")
            with col2:
                c2_dim = st.number_input("dim_ecr (pouces)", min_value=4.0, max_value=7.5, value=6.1, key="c2_d")
                c2_ram = st.number_input("RAM (Go)", min_value=1, max_value=64, value=8, key="c2_r")
                c2_stockage = st.number_input("Stockage (Go)", min_value=8, max_value=1024, value=128, key="c2_s")
            
            submit_tel = st.form_submit_button('Prédire l\'État')

        if submit_tel:
            try:
                # 1. DataFrame dans l'ordre exact du dataset d'entraînement
                input_data = pd.DataFrame({
                    'prix': [c2_prix],
                    'adresse': [c2_adresse],
                    'marque': [c2_marque],
                    'dim_ecr': [c2_dim],
                    'ram': [c2_ram],
                    'stockage': [c2_stockage]
                })

                # 2. Encodage des colonnes catégorielles (encodeurs = dict par colonne)
                input_data['adresse'] = encoder_tel['adresse'].transform(input_data['adresse'])
                input_data['marque'] = encoder_tel['marque'].transform(input_data['marque'])

                # 3. Normalisation puis prédiction
                prediction = model_tel.predict(scaler_tel.transform(input_data))[0]
                etat = encoder_tel['etat'].inverse_transform([prediction])[0]

                st.success(f"État du téléphone prédit : **{etat}**")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

    with tab2:
        st.subheader("Prédiction multiple (CSV) - Téléphones")
        file_tel = st.file_uploader('Upload a csv file (Téléphones)', type=['csv'], key='up_tel')
        if file_tel is not None:
            df_tel = pd.read_csv(file_tel)
            st.dataframe(df_tel.head())
            if st.button('Lancer la prédiction multiple (Téléphones)'):
                st.info("Traitement par lot en cours...")