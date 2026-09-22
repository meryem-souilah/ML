# Air Quality Dashboard 🌍

Application Streamlit qui prédit la qualité de l'air (*Good*, *Moderate*, *Unhealthy*, *Very Unhealthy*) à partir des mesures de polluants et des conditions météo, puis génère des graphiques, une interprétation et un rapport PDF.

## Fonctionnement

1. **Données** : 10 000 mesures de villes du monde (`global_air_quality_data_10000.csv`) : PM2.5, PM10, NO2, SO2, CO, O3, température, humidité, vent.
2. **Cible** : l'indice AQI est calculé pour PM2.5, PM10 et NO2 selon les seuils de l'EPA ; l'AQI global (le maximum) est converti en catégorie.
3. **Modèle** : pipeline scikit-learn (`StandardScaler` + `OneHotEncoder` → `DecisionTreeClassifier`), entraîné sur 80 % des données (split stratifié) et sauvegardé dans `final_model.pkl`.
4. **Application** : l'utilisateur saisit les mesures, obtient la prédiction et trois onglets : graphiques, analyse et export PDF.

## Stack

Python · pandas · NumPy · scikit-learn · Streamlit · Matplotlib · FPDF · joblib

## Lancer en local

```bash
git clone https://github.com/meryem-souilah/ML.git
cd ML
pip install -r requirements.txt
python train_model.py      # optionnel : ré-entraîne et régénère final_model.pkl
streamlit run app.py
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `train_model.py` | Calcul de l'AQI, création de la cible, entraînement et sauvegarde du modèle |
| `app.py` | Tableau de bord Streamlit |
| `final_model.pkl`, `categories.pkl` | Modèle entraîné et catégories utilisées par l'app |

## Auteure

**Meryem Souilah** — [LinkedIn](https://www.linkedin.com/in/souilah-meryem) · [Portfolio](https://meryem-souilah.github.io/My-Portfolio/)
