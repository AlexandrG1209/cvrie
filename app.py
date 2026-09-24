from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / 'models'
CATEGORIES = [
    'Abdomen', 'Cheville', 'Colonne cervicale', 'Thorax', 'Clavicules',
    'Coude', 'Pieds', 'Doigt', 'Avant-bras', 'Main', 'Hanche', 'Genou',
    'Jambe', 'Colonne lombaire', 'Autres', 'Bassin', 'Épaule', 'Sinus',
    'Crâne', 'Cuisse', 'Colonne thoracique', 'Poignet'
]


def prepare_image(image, size):
    height, width = image.shape
    top = (512 - height) // 2
    left = (512 - width) // 2
    square = np.zeros((512, 512), dtype='float32')
    square[top:top + height, left:left + width] = image
    while square.shape[0] > size:
        half = square.shape[0] // 2
        square = square.reshape(half, 2, half, 2).mean(axis=(1, 3))
    return square


st.set_page_config(page_title='CVRIE — Radiographies')
st.title('Radiographies')

if not (MODELS_DIR / 'validation.csv').is_file():
    st.info('Exécutez le notebook pour enregistrer les modèles et la liste des images de validation.')
    st.stop()

validation = pd.read_csv(MODELS_DIR / 'validation.csv')
index = st.selectbox('Image de validation', validation.index, format_func=lambda i: f'Image {i + 1}')
size = st.selectbox('Taille', [128, 256, 512])
name = st.selectbox('Modèle', ['kNN', 'RandomForest'])
row = validation.iloc[index]

try:
    image = plt.imread(ROOT / 'archive-2/images/train' / f"{row['SOPInstanceUID']}-c.png")
except FileNotFoundError:
    st.error('Téléchargez les images dans archive-2/images/train/.')
    st.stop()
prepared = prepare_image(image, size)
st.image(prepared, caption=f'{size} × {size}', width=300)
st.write('Catégories attendues : ' + ', '.join(CATEGORIES[int(code)] for code in row['Target'].split()))

if st.button('Prédire', type='primary'):
    model_path = MODELS_DIR / f'{name}_{size}.joblib'
    if not model_path.is_file():
        st.error('Exécutez le notebook pour enregistrer ce modèle.')
        st.stop()
    with st.spinner('Prédiction…'):
        model = joblib.load(model_path, mmap_mode='r')
        prediction = model.predict(prepared.reshape(1, -1))[0]
    categories = ', '.join(CATEGORIES[code] for code in np.flatnonzero(prediction))
    st.write('Prédiction : ' + (categories or 'Aucune catégorie'))
