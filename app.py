from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models/supervised'
TEXT_MODELS_DIR = ROOT / 'models/unsupervised'
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


st.set_page_config(page_title='CVRIE')
part = st.radio('Partie', ['Radiographies', 'Témoignages'], horizontal=True)

if part == 'Radiographies':
    st.title('Radiographies')

    source = st.radio('Images', ['Validation', 'Test'], horizontal=True)
    if source == 'Validation':
        if not (MODELS_DIR / 'validation.csv').is_file():
            st.info('Exécutez le notebook pour enregistrer les modèles et la liste des images de validation.')
            st.stop()
        validation = pd.read_csv(MODELS_DIR / 'validation.csv')
        image_paths = [ROOT / 'archive-2/images/train' / f'{uid}-c.png' for uid in validation['SOPInstanceUID']]
    else:
        image_paths = sorted((ROOT / 'archive-2/images/test').glob('*.png'))
        st.caption('Test : les catégories réelles ne sont pas disponibles.')

    if not image_paths:
        st.info('Aucune image disponible. Pour le test, placez les PNG dans archive-2/images/test/.')
        st.stop()

    index = st.selectbox('Image', range(len(image_paths)), format_func=lambda i: f'Image {i + 1}', key=source)
    size = st.selectbox('Taille', [128, 256, 512])
    name = st.selectbox('Modèle', ['kNN', 'RandomForest'])
    try:
        image = plt.imread(image_paths[index])
    except FileNotFoundError:
        st.error(f'Image introuvable : {image_paths[index].name}')
        st.stop()
    prepared = prepare_image(image, size)
    st.image(prepared, caption=f'{size} × {size}', width=300)
    if source == 'Validation':
        target = str(validation.iloc[index]['Target'])
        st.write('Catégories attendues : ' + ', '.join(CATEGORIES[int(code)] for code in target.split()))

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

else:
    st.title('Témoignages des patients')
    model_path = TEXT_MODELS_DIR / 'text_model.joblib'
    patients_path = TEXT_MODELS_DIR / 'patients.csv'
    if not model_path.is_file() or not patients_path.is_file():
        st.info('Lancez : python Unsupervised_Learning/Learning_model.py')
        st.stop()

    saved = joblib.load(model_path)
    vectorizer = saved['vectorizer']
    model = saved['model']
    patients = pd.read_csv(patients_path)

    group = st.selectbox('Groupe', range(model.n_clusters))
    st.dataframe(patients.loc[patients['cluster'] == group, ['id', 'text']], hide_index=True)

    text = st.text_area('Nouveau témoignage (en anglais)')
    if st.button('Trouver le groupe', disabled=not text.strip()):
        vector = vectorizer.transform([text])
        if vector.nnz == 0:
            st.info('Aucun mot reconnu. Ajoutez une description plus précise.')
        else:
            predicted_group = int(model.predict(vector)[0])
            st.write(f'Groupe proposé : {predicted_group}')
