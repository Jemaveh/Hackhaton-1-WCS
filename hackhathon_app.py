import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
from pyfood.utils import Shelf
shelf = Shelf(region='United Kingdom')

df_recettes = pd.read_csv('df3.csv', sep = ',', low_memory=False)


def onglet0():
    st.markdown("<h1 style='text-align: center;'>Noël à l'Ouest</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Analyses sur les recettes</h3>", unsafe_allow_html=True)
    df_recettes['Animal_Products'] = ~(df_recettes['Vegan'] | df_recettes['Vege'])
    vegan_counts = df_recettes['Vegan'].sum()
    vege_counts = df_recettes['Vege'].sum()
    animal_counts = df_recettes['Animal_Products'].sum()
    categories = ['Vegan', 'Vege', 'Animal Products']
    counts = [vegan_counts, vege_counts, animal_counts]
    
    st.markdown("<h3 style='text-align: center;'>Distribution des recettes par catégories</h3>", unsafe_allow_html=True)
    st.bar_chart(pd.DataFrame({'Catégories': categories,
                               'Nombre de recettes': counts}).set_index('Catégories'), color = '#c4893f')
    
    oven_counts = df_recettes['Oven'].value_counts()
    labels = oven_counts.index
    sizes = oven_counts.values

    st.markdown("<h3 style='text-align: center;'>Utilisation du four dans les recettes</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'color' : 'white'})
    fig.patch.set_facecolor('#02211e')
    st.pyplot(fig)

    
    st.markdown("<h3 style='text-align: center;'>Ingrédient les plus présents dans les recettes</h3>", unsafe_allow_html=True)
    url1 = 'https://drive.google.com/file/d/1-wM1BZoRO5ZnQVNvuJI271CFZjRcKTiF/view?usp=sharing'
    file_id = url1.split('/')[-2]
    new_url1 = f'https://drive.google.com/uc?export=view&id={file_id}'
    st.image(new_url1)

    


    
    
    
    
def onglet1():
    st.markdown("<h1 style='text-align: center;'>Noël à l'Ouest</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Recommandation des recettes</h3>", unsafe_allow_html=True)
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np

    selected_categories = st.selectbox("Selectionner votre catégorie de recettes",
                                       ('Toutes les recettes', 'Recettes végétariennes', 'Recettes vegan', 'Recettes sans four'))

    if selected_categories == 'Toutes les recettes':
        df = df_recettes.copy()
        df.reset_index(inplace = True, drop = True)
        st.write('Recherche dans toutes les recettes')
    elif selected_categories == 'Recettes végétariennes' :
        df = df_recettes[df_recettes['Vege_str'] == 'Oui']
        df.reset_index(inplace = True, drop = True)
        st.write('Recherche dans les recettes végétariennes')
    elif selected_categories == 'Recettes vegan' :
        df = df_recettes[df_recettes['Vegan_str'] == 'Oui']
        df.reset_index(inplace = True, drop = True)
        st.write('Recherche dans les recettes vegan')
    elif selected_categories == 'Recettes sans four': 
        df = df_recettes[df_recettes['Oven_str'] == 'Non']
        df.reset_index(inplace = True, drop = True)
        st.write('Recherche dans les recettes sans four')



    
    # Encodage pour  Names
    mlb_names = MultiLabelBinarizer()
    names_encoded = mlb_names.fit_transform(df['Name'])

    # Encodage pour Common_Ingredients_New
    mlb_ing = MultiLabelBinarizer()
    ing_encoded = mlb_ing.fit_transform(df['Common_Ingredients_New'])

    # Encodage pour Vegan
    mlb_vegan = MultiLabelBinarizer()
    vegan_encoded = mlb_vegan.fit_transform(df['Vegan_str'])

    # Encodage pour Vege
    mlb_vege = MultiLabelBinarizer()
    vege_encoded = mlb_vege.fit_transform(df['Vege_str'])
    
    # Concaténation des caractéristiques encodées pour former la matrice de données
    features = np.concatenate((names_encoded, ing_encoded, vegan_encoded, vege_encoded), axis=1)

    # Entraînement du modèle Nearest Neighbors
    nn2 = NearestNeighbors(n_neighbors=4, metric='cosine')
    nn2.fit(features)
    
    def get_recommendations_by_ing(ingredients):

        # Filtrer les recettes qui contiennent la liste d'ingrédients
        selected_indices = []
        for ing in ingredients:
            ing_indices = df[df['Common_Ingredients_New'].apply(lambda x: ing in x)].index.tolist()
            selected_indices.extend(ing_indices)

        selected_recipes = df.loc[selected_indices]

        #Encoder les caractéristiques des recettes
        name_encoded = mlb_names.transform(selected_recipes['Name'])
        ing_encoded = mlb_ing.transform(selected_recipes['Common_Ingredients_New'])
        vegan_encoded = mlb_vegan.transform(selected_recipes['Vegan_str'])
        vege_encoded = mlb_vege.transform(selected_recipes['Vege_str'])

        # Concaténer les caractéristiques
        selected_features = np.hstack((name_encoded, ing_encoded, vegan_encoded, vege_encoded))

        # Rechercher le voisin le plus proche des recettes trouvés
        neighbors = nn2.kneighbors(selected_features)
        recommended_indices = neighbors[1][0]

        #Lister des informations sur les recettes recommandées
        recommended_recipes = []
        for idx in recommended_indices:
            recipe_info = {
                'Name': df.loc[idx]['Name'],
                'Description': df.loc[idx]['Description'],
                'url' : df.loc[idx]['url'],
                'Type': ''
            }
            if df.loc[idx]['Vegan'] == True:
                recipe_info['Type'] = 'Recette vegan'
            elif df.loc[idx]['Vege'] == True:
                recipe_info['Type'] = 'Recette végétarienne'
            else :
                recipe_info['Type'] = "Recette contenant des ingrédients d'origine animale "

            recommended_recipes.append(recipe_info)

        return recommended_recipes


    ing = st.text_input("Entrez 3 ingrédients (en anglais) séparés par une virgule:")
    st.markdown("Exemple : beef, tomato, onion")
    
    #Entrée de l'utilisateur
    
    user_ingredients = ing.replace(' ','').split(',')  # Liste d'ingrédients entrée par l'utilisateur
    if ing == '' : 
        st.write('')
    else : 
        recommended_recipes_by_ing = get_recommendations_by_ing(user_ingredients)
        #Phrase d'affichage
        st.write("")
        st.write("Si vous avez ces ingrédients {}, vous pourriez aimer les recettes suivantes:".format(user_ingredients))
        for recipe in recommended_recipes_by_ing:
            st.write("Nom:", recipe['Name'])
            st.write("Description:", recipe['Description'])
            st.write("Type:", recipe['Type'])
            st.write("URL de la recette:", recipe['url'])
            st.write('----------------------------------------------')



def onglet3():
    st.markdown("<h1 style='text-align: center;'>Accueil</h1>", unsafe_allow_html=True)
    url = 'https://drive.google.com/uc?export=view&id=1zi-NIwA5H0Y6MhembsA7UTMGX90h2jQn'
    st.image(url, width=900)
    url4 = 'https://drive.google.com/uc?export=view&id=17CiRhDR9akT8kTkr8d_tAcbQvavDtQCm'
    st.image(url4, width=900)
    url5 = 'https://drive.google.com/uc?export=view&id=1Mrjx1I3CQqjSqadEx5r8qamcVBuZLgpx'
    st.image(url5, width=900)
    url6 = 'https://drive.google.com/uc?export=view&id=15UWYpBxoNc26DsGxeCxP1B_TWPXqRV1f'
    st.image(url6, width=900)
    url7 = 'https://drive.google.com/uc?export=view&id=1V7MMwhaphXUHZcBnoviVo9vH1q1RqtoM'
    st.image(url7, width=900)
    url9 = 'https://drive.google.com/uc?export=view&id=1YBQ62tG4Iu7x-YgMixt88PaDJL2b8y6j'
    st.image(url9, width=900)
    url10 = 'https://drive.google.com/uc?export=view&id=1BX7b3Pk1Cb5LoZmTCei0qmvDu6FNsrXC'
    st.image(url10, width=900)
    url8 = 'https://drive.google.com/uc?export=view&id=1bKpy0YkFLY-DCuwDZFNBBLmY-8HUtN7C'
    st.image(url8, width=900)
    #st.write(df_recettes)

def main():
    st.sidebar.title("Navigation")
    onglet_selectionne = st.sidebar.selectbox("Sélectionner un onglet", ["Accueil", "Analyse des recettes",
                                                                         "Recommandations de recettes"])

    if onglet_selectionne == "Analyse des recettes":
        onglet0()
    elif onglet_selectionne == "Recommandations de recettes":
        onglet1()
    else:
        onglet3()

if __name__ == "__main__":
    main()
