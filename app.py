#### IMPORTS
import os
import openai
import random
import spacy
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import base64
from pathlib import Path





#### functions
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

def get_attributes(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    attributes_lists = []
    current_attributes = []
    
    for token in doc:
        if token.pos_ in ['ADJ', 'NOUN'] and not token.is_stop:
            current_attributes.append(token.text)
        
        if token.text.lower() in ['with','and', 'or', ','] or token.i == len(doc) - 1:
            if current_attributes:
                attributes_lists.append(current_attributes)
                current_attributes = []
    
    return attributes_lists

def recommend(attributes,gender):
    num_neighbors = 10001
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine').fit(tfidf_matrix)
    user_input_vector = tfidf_vectorizer.transform([attributes])
    distances, indices = nbrs.kneighbors(user_input_vector)

    user_gender = gender
    relevant_indices = []
    for idx in indices.flatten():
        item_tags = new_df['tags'].iloc[idx]
        if user_gender in item_tags:
            relevant_indices.append(idx)
            if len(relevant_indices) >= 5:
                break
    
    if len(relevant_indices) >= 5:
        relevant_item_indices = relevant_indices[:5]
    else:
        relevant_item_indices = indices.flatten()[:5]

    # best_matches = new_df['productDisplayName'].iloc[relevant_item_indices]

    # print("Top 5 matching items:")
    # for idx, item in enumerate(best_matches, start=1):
    #     print(f"{idx}. {item}")
    # return best_matches
    return relevant_item_indices

#### Pickling and data framing 
outfits_dict=pickle.load(open('fashion.pkl','rb'))
new_df=pd.DataFrame(outfits_dict)
tfidf_matrix=pickle.load(open('tfidf_matrix.pkl','rb'))
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['tags'])

productName=new_df['productDisplayName'].values


#### st commands

st.title('Fashion Outfit Generator')
st.image(Image.open('fashion_img.jpg'))


# if 'user_input' not in st.session_state:
#         st.session_state.user_input = ""

#### user inputs
user_input = st.text_input('Search your outfit here',placeholder="Search here") 
gender='men'

# user_input = st.text_input('Search your outfit here', 'Show me some outfits for diwali like white shirt and black trouser and red watch',placeholder="Search here") 


attributes_lists = get_attributes(user_input)
for attributes in attributes_lists:
    print(f"Your attributes are {attributes}")



# Choose a human-like extra response template
response_templates = [
    "Here is your outfit:",
    "Nice choice! Below are your items:",
    "Amazing collection! Here are your new items:",
    "You're going to love this! Your outfit includes:",
    "Voilà! Behold your stylish ensemble:",
    "Time to impress! Feast your eyes on this outfit:",
    "Drumroll, please! Presenting your fashion-forward look:",
    "Get ready to shine! Here's what you've selected:",
    "Fashion alert! Check out these fabulous pieces:",
    "Prepare to dazzle! Your outfit is ready:",
    "Flawless taste! Here's your curated ensemble:",
    "Presenting your fashion statement:"
]


#### gpt process
# Provide initial user and assistant messages
gpt_input = "how is " + user_input +" within in 20 words"
messages = [
    {"role": "system", "content": "You are a helpful assistant that provides information about clothing attributes."},
    {"role": "user", "content": gpt_input}
]



selected_template = random.choice(response_templates)

content_list = []

gpt = selected_template

if st.button('Recommend'):
    st.balloons()

    new_content = []
    new_content.append(f'<p style="margin-top: 12px;border-radius: 6px;font-style: italic;text-align: right;padding: 3px 3px 3px 9px;background-color: rgb(38, 39, 48);">{user_input}</p>')
    new_content.append(f'<p style="padding: 3px 3px 3px 9px;">{gpt}</p>')

    for attributes in attributes_lists:
        attributes = ' '.join(attributes)
        rd = recommend(attributes,gender)
        # print(f"Your attributes are {attributes}")
        
        recommendations_content = []
        
        for i in (rd):
            recommendations_content.append(f'<div style="display: inline-block;width: calc(20% - 0px);text-align: center;padding: 10px;">')
            recommendations_content.append(f'<p>{new_df["productDisplayName"].iloc[i]}</p>')
            # img_path = f"../project/images/{new_df['id'].iloc[i]}.jpg"
            img_path = f"../code/images/{new_df['id'].iloc[i]}.jpg"
            recommendations_content.append(img_to_html(img_path))
            
            recommendations_content.append(f'<button style="display: inline-block;font-size: 16px;margin-top: 14px;font-weight: bold;text-align: center;text-decoration: none;border-radius: 5px;border: none;background-color: rgb(52 152 219);color: #ffffff;cursor: pointer;transition: background-color 0.3s ease 0s;">View more</button>')
            recommendations_content.append(f'</div>')

        new_content.extend(recommendations_content)

        new_content.append("<hr/>")

    # st.session_state.user_input = ""

    st.session_state.content_list.append(''.join(new_content))

if "content_list" not in st.session_state:
    st.session_state.content_list = []

for content in reversed(st.session_state.content_list):
    st.markdown(content, unsafe_allow_html=True)