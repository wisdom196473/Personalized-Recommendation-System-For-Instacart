## Author:

# Yu-Chih (Wisdom) Chen

# Date: 02/15/2024

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import copy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



df = pd.read_csv("baby_formula_df.csv")
df.head()

df_0 = copy.deepcopy(df)
df_0['metadata'] = df_0.apply(lambda x : x['aisle']+' '+x['department']+' '+x['product_name'], axis = 1)
product_details = df_0[['product_name', 'metadata']]

# Content-Based Filtering Recommendation Systems:
def content_based_method(input_words, data):
    # Convert input words to lowercase to ensure case-insensitivity
    input_words = input_words.lower()
    # Ensure metadata is treated in a case-insensitive manner
    data['metadata'] = data['metadata'].str.lower()
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])

    # Transform the input words
    input_vec = vectorizer.transform([input_words])

    # Compute the cosine similarity
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix)

    # Get the indices of the documents that have a similarity score greater than 0
    similar_indices = cosine_sim[0] > 0

    # Retrieve the product names based on the indices
    similar_products = data['product_name'].iloc[similar_indices].tolist()

    # Remove duplicates from the list
    similar_products = list(set(similar_products))

    # If no similar products are found, print a message and return an empty list
    if not similar_products:
        print(f'Product "{input_words}" cannot be found. Please try again.')
        return []

    # Determine the number of items to return
    item_count = min(len(similar_products), 5)

    # Return the specified number of similar products
    return similar_products[:item_count]