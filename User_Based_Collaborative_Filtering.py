## Author:

# Yu-Chih (Wisdom) Chen

# Date: 02/15/2024

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


# User-Based Collaborative Filtering:
def collaborative_filtering_user_based(user_id, df, num_recommendations=5):
    # Convert product names to lowercase to ensure case-insensitivity
    df['product_name'] = df['product_name'].str.lower()
    
    user_item_matrix = df.pivot_table(index='user_id', columns='product_name', values='reordered').fillna(0)

    # If the user_id does not exist in the user_item_matrix, return False
    if user_id not in user_item_matrix.index:
        print(f'User ID "{user_id}" not found. Please try again.')
        return False  # Indicate that the user ID was not found

    user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

    # Compute the similarity matrix
    similarity_matrix = cosine_similarity(user_item_matrix_sparse)

    # Get the index of the target user
    user_idx = list(user_item_matrix.index).index(user_id)

    # Get the most similar users to the target user
    similar_users = np.argsort(-similarity_matrix[user_idx])[1:num_recommendations+1]

    # Recommend items
    recommended_items = []
    for similar_user in similar_users:
        # Find items that this user liked
        liked_items = user_item_matrix.iloc[similar_user].index[user_item_matrix.iloc[similar_user].gt(0)]
        recommended_items.extend(liked_items)

    # Remove duplicates and return
    return list(set(recommended_items))[:num_recommendations]