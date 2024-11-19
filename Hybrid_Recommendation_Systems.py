## Author:

# Yu-Chih (Wisdom) Chen

# Date: 02/15/2024

import pandas as pd
import copy
import warnings
import sys

sys.path.append('D:\\Wisdom Chen\\University\\University of Chicago\\Winter 2024\\ADSP 31008 Data Mining Principles')

warnings.filterwarnings('ignore')
import Content_Based_Filtering_Recommendation_Systems
import User_Based_Collaborative_Filtering

df = pd.read_csv("baby_formula_df.csv")


df_0 = copy.deepcopy(df)
df_0['metadata'] = df_0.apply(lambda x: x['aisle']+' '+x['department']+' '+x['product_name'], axis = 1)

product_details = df_0[['product_name', 'metadata']]

def ask_user_role():
    while True:
        role = input("Are you a retailer or a customer? ").lower()
        if role in ['retailer', 'customer']:
            return role
        else:
            print("Invalid input. Please enter 'retailer' or 'customer'.")

def hybrid_recommendation_system(df, content_data, num_recommendations=5):
    while True:  # Start of the loop to run the recommendation process
        role = ask_user_role()

        recommendations = []  # Initialize recommendations as an empty list

        if role == 'customer':
            input_words = input("Please enter a product name: ").lower()  # Convert input to lowercase
            recommendations = Content_Based_Filtering_Recommendation_Systems.content_based_method(input_words, content_data)
            if not recommendations:
                continue
        elif role == 'retailer':
            while True:  # This loop will continue until a valid user ID is entered or the user chooses to exit
                try:
                    user_id_input = input("Please enter a user ID: ")
                    user_id = int(user_id_input)
                    recommendations = User_Based_Collaborative_Filtering.collaborative_filtering_user_based(user_id, df, num_recommendations)
                    # If the user_id is not found (function returns False), prompt for the role again
                    if recommendations is False:
                        break  # Exit the inner loop to ask for the role again
                    else:
                        # If valid recommendations are found, exit the loop
                        break
                except ValueError:
                    print("Invalid input. Please enter a numeric user ID.")
                    continue

        # Check if recommendations is a list before attempting to use len()
        if isinstance(recommendations, list) and len(recommendations) > num_recommendations:
            recommendations = recommendations[:num_recommendations]

        # Convert the list of recommendations to a DataFrame and display it
        if recommendations and role == 'customer':
            recommendations_df = pd.DataFrame(recommendations, columns=['Recommended Products'])
        elif recommendations and role == 'retailer':
            recommendations_df = pd.DataFrame(recommendations, columns=['User Purchase History'])
        else:
            continue  # Skip the rest of the loop if no valid recommendations were found

        print(recommendations_df)

        # Ask the user if they want to run the system again
        run_again = input("Do you want to run the system again? (yes/no) ").lower()
        if run_again != 'yes':
            print("Thank you for Using our System")
            break  # Exit the loop if the user does not want to continue

hybrid_recommendation_system(df_0, product_details, num_recommendations=5)