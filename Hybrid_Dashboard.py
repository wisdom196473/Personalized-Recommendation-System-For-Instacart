import os
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import Content_Based_Filtering_Recommendation_Systems
import User_Based_Collaborative_Filtering
from datasets import load_dataset

# Database setup
DATABASE = ':memory:'

def init_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS customer_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            recommendation_1 TEXT,
            recommendation_2 TEXT,
            recommendation_3 TEXT,
            recommendation_4 TEXT,
            recommendation_5 TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS retailer_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            recommendation_1 TEXT,
            recommendation_2 TEXT,
            recommendation_3 TEXT,
            recommendation_4 TEXT,
            recommendation_5 TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    return conn

def save_recommendations(conn, role, product_name, recommendations):
    table_name = f"{role}_recommendations"
    recommendations = recommendations + [None]*(5-len(recommendations))
    conn.execute(f'''
        INSERT INTO {table_name} (product_name, recommendation_1, recommendation_2, recommendation_3, recommendation_4, recommendation_5)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (product_name, *recommendations[:5]))
    conn.commit()

def display_recommendation_history(conn):
    # Fetch and display customer recommendations
    st.subheader('Customer Recommendations')
    customer_data = conn.execute('SELECT * FROM customer_recommendations').fetchall()
    if customer_data:
        customer_df = pd.DataFrame(customer_data,
                                   columns=['ID', 'Product Name', 'Recommendation 1', 'Recommendation 2',
                                            'Recommendation 3', 'Recommendation 4', 'Recommendation 5',
                                            'Timestamp'])
        # Transform the DataFrame
        customer_df_long = customer_df.melt(id_vars=['ID', 'Product Name', 'Timestamp'],
                                            value_vars=['Recommendation 1', 'Recommendation 2', 'Recommendation 3',
                                                        'Recommendation 4', 'Recommendation 5'],
                                            var_name='Recommendation Rank', value_name='Recommendation')
        customer_df_long = customer_df_long.dropna(subset=['Recommendation'])  # Remove rows with no recommendation

        # Convert DataFrame to CSV for download
        csv = convert_df(customer_df_long)
        st.download_button(
            label="Download Customer Recommendations as CSV",
            data=csv,
            file_name='customer_recommendations.csv',
            mime='text/csv',
        )

        # Group by Product Name and Timestamp to ensure they appear only once
        customer_df_grouped = customer_df_long.groupby(['Product Name', 'Timestamp'])
        for (product_name, timestamp), group in customer_df_grouped:
            st.write(f"**Product Name:** {product_name}")
            st.write(f"**Timestamp:** {timestamp}")
            st.table(group[['Recommendation Rank', 'Recommendation']])
    else:
        st.write("No customer recommendations found.")

    # Fetch and display retailer recommendations
    st.subheader('Retailer Recommendations')
    retailer_data = conn.execute('SELECT * FROM retailer_recommendations').fetchall()
    if retailer_data:
        retailer_df = pd.DataFrame(retailer_data,
                                   columns=['ID', 'Product Name', 'Recommendation 1', 'Recommendation 2',
                                            'Recommendation 3', 'Recommendation 4', 'Recommendation 5',
                                            'Timestamp'])
        # Transform the DataFrame
        retailer_df_long = retailer_df.melt(id_vars=['ID', 'Product Name', 'Timestamp'],
                                            value_vars=['Recommendation 1', 'Recommendation 2', 'Recommendation 3',
                                                        'Recommendation 4', 'Recommendation 5'],
                                            var_name='Recommendation Rank', value_name='Recommendation')
        retailer_df_long = retailer_df_long.dropna(subset=['Recommendation'])  # Remove rows with no recommendation

        # Convert DataFrame to CSV for download
        csv = convert_df(retailer_df_long)
        st.download_button(
            label="Download Retailer Recommendations as CSV",
            data=csv,
            file_name='retailer_recommendations.csv',
            mime='text/csv',
        )

        # Group by Product Name and Timestamp to ensure they appear only once
        retailer_df_grouped = retailer_df_long.groupby(['Product Name', 'Timestamp'])
        for (product_name, timestamp), group in retailer_df_grouped:
            st.write(f"**Product Name:** {product_name}")
            st.write(f"**Timestamp:** {timestamp}")
            st.table(group[['Recommendation Rank', 'Recommendation']])
    else:
        st.write("No retailer recommendations found.")

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def clear_history(conn):
    conn.execute('DELETE FROM customer_recommendations')
    conn.execute('DELETE FROM retailer_recommendations')
    conn.commit()

# Initialize the Streamlit application
st.title("Welcome to our Powerful System")

# Display the current date and time
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
st.write("**Current Time:**", current_time)

# Display the author names in bold, each on a new line, followed by their titles
st.markdown("**Author Names:**")
st.markdown("**Wisdom Chen** (Data Scientist)")

# Load your dataset
dataset = load_dataset("chen196473/Instacart_Baby_Formula")
df = dataset['train'].to_pandas()
df['metadata'] = df.apply(lambda x: x['aisle'] + ' ' + x['department'] + ' ' + x['product_name'], axis=1)
product_details = df[['product_name', 'metadata']]

# Initialize the database connection
conn = init_db()

# Define the customer dashboard function
def customer_dashboard():
    st.header('Customer Dashboard')
    product_input = st.text_input('Enter a product name')
    submit_val = st.button('Submit')

    if submit_val:
        recommendations = Content_Based_Filtering_Recommendation_Systems.content_based_method(product_input, product_details)
        if recommendations:
            save_recommendations(conn, 'customer', product_input, recommendations)
            recommendations_df = pd.DataFrame(recommendations, columns=['Recommended Products'])
            recommendations_df.index += 1  # To show rank starting from 1
            st.table(recommendations_df)
        else:
            st.write("No recommendations found.")

# Define the retailer dashboard function
def retailer_dashboard():
    st.header('Retailer Dashboard')
    user_id_input = st.number_input('Enter a user ID', step=1, format="%d")
    submit_user_id = st.button('Submit ID')

    if submit_user_id:
        recommendations = User_Based_Collaborative_Filtering.collaborative_filtering_user_based(user_id_input, df, 5)
        if recommendations:
            save_recommendations(conn, 'retailer', str(user_id_input), recommendations)
            recommendations_df = pd.DataFrame(recommendations, columns=['User Purchase History'])
            recommendations_df.index += 1  # To show rank starting from 1
            st.table(recommendations_df)
        else:
            st.write("No recommendations found or user ID not found.")

# Use a selectbox for tab-like functionality
tab = st.selectbox('Select a role', ['Customer', 'Retailer'])

# Display the appropriate dashboard based on the selected tab
if tab == 'Customer':
    customer_dashboard()
elif tab == 'Retailer':
    retailer_dashboard()

# Add a new section in Streamlit for viewing history
st.sidebar.header('Historical Recommendations')
if st.sidebar.button('View Recommendation History'):
    display_recommendation_history(conn)

# Add a button to clean history
st.sidebar.header('Manage Historical Recommendations')
if st.sidebar.button('Clean Database'):
    clear_history(conn)
    st.sidebar.write('History cleaned successfully.')

# Close the connection when the app is done
st.on_script_run_end(lambda: conn.close())
