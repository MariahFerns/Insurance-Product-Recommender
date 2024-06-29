# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### Import required libraries

import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime
from feature_engine.outliers import Winsorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ### Data Pre-processing

# +
# Load given dataset
products = pd.read_csv(r'products.csv')
clients = pd.read_csv(r'clients.csv')

# Creating combined dataset of Products and Clients
clients_unpivot = clients.melt(id_vars=['ClientID','join_date','sex','marital_status','birth_year','branch_code','occupation_code','occupation_category_code'],
                               value_vars=['P5DA', 'RIBP', '8NN1','7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
                                               'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3'],
                               var_name='ProductCode',
                               value_name='TakenProduct'
                                )
ProductClients = pd.merge(products,clients_unpivot, on='ProductCode').drop(columns='join_date')

# Data wrangling to transform pipe separated InsuranceType into individual rows

# Split pipe separated Types into individual columns
insurance_columns = ProductClients['InsuranceType'].str.split('|', expand=True )

# Concatenate with ProductClients df 
ProductClients_Ins_1 = pd.concat([ProductClients,insurance_columns], axis=1 ).drop(columns='InsuranceType')

# Unpivot the data
ProductClients_Ins = ProductClients_Ins_1.melt(id_vars=['ProductCode','ProductDescription','ClientID','sex','marital_status','birth_year','branch_code','occupation_code','occupation_category_code','TakenProduct'], 
                                              value_vars=[0, 1, 2], 
                                              value_name='InsuranceType').dropna().drop(columns='variable')
# Remove whitespaces 
ProductClients_Ins.InsuranceType = ProductClients_Ins.InsuranceType.str.strip()

# Deriving age from birth year
ProductClients_Ins['birth_year'] = datetime.now().year - ProductClients_Ins['birth_year']
ProductClients_Ins.rename(columns={'birth_year':'Age'}, inplace=True)

# Treating outliers
capper = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables='Age')
ProductClients_Ins_cap = capper.fit_transform(ProductClients_Ins)

# Create age bins
ProductClients_Ins_cap['Age'] = np.where(ProductClients_Ins_cap.Age<30,'Young', 
                                         np.where(ProductClients_Ins_cap.Age<50,'Middle-Aged',
                                         np.where(ProductClients_Ins_cap.Age<70,'Senior', 'Senior Citizen')))

# Drop occupation_code
ProductClients_Ins_cap.drop(columns='occupation_code', inplace=True)

# Convert ['marital_status'] 'R', 'P' and 'f' into 'Other' and 'U' to 'S'
ProductClients_Ins_cap['marital_status'] = np.where(ProductClients_Ins_cap.marital_status.isin (['R','P' ,'f']),'Other', 
                                                np.where(ProductClients_Ins_cap.marital_status=='U','S',ProductClients_Ins_cap.marital_status)
                                                )

# Total policies at Insurance-product level
# Create a dataframe that gives Insurance type & product wise number of clients who have taken that product

# Filter for clients who have taken 1 or more insurance products
clients_with_ins = ProductClients_Ins_cap[ProductClients_Ins_cap['TakenProduct']>0]
# Num Clients
Insurance_Product_summary = pd.DataFrame(clients_with_ins.groupby(['InsuranceType','ProductDescription'])['ClientID'].nunique()).rename(columns={'ClientID':'Num Clients'})
Insurance_Product_summary = Insurance_Product_summary.reset_index()

# Step 1: Tf-idf Vectorizer
# Using TfidfVectorizer to find most significant insurance type for each product
tf = TfidfVectorizer(stop_words='english')
tf_matrix = tf.fit_transform(products['InsuranceType'])

# Step 2: Cosine similarity
# Calculating similarity between products using cosine similarity
product_similarity = cosine_similarity(tf_matrix) 
# Changing similarity of movie with itself from 1 to 0
np.fill_diagonal(product_similarity, 0)     
# Creating a dataframe
product_similarity_df = pd.DataFrame(product_similarity)


# -

# ### Design the 3 different types of recommendation modules

# +
# 1. Popularity-based recommender system
def popularity_based_recommendation(InsuranceType, minclients, numrecomm):
    g = InsuranceType
    t = minclients
    N = numrecomm

    # Create a subset of products for InsuranceType and minimum clients entered by user
    Productset_Popularity_1 = Insurance_Product_summary[(Insurance_Product_summary['InsuranceType']==g) & (Insurance_Product_summary['Num Clients']>=t)]

    # Add S.No to above subset by sorting on avg. movie rating in descending order    
    Productset_Popularity_1['S.No'] = Productset_Popularity_1.sort_values('Num Clients', ascending=False)\
                        .groupby('InsuranceType')\
                        .cumcount()+1

    # Rearrange column order for ease of understanding
    Productset_Popularity_1.drop(columns='InsuranceType')
    Productset_Popularity_1 = Productset_Popularity_1[['S.No','ProductDescription','Num Clients']]\
                            .rename(columns={'ProductDescription':'Product Description'})
    # Output dataframe that filters for number of recommendations that the user wants
    Productset_Popularity = Productset_Popularity_1[Productset_Popularity_1['S.No']<=N].sort_values('S.No')
    
    return Productset_Popularity


# 2. Content-based recommender system
def content_based_recommendation(product, numrecomm):
    t = product
    N = int(numrecomm)

    global products
    
    # Fetch index of user input product from products dataframe
    index_t = products[products['ProductDescription']==t].index.tolist()  
    index_t = index_t[0] 

    # Adding similarity from product_similarity using index
    products['similarity'] = product_similarity_df.iloc[index_t]

    Productset_Content_1 = products.copy().sort_values('similarity', ascending=False)
    products= products.drop(columns='similarity')

    Productset_Content_1['Sl.No'] = range(1, len(Productset_Content_1)+1)
    Productset_Content_1 = Productset_Content_1[['Sl.No','ProductDescription']].rename(columns={'ProductDescription':'Product Description'})

    # Output dataframe that filters for number of recommendations that the user wants
    Productset_Content = Productset_Content_1[Productset_Content_1['Sl.No']<=N]

    # Removing the default integer index while printing the dataframe
    return Productset_Content


# 3. Collaborative-based recommender system
def collaborative_based_recommendation(clientID, numrecomm, simusers):
    u = clientID
    N = numrecomm
    k = simusers

# Create the dataframe containing list of products taken by the target client
    
    Collab_df = ProductClients[['ClientID','ProductCode','ProductDescription','TakenProduct']]
    
    client_taken = Collab_df[(Collab_df['ClientID']==u) & (Collab_df['TakenProduct']>0)]
    # client_taken.head()
    
    # Find peers who have taken the same products as the target client 
    peers = Collab_df[(Collab_df['ProductCode'].isin(client_taken['ProductCode'])) & (Collab_df['TakenProduct']>0)]      
    
    # Excluding target client from peer list
    peers = peers[peers['ClientID'] != u]
    peers.head()
    # print('There are ',peers['ClientID'].nunique(), ' clients who have taken the same products as the target user')
        
    # Grouping by ClientID to create sub dataframes for each client
    # This will help in quantifying the number of products that are common between each peer and the target client
    # Assumption is: Higher the number of common products taken, more similar are the clients 
    
    peers_grp = peers.groupby('ClientID')
    
    # Create Pearson Coefficient of Correlation to find clients similar to the target client based on products taken
    pearson_coef={}
    
    for name, peer in peers_grp:
    
        # Fetch only those target client products that the peer has taken
        client_peer_taken = client_taken[client_taken['ProductCode'].isin(peer['ProductCode'])]
        
        # Sort by product code
        client_peer_taken = client_peer_taken.sort_values('ProductCode')
        peer = peer.sort_values('ProductCode')
    
        # Add TakenProduct to a list
        client_taken_list = client_peer_taken['TakenProduct'].tolist()
        peer_taken_list = peer['TakenProduct'].tolist()
    
        # Calculate Pearson's coefficient
        if (len(client_taken_list)>1) & (len(peer_taken_list)>1):
            pearson_coef[name] = stats.pearsonr(client_taken_list, peer_taken_list).statistic
            
    # Convert dict to dataframe
    pearson_coef_df = pd.DataFrame.from_dict(pearson_coef, orient='index').fillna(0).reset_index().rename(columns={'index':'ClientID', 0:'SimilarityCoef'})
    # pearson_coef_df['SimilarityCoef'].value_counts()
    
    # Filter the above to get list of "K" similar clients
    pearson_coef_df = pearson_coef_df.sort_values('SimilarityCoef', ascending = False)[:k]
    
    # Fetching taken products for list of similar peers
    similar_peers = Collab_df[(Collab_df['ClientID'].isin(pearson_coef_df['ClientID'])) & (Collab_df['TakenProduct']>0)]
    
    # Exclude products that are already taken by the target client
    similar_peers = similar_peers[~similar_peers['ProductCode'].isin(client_taken['ProductCode'])]
    
    # Adding Similarity Coefficient
    similar_peers = similar_peers.merge(pearson_coef_df, on='ClientID')
    
    # The similar peers may have taken common products. In order to determine which are the top products to recommend,
    # we take weighted average of TakenProduct using SimilarityCoef as the weight
    
    similar_peers['wt_takenproduct'] = similar_peers['TakenProduct']*similar_peers['SimilarityCoef']

    
    # For each product, calculate the weighted average TakenProduct across all similar clients
    peer_recommended_products = pd.DataFrame(similar_peers.groupby(['ProductDescription'])[['wt_takenproduct','SimilarityCoef']].sum()).reset_index().rename(columns={'wt_takenproduct':'WeightedSum','SimilarityCoef':'SumOfWeights'})
    peer_recommended_products['WeightedAvgScore'] = peer_recommended_products['WeightedSum']/ peer_recommended_products['SumOfWeights']
    
    # Sorting in descending order of weighted average TakenProduct
    peer_recommended_products = peer_recommended_products.sort_values('WeightedAvgScore', ascending=False)
    peer_recommended_products['S.No'] = range(1,len(peer_recommended_products)+1)
    
    # Filter for top N peer recommended products 
    Productset_Collaborative = peer_recommended_products[peer_recommended_products['S.No']<=N]
    Productset_Collaborative = Productset_Collaborative[['S.No','ProductDescription']].rename(columns={'ProductDescription':'Product Description'})

    return Productset_Collaborative
# -

# ## Creating a GUI interface using Python library Streamlit 

# +
import streamlit as st

st.set_page_config(page_title='🔖Insurance Product Recommender')

st.header('Select Recommender type:')
option = st.radio('Select Recommender type',('Popularity-based', 'Content-based', 'Collaborative'),label_visibility='hidden')
st.title(f'🔖{option} Recommender')

if (option=='Popularity-based'):
    
    # Take inputs from the user
    insurance_type = st.selectbox('Enter Insurance Type:', ProductClients_Ins['InsuranceType'].unique().tolist())
    min_clients = st.text_input('Enter minimum clients:')
    num_recommendations = st.text_input('Enter number of recommendations you would like to see:')

    # Recommend button - enabled only if all inputs are entered by the user
    recommend = st.button('Recommend', disabled = not(insurance_type and min_clients and num_recommendations))

    response = pd.DataFrame()
    if (recommend):
        with st.spinner('Processing...'):
            # Call the Popularity-based recommender
            response = popularity_based_recommendation(insurance_type, int(min_clients), int(num_recommendations))
    if len(response):
        st.dataframe(response, hide_index=True)
        
elif (option=='Content-based'):
    # Take inputs from the user
    product = st.selectbox('Select the reference product based on which recommendations are to be made:', products['ProductDescription'].unique().tolist())
    num_recommendations = st.text_input('Enter number of recommendations you would like to see:')

    # Recommend button - enabled only if all inputs are entered by the user
    recommend = st.button('Recommend', disabled = not(product and num_recommendations))

    response = pd.DataFrame()
    if (recommend):
        with st.spinner('Processing...'):
            # Call the Contentana-based recommender
            response = content_based_recommendation(product, int(num_recommendations))
    if len(response):
        st.dataframe(response, hide_index=True)

else:
    # Take inputs from the user
    client_id = st.selectbox('Enter target Client ID:', clients['ClientID'].unique().tolist())
    num_recommendations = st.text_input('Enter number of recommendations you would like to see:')
    sim_clients = st.text_input('Enter the threshold for similar clients based on whom recommendations are to be made:')

    # Recommend button - enabled only if all inputs are entered by the user
    recommend = st.button('Recommend', disabled = not(client_id and num_recommendations and sim_clients))

    response = pd.DataFrame()
    if (recommend):
        with st.spinner('Processing...'):
            # Call the Collaborative-based recommender
            response = collaborative_based_recommendation(client_id, int(num_recommendations), int(sim_clients))
    if len(response):
        st.dataframe(response, hide_index=True)
