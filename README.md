# 🔰Insurance Recommendation System [![Open app in Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://appuct-recommender---insurance-6yuzsshinnxnpvenrfun8j.streamlit.app/)

## Introduction
This project aims to develop a recommendation platform for a leading insurance provider. The platform includes three types of recommender systems: popularity-based, content-based, and collaborative filtering. The system is designed to help users find the best insurance products tailored to their needs and preferences.

## Objectives
1. **Popularity-based Recommender System**:
   - Recommend top N products within a specific insurance type (g).
   - Consider products with a minimum rating threshold (t).
   - Order products by ratings in descending order, ensuring each product has at least (t) reviews.

2. **Content-based Recommender System**:
   - Recommend top N products based on similar product types.

3. **Collaborative-based Recommender System**:
   - Recommend top N products based on "K" similar users for a target user "u".

## Methodology
### Data Preparation
1. **Import Libraries and Load Dataset**:
   - Import necessary Python libraries for data manipulation, visualization, and modeling.
   - Load the dataset containing insurance product information and user reviews.

### Exploratory Data Analysis (EDA)
1. **Understanding Feature Distribution**:
   - Analyze the distribution of various features in the dataset.
2. **Unique Users and Products**:
   - Identify the number of unique users and products in the dataset.
3. **Average Rating and Total Products by Insurance Type**:
   - Calculate the average rating and total number of products for each insurance type.
4. **Unique Insurances**:
   - Determine the unique insurance types considered in the dataset.

### Recommendation Modules
1. **Popularity-based Recommender System**:
   - Filter products based on the insurance type and minimum rating threshold.
   - Sort products by ratings in descending order.
   - Recommend the top N products.

2. **Content-based Recommender System**:
   - Use product features to find similar products.
   - Recommend top N products based on similarity to a given product type.

3. **Collaborative-based Recommender System**:
   - Identify similar users based on their ratings and preferences.
   - Recommend top N products for a target user based on the preferences of similar users.

### GUI Interface
- Create a user-friendly interface using Streamlit to interact with the recommendation modules.
- Allow users to input parameters such as insurance type, minimum rating threshold, number of recommendations, product type, and target user.

## Folder Structure
- `data/`: Contains the dataset used for the project.
- `notebooks/`: Jupyter notebooks for data analysis, modeling, and testing.
- `src/`: Python scripts for the recommendation modules and Streamlit app.
- `api/`: Api code (if any).
- `models/`: Trained models and saved results.
- `docs/`: Output files including recommendation results and evaluation metrics.

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MariahFerns/Product-Recommender---Insurance.git
   cd Product-Recommender---Insurance

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt

3. Run Jupyter Notebooks:
   Navigate to the notebooks/ folder and open the notebooks to explore data analysis and model development.

4. Run Streamlit Application:
   ```bash
   streamlit run src/app.py
