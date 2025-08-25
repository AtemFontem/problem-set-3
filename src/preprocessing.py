'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import ast

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''

    model_pred_df = pd.read_csv("data/prediction_model_03.csv")
    genres_df = pd.read_csv("data/genres.csv")
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    genre_list = set()
    genre_true_counts = {}
    genre_tp_counts = {}
    genre_fp_counts = {}

    for idx, row in model_pred_df.iterrows():

        s = str(row["actual genres"]).strip()
        s = s.strip("[]")
        parts = [] if (s == "" or s.lower() == "nan") else s.split(",")
        actual = []

        for p in parts:
            g = p.strip().strip("'").strip('"') # trim spaces and quotes

            if g:
                actual.append(g)

        pred = str(row["predicted"]).strip()

        #build genre universe
        for g in actual:
            genre_list.add(g)
        genre_list.add(pred)

        #true counts- each actual label counts once for that row
        for g in actual:
            genre_true_counts[g] = genre_true_counts.get(g, 0) + 1

        if pred in actual:
            genre_tp_counts[pred] = genre_tp_counts.get(pred, 0) + 1
        else:
            genre_fp_counts[pred] = genre_fp_counts.get(pred, 0) + 1

    if "genre" in genres_df.columns:
        for g in genres_df["genre"].astype(str).tolist():
            genre_list.add(g)


    for g in genre_list:
        genre_true_counts.setdefault(g, 0)
        genre_tp_counts.setdefault(g, 0)
        genre_fp_counts.setdefault(g, 0)
    
    return sorted(genre_list), genre_true_counts, genre_tp_counts, genre_fp_counts

