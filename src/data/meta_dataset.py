import os
import sys
import pandas as pd
from ast import literal_eval
from imdb import Cinemagoer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import re


def get_metadata(file_path = 'data/raw/AllMoviesDetailsCleaned.csv',sep=';'):
    # Read Data #
    meta_data = pd.read_csv(file_path,sep=sep)

    ## Filter rows ##
    meta_data['release_date'] = pd.to_datetime(meta_data['release_date'],errors='coerce')

    #This can be changed to "made_in_us" filter instead of original language
    mask1 = (meta_data['release_date'].dt.year>=1980)&\
            (meta_data['original_language']=='en')&\
            (meta_data['status']=='Released')

    meta_data = meta_data[mask1]

    ## Format data ##
    meta_data['imdb_id'] = pd.to_numeric(meta_data['imdb_id'].str.replace('tt',''),errors='coerce')
    meta_data['budget'] = meta_data['budget'].astype(float)
    meta_data['id'] = meta_data['id'].astype(int)


    ## Filter columns ##
    meta_data_cols = [
                    'title',
                    'id',
                    'imdb_id',
                    'original_language',
                    'original_title',
                    'overview',
                    'production_companies',
                    'budget',
                    'genres',
                    'production_countries',
                    'release_date',
                    'revenue',
                    'runtime',
                    'spoken_languages',
                    'production_companies_number',
                    'production_countries_number',
                     'spoken_languages_number'
                        ]
    meta_data = meta_data[meta_data_cols]

    ## Clean bad overviews ##
    meta_data[meta_data['overview'].apply(lambda x: len(x) if isinstance(x,str) else x)<=2]=np.nan


    ## Drop duplicates and reset indexes ##
    meta_data.drop_duplicates(subset=['imdb_id'],inplace=True)
    meta_data.reset_index(drop=True,inplace=True)
    meta_data.to_csv('data/processed/meta_data_cleaned.csv',index = False)


if __name__ == "__main__":
    get_metadata()


