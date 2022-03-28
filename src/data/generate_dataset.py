import pandas as pd
from decimal import Decimal
import re
import numpy as np
import os
import argparse
import requests
from joblib import Parallel, delayed

USE_COLUMNS = ['imdb_id', 'runtimeMinutes', 'genres', 'cast', 'averageRating',
       'numVotes', 'title', 'original_title', 'overview', 'release_date',
       'poster_link','revenue_worldwide_BOM', 'director', 'countries', 'country codes',
       'language codes', 'languages', 'Budget', 'cover url', 'production companies']

processed_meta_data = pd.read_csv('data/processed/meta_data_cleaned.csv')


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


def get_imdb_dump_data(imdb_ids_list = processed_meta_data['imdb_id'].unique(),input_dir ='data/raw/oficial_imdb_dataset'):

    
    # READ DATA #
    names = pd.read_csv(os.path.join(input_dir,'name.basics.tsv'),
                        sep='\t',
                    usecols = ['nconst','primaryName'])

    titles = pd.read_csv(os.path.join(input_dir,'title.basics.tsv'),
                        sep='\t',
                        usecols = ['tconst','titleType','genres','runtimeMinutes']
                        )
                        
    principals = pd.read_csv(os.path.join(input_dir,'title.principals.tsv'),
                            sep='\t')


    ratings = pd.read_csv(os.path.join(input_dir,'title.ratings.tsv'),
                            sep='\t')
    



    # TITLES #
    titles['tconst'] = titles['tconst'].str.replace('tt','').astype(float)
    titles.rename(columns = {'tconst':'imdb_id'},inplace=True)

    titles = titles[titles['imdb_id'].isin(imdb_ids_list)]
    titles = titles[titles['titleType']=='movie']

    titles['genres'] = titles['genres'].str.lower().str.replace(',','|')




    # PRINCIPALS #
    principals['tconst'] = principals['tconst'].str.replace('tt','').astype(float)
    principals.rename(columns = {'tconst':'imdb_id'},inplace=True)

    principals = principals[principals['imdb_id'].isin(imdb_ids_list)]
    principals = principals.merge(names,on='nconst',how='left')
    principals.replace({'actor':'cast',
                        'actress':'cast',
                        'self':'cast'},inplace=True)

    ## Sort by credits order
    principals.sort_values(by = ['imdb_id','category','ordering'],ascending=True,inplace=True)

    ## Create grouped version of principals
    top_n = 3
    principals['primaryName'] = principals['primaryName'].fillna('')

    grouped_principals = principals\
                        .groupby(['imdb_id','category'])['primaryName']\
                        .apply(lambda x: '|'.join(x.iloc[:top_n].tolist() ))\
                        .reset_index()

    grouped_principals = grouped_principals.pivot(columns = 'category',
                                                index = 'imdb_id',
                                                values='primaryName')\
                        .reset_index()\
                        .rename_axis(None, axis=1)

    # filter columns and normalize 
    cols = ['imdb_id','director','cast']
    grouped_principals = grouped_principals[cols]

    categ = grouped_principals.columns[grouped_principals.dtypes==object]
    grouped_principals[categ] = grouped_principals[categ].apply(lambda x: x.str.lower().str.strip(),axis=1)




    # RATINGS #
    ratings['tconst'] = ratings['tconst'].str.replace('tt','').astype(float)
    ratings.rename(columns = {'tconst':'imdb_id'},inplace=True)




    # MERGE #
    titles = titles.merge(grouped_principals,how='left',on='imdb_id')
    titles = titles.merge(ratings,how='left',on='imdb_id')
    titles.to_csv('data/processed/imdb.csv',index=False)

def merge_all(use_columns=USE_COLUMNS):
    # Read data
    box_office_mojo_data_v2 = pd.read_csv('data/processed/box_office_mojo_data_v2.csv')
    processed_meta_data = pd.read_csv('data/processed/meta_data_cleaned.csv')
    titles = pd.read_csv('data/processed/imdb.csv')
    imdb_api_data = pd.read_csv('data/processed/imdb_api_data.csv') 
    poster_ratings = pd.read_csv('data/processed/posters_links.csv') #TODO: This is unnecessary because i get it from API


    #Rename Columns
    box_office_mojo_data_v2.rename(columns = {i:f'revenue_{i.lower()}_BOM'for i in box_office_mojo_data_v2.columns[:3]},
                                inplace=True)

    # MERGE #
    df = titles.drop(['director'],axis=1).merge(processed_meta_data.drop(['genres'],axis=1),
                    how = 'left',
                    on='imdb_id')

    df = df.merge(poster_ratings,
                how = 'left',
                on='imdb_id')

    df = df.merge(box_office_mojo_data_v2,
                how='left',
                on='imdb_id')

    df[['imdb_id']].to_csv('data/processed/filtered_id_list.csv',index=False)
    df = df.dropna(subset=['revenue_worldwide_BOM'])


    #Final language filter
    df = df[df['spoken_languages']=='English'] #TODO: This could have been done earlier
    df = df.reset_index(drop=True).sort_values(by='imdb_id')


    df = df.merge(imdb_api_data,how='left',on='imdb_id')
    df['Budget'] = df['Budget'].apply(lambda x: float(Decimal(re.sub(r'[^\d.]', '', x))) \
                   if isinstance(x,str) else np.nan )\
        .replace(0,np.nan)

    #Filter columns
    df = df[use_columns]

    df.to_csv('data/processed/df.csv',index=False)

def download_image_url(url,file_path):
    
    if not isinstance(url,str):
        return 'ERROR'

    response = requests.get(url)
    img = response.content

    if response.status_code==200:
        with open(file_path, 'wb') as handler:
            handler.write(img)
        return 'OK'

    return 'ERROR'

def download_poster(row,data_path = 'data/processed',id_name = 'imdb_id',poster_col = 'cover url'):
    
    posters_folder_path = os.path.join(data_path,f"posters")
    if not os.path.exists(posters_folder_path):
        os.mkdir(posters_folder_path)

    file_path = os.path.join(data_path,
                             f"posters/{int(row[id_name])}.jpg")

    status = download_image_url(row[poster_col],
                   file_path
                  )
    if status == 'ERROR':
        return int(row[id_name])


def data_splits(input_dir ='data/processed',output_dir ='data/processed',train_proportion = 0.7):    
    df = pd.read_csv(os.path.join(input_dir,"df.csv"),parse_dates=['release_date'],usecols=['imdb_id','release_date'])
    df=df.sort_values(by='release_date',ascending=True)

    train_val_split = int(round(len(df)*train_proportion))
    val_test_split = int(round(len(df)*(train_proportion+(1-train_proportion)/2)))

    train_ids = df[:train_val_split]['imdb_id']
    val_ids = df[train_val_split:val_test_split]['imdb_id'].values
    test_ids = df[val_test_split:]['imdb_id'].values

    df[df['imdb_id'].isin(train_ids)].to_csv(os.path.join(output_dir,"train.csv"),index=False)
    df[df['imdb_id'].isin(val_ids)].to_csv(os.path.join(output_dir,"val.csv"),index=False)
    df[df['imdb_id'].isin(test_ids)].to_csv(os.path.join(output_dir,"test.csv"),index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--imdb_api_data',default=0,type=int)
    parser.add_argument('--bom_data',default=0,type=int)
    parser.add_argument('--download_posters',default=0,type=int)
    parser.add_argument('--metadata',default=0,type=int)
    parser.add_argument('--imdb_dump_data',default=0,type=int)
    parser.add_argument('--train_proportion',default=0.7,type=float)

    args = parser.parse_args()

    if args.metadata == 1:
        print('creating metadata df')
        get_metadata()
    
    if args.imdb_dump_data == 1:
        print('processing imdb dump data')
        get_imdb_dump_data()
    
    print('merging all')
    merge_all()
    data_splits(train_proportion = args.train_proportion)

    if args.download_posters == 1:
        print('downloading posters')
        df = pd.read_csv('data/processed/df.csv')

        errors = Parallel(n_jobs=-1)(delayed(download_poster)(row) for _,row in df[['imdb_id','cover url']].iterrows())
        errors = list(filter(None,errors))

        print(f'{len(errors)} posters could not be downloaded')
