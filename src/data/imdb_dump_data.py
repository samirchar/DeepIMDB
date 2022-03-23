import pandas as pd
import os

processed_meta_data = pd.read_csv('data/processed/meta_data_cleaned.csv')

def get_imdb_dump_data(imdb_ids_list = processed_meta_data['imdb_id'].unique(),input_dir ='data/processed/oficial_imdb_dataset'):

    
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

if __name__ == "__main__":
    get_imdb_dump_data()