import pandas as pd



USE_COLUMNS = ['imdb_id', 'runtimeMinutes', 'genres', 'cast', 'averageRating',
       'numVotes', 'title', 'original_title', 'overview', 'release_date',
       'poster_link', 'revenue_domestic_BOM', 'revenue_international_BOM',
       'revenue_worldwide_BOM', 'director', 'countries', 'country codes',
       'language codes', 'languages', 'Budget', 'cover url',
       'full-size cover url', 'production companies', 'Opening Weekend']

def merge_all(use_columns = USE_COLUMNS):
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
    df = titles.merge(processed_meta_data.drop(['genres'],axis=1),
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

    #Filter columns
    df = df[use_columns]


    df = df.merge(imdb_api_data,how='left',on='imdb_id')
    df.to_csv('data/processed/df.csv',index=False)