import pandas as pd


def main():
    poster_ratings = pd.read_csv('data/raw/MovieGenre.csv',encoding = "ISO-8859-1")

    # Posters dataset #
    poster_ratings.drop_duplicates(subset=['imdbId'],inplace=True)
    poster_ratings.reset_index(drop=True,inplace=True)

    ## rename columns ##
    poster_ratings = poster_ratings[['imdbId','IMDB Score','Poster']]\
                                .rename(columns = {'imdbId':'imdb_id',
                                                    'IMDB Score':'imdb_rating',
                                                    'Poster':'poster_link'})

    ## format data ##
    poster_ratings['imdb_id'] = poster_ratings['imdb_id'].astype(float)

    poster_ratings.to_csv('data/processed/posters_links.csv',index = False)

if __name__ == "__main__":
    main()