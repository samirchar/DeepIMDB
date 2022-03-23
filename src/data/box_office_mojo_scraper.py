import re
from decimal import Decimal
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from stem.control import Controller
from stem import Signal

def get_tor_session():
    # initialize a requests Session
    session = requests.Session()
    # setting the proxy of both http & https to the localhost:9050 
    # this requires a running Tor service in your machine and listening on port 9050 (by default)
    session.proxies = {"http": "socks5://localhost:9050", "https": "socks5://localhost:9050"}
    return session

def renew_connection():
    with Controller.from_port(port=9051) as c:
        c.authenticate(password="password")
        # send NEWNYM signal to establish a new clean connection through the Tor network
        c.signal(Signal.NEWNYM)


class BoxOfficeMojoScraper:
    
    def __init__(self,imdb_id_list,tor_sess=None):
        self.imdb_id_list = imdb_id_list
        self.movies = []
        self.tor_sess = tor_sess
        
    def __get_movies_objects(self,imdb_id):
        
        imdb_id = str(int(imdb_id)).zfill(7)
        url = f"https://www.boxofficemojo.com/title/tt{imdb_id}/"
        
        
        table = []
        objects = []

        if self.tor_sess:

            
            while table == []:
                try:
                    page = self.tor_sess.get(url)#requests.get(url)
                    soup=BeautifulSoup(page.text,"html.parser")
                    table = soup.find("div",class_="a-section a-spacing-none mojo-performance-summary-table")

                    if table:
                        objects = table.find_all("div")
                    else:
                        raise requests.exceptions.ConnectionError
                except requests.exceptions.ConnectionError:
                    print('renewing connection')
                    renew_connection()
                    self.tor_sess = get_tor_session()
                    sleep(5)

        else:
            page = requests.get(url)
            soup=BeautifulSoup(page.text,"html.parser")
            table = soup.find("div",class_="a-section a-spacing-none mojo-performance-summary-table")
            
            if table:
                objects = table.find_all("div")
                print(imdb_id)

        return objects
    
    def __get_single_movie_row(self,objects):
        row = {}
        for o in objects:
            col = re.findall("[a-zA-Z]+",o.find('span',class_="a-size-small").text.strip())[0]
            data = o.find('span',class_="a-size-medium a-text-bold").text.strip()
            
            if (data =='-')|(data=='–'):
                data = np.nan
            else:
                data = float(Decimal(re.sub(r'[^\d.]', '', data)))
                
            row[col] = data
        return row

    
    def extract_movies(self):
        tot_movies = len(self.imdb_id_list)
        c = 1
        for imdb_id in self.imdb_id_list:
            objects = self.__get_movies_objects(imdb_id)
            row = self.__get_single_movie_row(objects)
            row['imdb_id'] = imdb_id
            self.movies.append(row)
            

            checkpoint = round(tot_movies/10)-1
            completed_pct = c*100/tot_movies

            if checkpoint>0:
                if (c%checkpoint==0):
                    print(f'{completed_pct}% done!')

                    self.to_df().to_csv(f'data/raw/box_office_mojo_data_chkp_{c}_{imdb_id}.csv',index=False)
                    self.movies = [] # reset otherwise would generate duplicates

            if (c%100==0)&(self.tor_sess is not None):
                renew_connection()
                self.tor_sess = get_tor_session()
                #print(self.tor_sess.get("http://icanhazip.com").text)
                print("getting new ip")
            c+=1
        return self.to_df()
    
    def to_df(self):
        return pd.DataFrame(self.movies)

from time import sleep
import multiprocess as mp
from glob import glob

if __name__ == "__main__":

    imdb_ids_list = pd.read_csv('data/processed/filtered_id_list.csv')['imdb_id'].values

    
    scraped_files =  glob('data/raw/*chkp*.csv')

    if scraped_files:
        imdb_ids_scraped = pd.concat([pd.read_csv(i,usecols=['imdb_id']) \
            for i in glob('data/raw/*chkp*.csv')],ignore_index=True).values.flatten()
        imdb_ids_list = list(set(imdb_ids_list) - set(imdb_ids_scraped))

    print(f'scraping {len(imdb_ids_list)} movies')
    
    def extract(imdb_ids,use_tor = True):
        
        tor_sess = None

        if use_tor:
            renew_connection()

            sleep(5)
            tor_sess = get_tor_session()

        bom2=BoxOfficeMojoScraper(imdb_ids,tor_sess)
        return bom2.extract_movies()

    n_jobs = mp.cpu_count()

    batches = list(range(1,len(imdb_ids_list),len(imdb_ids_list)//n_jobs))
    imdb_ids_batches = []
    for i,j in zip(batches[:-1],batches[1:]):
        imdb_ids_batches.append(list(imdb_ids_list[i:j]))
    imdb_ids_batches.append(list(imdb_ids_list[batches[-1]:]))

    with mp.Pool(n_jobs) as p:
        data = p.map(extract,imdb_ids_batches)
    

    #box_office_mojo_data_v2 = pd.concat(box_office_mojo_data_v2,ignore_index=True)

    #Save dataframe

    box_office_mojo_data_v2 = pd.concat([pd.read_csv(i) \
        for i in glob('data/raw/*chkp*.csv')],ignore_index=True)
    
    box_office_mojo_data_v2 = box_office_mojo_data_v2\
            .drop_duplicates()\
            .dropna(subset=['Worldwide'])

    box_office_mojo_data_v2.reset_index(drop=True,inplace=True)

    box_office_mojo_data_v2.to_csv('data/processed/box_office_mojo_data_v2.csv',index=False)
