import re
from decimal import Decimal
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from stem.control import Controller
from stem import Signal
from time import sleep
import multiprocess as mp
from glob import glob
import os


def get_tor_session():
    # initialize a requests Session
    session = requests.Session()
    # setting the proxy of both http & https to the localhost:9050
    # this requires a running Tor service in your machine and listening on port 9050 (by default)
    session.proxies = {
        "http": "socks5://localhost:9050",
        "https": "socks5://localhost:9050",
    }
    return session


def renew_connection():
    with Controller.from_port(port=9051) as c:
        c.authenticate(password="password")
        # send NEWNYM signal to establish a new clean connection through the Tor network
        c.signal(Signal.NEWNYM)


class BoxOfficeMojoScraper:
    def __init__(self, imdb_id_list, tor_sess=None):
        self.imdb_id_list = imdb_id_list
        self.movies = []
        self.tor_sess = tor_sess

    def __get_movies_objects(self, imdb_id):

        imdb_id = str(int(imdb_id)).zfill(7)
        url = f"https://www.boxofficemojo.com/title/tt{imdb_id}/"

        table = []
        objects = []

        if self.tor_sess:

            while table == []:
                try:
                    page = self.tor_sess.get(url)  # requests.get(url)
                    soup = BeautifulSoup(page.text, "html.parser")
                    table = soup.find(
                        "div",
                        class_="a-section a-spacing-none mojo-performance-summary-table",
                    )

                    if table:
                        objects = table.find_all("div")
                    else:
                        raise requests.exceptions.ConnectionError
                except requests.exceptions.ConnectionError:
                    print("renewing connection")
                    renew_connection()
                    self.tor_sess = get_tor_session()
                    sleep(5)

        else:
            page = requests.get(url)
            soup = BeautifulSoup(page.text, "html.parser")
            table = soup.find(
                "div", class_="a-section a-spacing-none mojo-performance-summary-table"
            )

            if table:
                objects = table.find_all("div")
                print(imdb_id)

        return objects

    def __get_single_movie_row(self, objects):
        row = {}
        for o in objects:
            col = re.findall(
                "[a-zA-Z]+", o.find("span", class_="a-size-small").text.strip()
            )[0]
            data = o.find("span", class_="a-size-medium a-text-bold").text.strip()

            if (data == "-") | (data == "–"):
                data = np.nan
            else:
                data = float(Decimal(re.sub(r"[^\d.]", "", data)))

            row[col] = data
        return row


    def save_checkpoint(self,count,imdb_id,completed_pct):
        print(f"{completed_pct}% done!")
        self.to_df().to_csv(
            f"data/raw/box_office_mojo_data_chkp_{count}_{imdb_id}.csv",
            index=False,
        )
        self.movies = []  # reset otherwise would generate duplicates

    def extract_movies(self):
        tot_movies = len(self.imdb_id_list)
        checkpoint = round(tot_movies / 10) - 1
        c = 1
        for imdb_id in self.imdb_id_list:
            objects = self.__get_movies_objects(imdb_id)
            row = self.__get_single_movie_row(objects)
            row["imdb_id"] = imdb_id
            self.movies.append(row)

            completed_pct = c * 100 / tot_movies
            
            if checkpoint >0:
                if (c % checkpoint == 0): 
                    self.save_checkpoint(c,imdb_id,completed_pct)
            else:
                self.save_checkpoint(c,imdb_id,completed_pct) #For cases where there are just a few movies left


            if (c % 100 == 0) & (self.tor_sess is not None):
                renew_connection()
                self.tor_sess = get_tor_session()
                # print(self.tor_sess.get("http://icanhazip.com").text)
                print("getting new ip")
            c += 1
        return self.to_df()

    def to_df(self):
        return pd.DataFrame(self.movies)


def extract(imdb_ids, use_tor=True):

    tor_sess = None

    if use_tor:
        renew_connection()

        sleep(5)
        tor_sess = get_tor_session()

    bom2 = BoxOfficeMojoScraper(imdb_ids, tor_sess)
    return bom2.extract_movies()
    

class BoxOfficeMojoTORScraper:

    def __init__(self,ids_to_download = 'data/processed/filtered_id_list.csv',input_dir="data/raw",output_dir="data/processed"):
        '''
        download ids from list of ids or from a csv file with an imdb_id column
        '''

        self.filtered_id_csv = pd.read_csv(ids_to_download)["imdb_id"].values
        self.ids_to_download = self.filtered_id_csv if isinstance(ids_to_download,str) else ids_to_download
        self.input_dir = input_dir
        self.output_dir = output_dir

    def get_scraped_bom_files(self):
            return glob(os.path.join(self.input_dir,"box_office_mojo_data_chkp*.csv"))

    def run(self,n_jobs = -1):

        scraped_files = self.get_scraped_bom_files()

        if scraped_files:
            imdb_ids_scraped = pd.concat(
                [pd.read_csv(i, usecols=["imdb_id"]) for i in scraped_files],
                ignore_index=True,
            ).values.flatten()
            self.ids_to_download = list(set(self.ids_to_download) - set(imdb_ids_scraped))

        print(f"scraping {len(self.ids_to_download)} movies")
        
    
        #TODO: This could be a function in itself


        if len(self.ids_to_download)>0:
            print(len(self.ids_to_download))
            if n_jobs==-1:
                n_jobs = mp.cpu_count()
                
            if len(self.ids_to_download) <= n_jobs:
                n_jobs = 1
            
            batches = list(range(1, len(self.ids_to_download)+1, len(self.ids_to_download) // n_jobs))
            
            imdb_ids_batches = []
            for i, j in zip(batches[:-1], batches[1:]):
                imdb_ids_batches.append(list(self.ids_to_download[i:j]))
            imdb_ids_batches.append(list(self.ids_to_download[batches[-1] :]))

            with mp.Pool(n_jobs) as p:
                data = p.map(extract, imdb_ids_batches)

            # box_office_mojo_data_v2 = pd.concat(box_office_mojo_data_v2,ignore_index=True)

            # Save dataframe

            box_office_mojo_data_v2 = pd.concat(
                [pd.read_csv(i) for i in self.get_scraped_bom_files()], ignore_index=True
            )

            box_office_mojo_data_v2 = box_office_mojo_data_v2.drop_duplicates().dropna(
                subset=["Worldwide"]
            )

            box_office_mojo_data_v2.reset_index(drop=True, inplace=True)

            box_office_mojo_data_v2.to_csv(
                os.path.join(self.output_dir, "box_office_mojo_data_v2.csv"), index=False
            )


            
        else:
            print('all movies scraped')

if __name__ == "__main__":
    #os.system("tor")
    bom_scraper = BoxOfficeMojoTORScraper()
    bom_scraper.run()