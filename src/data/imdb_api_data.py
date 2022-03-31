import pickle
from glob import glob
import pandas as pd
from imdb import Cinemagoer
import numpy as np
import os


def extract_from_list_col(dataframe, col, max_items=4, normalize=True):
    return dataframe[col].apply(
        lambda x: extract_from_list(x, max_items=max_items, normalize=normalize)
    )


def extract_from_list(list_, max_items=4, normalize=True):
    if isinstance(list_, list):
        if normalize:
            list_ = [i.lower().strip() for i in list_[:4]]
        else:
            list_ = list_[:4]
    else:
        return np.nan

    return "|".join(list_)


COLUMNS_TO_DOWNLOAD = [
    "director",
    "countries",
    "country codes",
    "language codes",
    "languages",
    "box office",
    "cover url",
    "full-size cover url",
    "production companies",
]


class IMDBApiDataLoader:
    def __init__(
        self,
        input_dir="data/raw",
        output_dir="data/processed",
        columns_to_download=COLUMNS_TO_DOWNLOAD,
    ):

        self.columns_to_download = columns_to_download
        self.input_dir = input_dir
        self.output_dir = output_dir

    def download_from_api(self, ids_to_download, movies_per_file=100):
        
        # Get Data From API
        loaded_ids = []
        cg = Cinemagoer()

        # Continue where we left off
        files_names = glob(os.path.join(self.input_dir, "imdb_api_chkp_*.pickle"))
        #print(files_names)
        if files_names:
            for i in files_names:
                with open(i, "rb") as handle:
                    b = pickle.load(handle)
                loaded_ids.extend([j["imdb_id"] for j in b])

        #print(len(loaded_ids),len(ids_to_download))
        ids_to_download = list(set(ids_to_download) - set(loaded_ids))

        num_movies = len(ids_to_download)
        movies_per_file = min(num_movies,movies_per_file)

        print(f"fetching data of {num_movies} movies")
        c = 1

        rows = []
        for imdb_id in ids_to_download:
            obj = cg.get_movie(imdb_id)
            row = {"imdb_id": imdb_id}

            for col in self.columns_to_download:

                try:
                    row[col] = obj[col]
                    if col == "production companies":
                        row[col] = [i["name"] for i in row["production companies"]]
                    if col == "director":
                        row[col] = [i["name"] for i in row["director"]]
                    if col == "box office":
                        for k, v in row["box office"].items():
                            row[k] = v
                        row.pop("box office")
                except KeyError:
                    row[col] = np.nan

            rows.append(row)

            if c % movies_per_file == 0:
                print(round(c * 100 / num_movies, 2))
                with open(
                    os.path.join(self.input_dir, f"imdb_api_chkp_{c}_{imdb_id}.pickle"),
                    "wb",
                ) as handle:
                    pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
                rows = []
            
            c += 1
            

    def to_df(self):
        # Generate dataframe from downloaded data
        data = []
        files_names = glob(os.path.join(self.input_dir, "imdb_api_chkp_*.pickle"))
        if files_names:
            for i in files_names:
                with open(i, "rb") as handle:
                    b = pickle.load(handle)
                data.extend(b)
        data = pd.DataFrame(data)
        return data

    # Clean a bit
    def clean(self, data):
        data["Opening Weekend"] = data[
            [i for i in data.columns if "Opening Weekend" in i]
        ].apply(lambda x: x.dropna().sum(), axis=1)

        data = data[
            [
                "imdb_id",
                "director",
                "countries",
                "country codes",
                "language codes",
                "languages",
                "Budget",
                "cover url",
                "full-size cover url",
                "production companies",
                "Opening Weekend",
            ]
        ]

        list_cols = [
            "director",
            "countries",
            "country codes",
            "language codes",
            "languages",
            "production companies",
        ]

        for i in list_cols:
            data.loc[:, i] = extract_from_list_col(data, i)

        return data

    def to_csv(self, data):
        # Save
        data.to_csv(os.path.join(self.output_dir, "imdb_api_data.csv"), index=False)

    def run_all(self, ids_to_download, movies_per_file=100):
        self.download_from_api(ids_to_download, movies_per_file)
        data = self.to_df()
        data = self.clean(data)
        self.to_csv(data)
        return data


if __name__ == "__main__":

    ids_to_download = pd.read_csv("data/processed/filtered_id_list.csv")[
        "imdb_id"
    ].unique()

    idl = IMDBApiDataLoader()
    idl.run_all(ids_to_download)
