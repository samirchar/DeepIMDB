from src.models.multi_modal_transformers import get_model,AugmentedDistilBertForSequenceClassification
from src.data.model_dataprep import *
from transformers import Trainer, EarlyStoppingCallback, AutoTokenizer, TrainingArguments
import pandas as pd
from src.utils.abstract_classes import DataProcessor

    

def cyclical_encoding(data, col, max_val, min_val = 1, drop = True):
    """Encoding of cyclical features using sine and cosine transformation.
    Examples of cyclical features are: hour of day, month, day of week.

    :param df: A dataframe containing the column we want to encode
    :type df: :py:class:`pandas.DataFrame`
    :param col: The name of the column we want to encode.
    :type col: str
    :param max_val: The maximum value the variable can have. e.g. in hour of day, max value = 23
    :type max_val: int
    :param min_val: The minimum value the variable can have. e.g. in hour of day, min value = 1, defaults to 1
    :type min_val: int
    :return: dataframe with three new variables: sine and cosine of the features + the multiplicationof these two columns
    :rtype: :py:class:`pandas.DataFrame`
    """

    data[col] = data[col] - min_val #ensure min value is 0
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)

    if drop:
        data.drop(col,axis=1,inplace=True)

class IMDBDataProcessor(DataProcessor):
    """Base processor to be used for all preparation."""
    def __init__(self,
                 all_cols,
                 categoric_cols,
                 text_cols,
                 date_cols,
                 numeric_cols,
                 date_as_text,
                 numeric_as_text,
                 categories_as_text,
                 input_directory = 'data/processed',
                 output_directory = None):

        self.input_directory = input_directory
        self.output_directory = output_directory
        self.all_cols = all_cols,
        self.categoric_cols = categoric_cols,
        self.text_cols = text_cols,
        self.date_cols = date_cols,
        self.numeric_cols = numeric_cols

        self.date_as_text = date_as_text,
        self.numeric_as_text = numeric_as_text,
        self.categories_as_text = categories_as_text

    
    def read(self):
        """Read raw data."""
        self.train_ids = pd.read_csv(os.path.join(self.input_directory,'train.csv'),usecols=['imdb_id'])['imdb_id'].tolist()
        self.val_ids = pd.read_csv(os.path.join(self.input_directory,'val.csv'),usecols=['imdb_id'])['imdb_id'].tolist()
        self.test_ids = pd.read_csv(os.path.join(self.input_directory,'test.csv'),usecols=['imdb_id'])['imdb_id'].tolist()
        df = pd.read_csv(os.path.join(self.input_directory,'df.csv'),usecols = self.all_cols,parse_dates=['release_date']).sample(frac=1,random_state=42) #shuffle
        return df


    def pre_process(self,df,text_input_col,target_col):
        """Pre process the data so it is on the way needed by a model.
         For example, if I am building a seq2seq for time series, i need to to windowing,
         here is where I do that"""

        train,val,test = self.clean(df)

        train_dataset=create_dataset_split(train,self.text_cols,text_input_col,target_col,self.numeric_cols)
        val_dataset=create_dataset_split(val,self.text_cols,text_input_col,target_col,self.numeric_cols)
        test_dataset=create_dataset_split(test,self.text_cols,text_input_col,target_col,self.numeric_cols)

        return train_dataset,val_dataset,test_dataset

    def clean(self,df):
        """Cleaning data"""

        #Create splits
        if DEBUG:
            train = df[df['imdb_id'].isin(self.train_ids)].sample(frac=0.2)
            val = df[df['imdb_id'].isin(self.val_ids)].sample(frac=0.2)
            test = df[df['imdb_id'].isin(self.test_ids)]
        else:
            train = df[df['imdb_id'].isin(self.train_ids)]
            val = df[df['imdb_id'].isin(self.val_ids)]
            test = df[df['imdb_id'].isin(self.test_ids)]

        #Fill na in some columns with statistics
        naf = NAFiller(train)
        sc = StandardScaler()

        cols_to_impute = [i for i in self.numeric_cols if ('cos' not in i)&('sin' not in i)]
            
        for col in cols_to_impute:
            naf.fit(column = col,groupby=['top_genre','top_country'])
            naf.transform(train,round=True)
            naf.transform(val,round=True)
            naf.transform(test,round=True)

        if not self.numeric_as_text:
            train[self.numeric_cols] = sc.fit_transform(train[self.numeric_cols])
            val[self.numeric_cols] = sc.transform(val[self.numeric_cols])
            test[self.numeric_cols] = sc.transform(test[self.numeric_cols])

        return train,val,test

    def feature_engineer(self,df):
        """Class to perform feature engineering i.e. create new features"""

        #Additional auxilary columns
        df['top_genre'] = df['genres'].apply(lambda x: x.split(', ')[0])
        df['top_country'] = df['countries'].apply(lambda x: x.split(', ')[0] if isinstance(x,str) else x)

        if (not self.date_as_text): #If date is not as text, include numeri date features
            df['year'] = df['release_date'].dt.year
            df['month'] = df['release_date'].dt.month
            df['day'] = df['release_date'].dt.day
            df['season'] = df['release_date'].apply(date_to_season)
            df['dayofweek'] = df['release_date'].dt.dayofweek

            cyclical_encoding(df, 'month', max_val = 12, min_val = 1, drop = True)
            cyclical_encoding(df, 'day', max_val = 31, min_val = 1, drop = True) #TODO: max_val nsot exactly true
            cyclical_encoding(df, 'season', max_val = 4, min_val = 1, drop = True)
            cyclical_encoding(df, 'dayofweek', max_val = 6, min_val = 0, drop = True)

        df[categoric_cols] = df[categoric_cols].apply(lambda x: x.str.replace('|',', '),axis=0) #Change pipe to comma, its more meaningful
        df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'],errors='coerce')

        self.numeric_cols = list(df.dtypes.index[(df.dtypes == int)|(df.dtypes == float)].drop(['imdb_id',
                                                                                    'averageRating',
                                                                                    'revenue_worldwide_BOM']))

        if self.categories_as_text:
            self.text_cols+=self.categoric_cols

        if self.numeric_as_text:
            self.text_cols+=self.numeric_cols

        if self.date_as_text:
            self.text_cols+=self.date_cols


        return df, self.text_cols

    def save(self):
        """Saves processed data."""



if __name__ == "__main__":
        
    MODEL_NAME =  "distilbert-base-uncased" #"roberta-base" 
    TARGET_COL = 'averageRating'#'revenue_worldwide_BOM'
    MODEL_FOLDER = 'everything_as_text_except_numbers'#'everything_as_text'
    text_input_col = 'text_input'
    CATEGORIES_AS_TEXT = True
    NUMERIC_AS_TEXT = False
    DATE_AS_TEXT = False
    ADJUST_INFLATION = False
    USE_COLUMN_NAMES = False
    DEBUG = False

    FINAL_MODEL_NAME = f"{MODEL_NAME}-{TARGET_COL}"

    if ADJUST_INFLATION:
        FINAL_MODEL_NAME+='-inflation_adjusted'
        
    if USE_COLUMN_NAMES:
        FINAL_MODEL_NAME+='-with_column_names'
        
    FINAL_MODEL_PATH = f'models/{MODEL_FOLDER}/{FINAL_MODEL_NAME}'
    TRIALS_DF_PATH = f'models/{MODEL_FOLDER}/{FINAL_MODEL_NAME}_hparams_trials.csv'
    TEST_PERFORMANCE_PATH = f'models/{MODEL_FOLDER}/{FINAL_MODEL_NAME}_test_stats_best_model.csv'
        
    if USE_COLUMN_NAMES:
        assert CATEGORIES_AS_TEXT|NUMERIC_AS_TEXT|DATE_AS_TEXT, "can't use column names as text if there are no columns to treat as text!"
        
    print('Final model name: ',FINAL_MODEL_NAME)
    print('Saving at: ',MODEL_FOLDER)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    all_cols =  ['Budget',
                'averageRating',
                'cast',
                'countries',
                'director',
                'genres',
                'imdb_id',
                'languages',
                'overview',
                'production companies',
                'release_date',
                'revenue_worldwide_BOM',
                'runtimeMinutes',
                'title']


    categoric_cols = ['cast',
                    'countries',
                    'director',
                    'genres',
                    'languages',
                    'production companies']

    text_cols = ['title','overview']                  
    date_cols = ['release_date']
    numeric_cols = ['Budget',
                    'runtimeMinutes'
                ] 


    train_ids = pd.read_csv('data/processed/train.csv',usecols=['imdb_id'])['imdb_id'].tolist()
    val_ids = pd.read_csv('data/processed/val.csv',usecols=['imdb_id'])['imdb_id'].tolist()
    test_ids = pd.read_csv('data/processed/test.csv',usecols=['imdb_id'])['imdb_id'].tolist()
    df = pd.read_csv('data/processed/df.csv',usecols = all_cols,parse_dates=['release_date']).sample(frac=1,random_state=42) #shuffle


    #Additional auxilary columns
    df['top_genre'] = df['genres'].apply(lambda x: x.split(', ')[0])
    df['top_country'] = df['countries'].apply(lambda x: x.split(', ')[0] if isinstance(x,str) else x)

    df['year'] = df['release_date'].dt.year
    df['month'] = df['release_date'].dt.month
    df['day'] = df['release_date'].dt.day
    df['season'] = df['release_date'].apply(date_to_season)
    df['dayofweek'] = df['release_date'].dt.dayofweek


    if (not DATE_AS_TEXT): #If date is not as text, include numeri date features
        numeric_cols += ['year',
                        'month',
                        'day',
                        'season',
                        'dayofweek']

    if CATEGORIES_AS_TEXT:
    text_cols+=categoric_cols

    if NUMERIC_AS_TEXT:
    text_cols+=numeric_cols

    if DATE_AS_TEXT:
    text_cols+=date_cols


    df[categoric_cols] = df[categoric_cols].apply(lambda x: x.str.replace('|',', '),axis=0) #Change pipe to comma, its more meaningful
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'],errors='coerce')


    #Create splits
    if DEBUG:
        train = df[df['imdb_id'].isin(train_ids)].sample(frac=0.2)
        val = df[df['imdb_id'].isin(val_ids)].sample(frac=0.2)
        test = df[df['imdb_id'].isin(test_ids)]
    else:
        train = df[df['imdb_id'].isin(train_ids)]
        val = df[df['imdb_id'].isin(val_ids)]
        test = df[df['imdb_id'].isin(test_ids)]


    #Fill na in some columns with statistics
    naf = NAFiller(train)
    sc = StandardScaler()

    for col in numeric_cols:
        naf.fit(column = col,groupby=['top_genre','top_country'])
        naf.transform(train,round=True)
        naf.transform(val,round=True)
        naf.transform(test,round=True)

    if not NUMERIC_AS_TEXT:
        for col in numeric_cols:
            train[numeric_cols] = sc.fit_transform(train[numeric_cols])
            val[numeric_cols] = sc.transform(val[numeric_cols])
            test[numeric_cols] = sc.transform(test[numeric_cols])


    train_dataset=create_dataset_split(train,text_cols,text_input_col,TARGET_COL,numeric_cols)
    val_dataset=create_dataset_split(val,text_cols,text_input_col,TARGET_COL,numeric_cols)
    test_dataset=create_dataset_split(test,text_cols,text_input_col,TARGET_COL,numeric_cols)



    epochs = 15
    num_evals = 20
    patience = 2 if DEBUG else 30
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    eval_steps = 50 if DEBUG else 100

    hparams = {'batch_size' : [8,16,32],
            'learning_rate' : [1e-5, 2e-5, 3e-5,5e-5],
            'weight_decay' : [0.1,0.01],
            'repeats': range(1)}

    combs = list(product(*[range(len(i)) for i in list(hparams.values())]))
    scores = np.zeros([len(i) for i in list(hparams.values())])

    #trials_df_rows = []

    field_names = list(hparams.keys()) + ['score']
    dw = DictWriter(TRIALS_DF_PATH,field_names)

    currernt_trials_df = pd.read_csv(TRIALS_DF_PATH) #This can be empty or not.
    done_trials = currernt_trials_df.drop('score',axis=1).to_dict(orient='records') #empty list or not
    best_score = min(float('inf'),currernt_trials_df['score'].min())

    print(f'current best val score = {best_score}')

    for idx,comb_indexes in enumerate(combs):
        comb_values = {name:val[idx] for name,val,idx in zip(hparams.keys(),hparams.values(),comb_indexes)}
        
        if comb_values not in done_trials: #Check if trial alrready exists. If it does, skip.
            print('training with following hparams:')
            pprint(comb_values)

            training_args = TrainingArguments(output_dir=f"{MODEL_NAME}-{TARGET_COL}",
                                            per_device_train_batch_size = comb_values['batch_size'],
                                            learning_rate=comb_values['learning_rate'],
                                            weight_decay=comb_values['weight_decay'],
                                            seed = 42,
                                            fp16=True,
                                            per_device_eval_batch_size = 16,
                                            warmup_ratio=0.06,
                                            num_train_epochs = epochs,
                                            evaluation_strategy = "steps",
                                            save_strategy = "steps",
                                            load_best_model_at_end=True,
                                            eval_steps = eval_steps,
                                            save_steps = eval_steps,
                                            save_total_limit = 1,
                                            log_level = 'error',
                                            disable_tqdm = True

                                            )

            
            multi_modal_model = get_model(model_name = MODEL_NAME,
                                seed = training_args.seed,
                                num_numeric_features = len(numeric_cols),
                                num_image_features = 0)
            
            trainer = Trainer(
                model=multi_modal_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks = callbacks
            )
            


            trainer.train()

            score = trainer.evaluate()['eval_loss']

            scores[tuple(comb_indexes)] = score #outdated

            comb_values['score'] = score

            dw.add_rows([comb_values]) #Append to dataframe

            #trials_df_rows.append(comb_values)

            if score<best_score:
                print(f'got a better model, with score {np.round(score,4)} saving...')
                best_score = score
                trainer.save_model(FINAL_MODEL_PATH)
                
                print('saved')
        else:
            print('skipping trial because already exists')

