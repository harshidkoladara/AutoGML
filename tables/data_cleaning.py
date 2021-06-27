import pandas as pd
import os
from math import floor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pickle


class DataCleaning():
    def __init__(self, path):
        self.path = path
        self.raw_files = [x for x in os.listdir(self.path) if x[0:3] == 'raw']
        self.data_file = [x for x in os.listdir(self.path) if x[0:3] != 'raw']

    def combine_and_remove_duplicate(self):
        na = ['-', 'na', 'nan', 'NaN', 'NA']
        raw = pd.read_csv(self.path + self.raw_files[0], na_values=na)

        for x in range(1, len(self.raw_files)):
            df = pd.read_csv(self.path + self.raw_files[x], na_values=na)
            raw = pd.concat([raw, df])

        raw = raw.drop_duplicates()
        return raw

    def remove_missing_value_column(self, raw):
        data = pd.read_csv(self.path + self.data_file[0])
        for x in raw.columns:
            try:
                missing = ((raw[x].isnull().sum()*100) / raw.shape[0])
                if list(data[data['Columns'] == x]['Nullable'])[0]:
                    raw = raw.drop(columns=[x])
                elif missing > 75:
                    raw = raw.drop(columns=[x])
            except:
                pass
        return raw

    def remove_missing_value_row(self, raw):
        threshold = floor(raw.shape[1]/2)
        droppable_rows = []
        for x in range(raw.shape[0]):
            try:
                if raw.loc[x, :].isna().any(axis=0) and raw.loc[x, :].isna().sum(axis=0) > threshold:
                    droppable_rows.append(x)
            except Exception as e:
                pass
        else:
            raw = raw.drop(droppable_rows, axis=0)

        return raw

    def transform_data(self, raw):
        encoder = {}
        data = pd.read_csv(self.path + self.data_file[0])
        for x in raw.columns:
            if (list(data[data['Columns'] == x]['Dtype'])[0] == 'category' or list(
                    data[data['Columns'] == x]['Dtype'])[0] == 'bool'):
                le = LabelEncoder()
                raw[x] = le.fit_transform(raw[x])
                encoder[x] = le
        filename = f'{self.path}/encoder.h5'
        pickle.dump(encoder, open(filename, 'wb'))
        return raw
        # One way is that save object with dataset_column_name as pickel and apply also in testing when needed
        # but It may consume a way more storage and have a lots of operations

        # same will be done on standerization too

    def fill_missing(self, raw):
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        columns = raw.columns
        X = imputer.fit_transform(raw)
        raw = pd.DataFrame(X, columns=columns)
        return raw
