import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_data(path_to_data='/home/arvind/ciml/data/courses_data_train.csv'):
    with open(path_to_data) as f:
        reader = csv.DictReader(f)
        features = set(reader.fieldnames)

        if 'target' in features:
            features.remove('target')
        if 'rating' in features:
            features.remove('rating')

        data = [row for row in reader]
        return data, features


def get_data_numeric(path_to_data='/home/arvind/ciml/data/courses_data_train.csv'):
    df = pd.read_csv(path_to_data)

    # Using LabelEncoder here to convert categorical values to numeric
    le = LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])

    x = df.drop(['target', 'rating'], axis=1).values
    y = df['target'].values
    return x, y
