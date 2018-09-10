import pandas as pd
from sklearn import tree, metrics
from sklearn.preprocessing import LabelEncoder


def get_data(path_to_data='/home/arvind/ciml/decision_tree/courses_data_train.csv'):
    df = pd.read_csv(path_to_data)

    # Using LabelEncoder here to convert categorical values to numeric
    le = LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])

    x = df.drop(['target', 'rating'], axis=1).values
    y = df['target'].values
    return x, y


x, y = get_data()

clf = tree.DecisionTreeClassifier()
clf_fit = clf.fit(x, y)
predicted = clf_fit.predict(x)

print('Training accuracy: {}'.format(metrics.accuracy_score(y, predicted)))
