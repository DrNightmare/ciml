from sklearn import tree, metrics
from data.utils import get_data_numeric

x, y = get_data_numeric()

clf = tree.DecisionTreeClassifier()
clf_fit = clf.fit(x, y)
predicted = clf_fit.predict(x)

print('Training accuracy: {}'.format(metrics.accuracy_score(y, predicted)))
