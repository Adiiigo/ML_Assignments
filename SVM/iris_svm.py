from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = datasets.load_iris()
print(iris.feature_names[:2])
X = iris.data[:, :2]  # we only take the first two features.
print(X)
y = iris.target
print(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3)    # this is just conversion of iris data in proper format..jo naam hai wo satya hai
print("-----------test data-----------------")
print(x_test)
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(x_train, y_train)
print("--------------------")
print("y_test")
print(y_test)
print("predicted_label_from_svc")
print(clf.predict(x_test))
print(metrics.classification_report(y_test, clf.predict(x_test)  , digits=2))


