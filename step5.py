from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X, y = iris.data, iris.target

#model = svm.SVC(probability=True, random_state=0)
model = LogisticRegression()
scores = cross_val_score(model, X, y, scoring='accuracy', cv=3)
print(scores)