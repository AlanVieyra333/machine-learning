from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=123
)

ada = AdaBoostClassifier(n_estimators=200, random_state=123)
ada.fit(X_train, y_train)

exactitud = ada.score(X_train, y_train)
precision = ada.score(X_test, y_test)

print('exactitud', exactitud)
print('precision', precision)
