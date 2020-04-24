from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

iris = load_breast_cancer()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=123
)

# Inicializacion de los 3 clasificadores a ensamblar.
model1 = LogisticRegression(random_state=123, max_iter = 2500)
model2 = GaussianNB()
model3 = RandomForestClassifier(random_state=123)

# Ensamble de los clasificadores.
vote = VotingClassifier(estimators=[('lr', model1),
                                    ('gnb', model2),
                                    ('rfc', model3)],
                        voting='soft')

vote.fit(X_train, y_train)

print('Clasificador por votación:')
print('Exactitud:', vote.score(X_train, y_train))
print('Precision:', vote.score(X_test, y_test))

print('###############################')

model1.fit(X_train, y_train)
print('Regresion logistica:')
print('Exactitud:', model1.score(X_train, y_train))
print('Precision:', model1.score(X_test, y_test))

model2.fit(X_train, y_train)
print('Bayes ingenuo:')
print('Exactitud:', model2.score(X_train, y_train))
print('Precision:', model2.score(X_test, y_test))

model3.fit(X_train, y_train)
print('Bosque aleatorio:')
print('Exactitud:', model3.score(X_train, y_train))
print('Precision:', model3.score(X_test, y_test))