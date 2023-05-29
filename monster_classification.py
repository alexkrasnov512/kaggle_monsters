import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Данные по соревнованию от Kaggle, состоящие из обучающего и тестового набора
train = pd.read_csv('csv_files/train.csv')
test = pd.read_csv('csv_files/test.csv')

# Замена в тестовом и обучающем наборе параметра "color" на числовые значнения для предсказания
color = {'color': {'clear': 0, 'white': 0.2, 'blue': 0.4, 'green': 0.6, 'blood': 0.8, 'black': 1}}
train = train.replace(color)
test = test.replace(color)

# Выбор признаков и целевой переменной
X_train = train.iloc[:, 1:6]
y_train = train['type']
X_test = test.iloc[:, 1:6]

# Использование различных методов классификации для предсказания целевой переменной, в том числе:
# 1) Метод К-ближайших соседей
params_for_knn = {'n_neighbors': range(1, 51, 2), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
KNN_model = KNeighborsClassifier()
gKNN_model = GridSearchCV(KNN_model, params_for_knn, cv=5)
gKNN_model.fit(X_train, y_train)
KNN_prediction = gKNN_model.predict(X_test)   # результат на Kaggle, точность 0,72211

# 2) Метод опорных векторов
SVC_model = SVC(kernel='linear', C=0.4)
SVC_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)    # Лучший результат на Kaggle, точность 0,74102

# 3) Дерево решений
params_for_trees = {'max_depth': range(1, 22, 2),
                    'criterion': ['entropy'],
                    }
trees_model = DecisionTreeClassifier()
gtrees_model = GridSearchCV(trees_model, params_for_trees, cv=5)
gtrees_model.fit(X_train, y_train)
trees_prediction = gtrees_model.predict(X_test)   # результат на Kaggle, точность 0,65973

# 4) Гауссовский наивный Байес
gns = GaussianNB()
gns.fit(X_train, y_train)
gns_prediction = gns.predict(X_test)   # Лучший результат на Kaggle, точность 0,74102

# 5) Логистическая регрессия
log = LogisticRegression(C=0.4)
log.fit(X_train, y_train)
log_prediction = log.predict(X_test)    # Лучший результат на Kaggle, точность 0,74102

# Перевод предсказанных значений в серию pandas и сохранение в csv-файл
p_knn = pd.Series(KNN_prediction, index=test.id).to_frame(name='type').reset_index()
p_svc = pd.Series(SVC_prediction, index=test.id).to_frame(name='type').reset_index()
p_tree = pd.Series(trees_prediction, index=test.id).to_frame(name='type').reset_index()
p_gns = pd.Series(gns_prediction, index=test.id).to_frame(name='type').reset_index()
p_log = pd.Series(log_prediction, index=test.id).to_frame(name='type').reset_index()
p_knn.to_csv(r'C:\Py projects\kaggle_monsters\csv_files\predict_monsters_knn.csv', index=False)
p_svc.to_csv(r'C:\Py projects\kaggle_monsters\csv_files\predict_monsters_svc.csv', index=False)
p_tree.to_csv(r'C:\Py projects\kaggle_monsters\csv_files\predict_monsters_tree.csv', index=False)
p_gns.to_csv(r'C:\Py projects\kaggle_monsters\csv_files\predict_monsters_gns.csv', index=False)
p_log.to_csv(r'C:\Py projects\kaggle_monsters\csv_files\predict_monsters_log.csv', index=False)
