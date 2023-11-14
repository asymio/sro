from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Генерация случайных данных для примера
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация классификаторов
clf1 = LogisticRegression(random_state=42)
clf2 = RandomForestClassifier(random_state=42)
clf3 = SVC(random_state=42, probability=True)

# Создание квазилинейной композиции с использованием метода голосования
ensemble_clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='soft')

# Обучение композиции на обучающем наборе
ensemble_clf.fit(X_train, y_train)

# Получение предсказаний на тестовом наборе
y_pred = ensemble_clf.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность квазилинейной композиции: {accuracy}')
# sro
