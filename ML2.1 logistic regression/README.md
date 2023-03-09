### Логистическая регрессия

#### Цель работы

Познакомиться с широко используемым методом бинарной классификации - логистической регрессией.

#### Содержание работы

1. Сгенерировать матрицу признаков и вектор целей для задачи классификации с использованием `make_classification` из библиотеки `sklearn.datasets`. Число классов возьмите равным двум.
2. Реализовать модель логистической регрессии методом градиентного спуска, не используя библиотечные функции.
3. Оценить качество построенной модели, используя метрики accuracy и F1-score.
4. Реализовать модель логистической регрессии `LogisticRegression` из библиотеки `sklearn.linear_model` и оценить качество построенной модели, используя метрики accuracy и F1-score.
5. Сравнить результаты двух реализаций.

#### Методические указания

Загрузим необходимые библиотеки:

```py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
```

Сгенерируем матрицу признаков и вектор целей для задачи классификации:

```py
from sklearn.datasets import make_classification
X,y = make_classification (n_samples=10000,
                          n_features=3,
                          n_informative=3,
                          n_redundant=0,
                          n_classes=2,
                          random_state=1)
```

Выведем первые пять строк X:

```py
pd.DataFrame(X).head()
```

Создадим конструктор класса, реализующего градиентный спуск:

```py
class SGD():
    def __init__(self, alpha, n_iters):
        self.theta = None
        self._alpha = alpha
        self._n_iters = n_iters

    def gradient_step(self, theta, theta_grad):
        return theta - self._alpha * theta_grad

    def optimize(self, X, y, start_theta, n_iters):
        theta = start_theta.copy()
        for i in range(n_iters):
            theta_grad = self.grad_func(X, y, theta)
            theta = self.gradient_step(theta, theta_grad)
        return theta

    def fit(self, X, y):
        m = X.shape[1]
        start_theta = np.ones(m)
        self.theta = self.optimize(X, y, start_theta, self._n_iters)
```

Создадим конструктор класса, реализующего логистическую регрессию:

```py
class LogReg(SGD):
    def sigmoid(self, X, theta):
        return 1. / (1. + np.exp(-X.dot(theta)))

    def grad_func(self, X, y, theta):
        n = X.shape[0]
        grad = 1. / n * X.transpose().dot(self.sigmoid(X, theta) - y)
        return grad

    def predict_proba(self, X):
        return self.sigmoid(X, self.theta)

    def predict(self, X):
        y_pred = self.predict_proba(X) > 0.5
        return y_pred
```
Создаём экземпляр класса со следующими параметрами:

```py
logreg = LogReg(1, 1000)
```

Добавим фиктивный столбец единиц к матрице признаков X:

```py
X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
```

Разделим данные на обучающую и валидационную части:

```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Обучим модель и сделаем предсказание:

```py
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
```

Выводим метрики качества:

```py
ac = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'accuracy = {ac:.2f} F1-score = {f1:.2f}')
```
Выведем значения вероятностей для каждого объекта принадлежать тому или иному классу:

```py
y_pred_proba = logreg.predict_proba(X_test)
```

Теперь проделаем то же самое, используя библиотечные функции.
Разделим данные на обучающую и валидационную части:

```py
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.2)
```

Создадим экземпляр класса:

```py
model = LogisticRegression()
```
Обучим модель и сделаем предсказание:

```py
model.fit(X_train, y_train)
y_pred_lr = model.predict(X_test)
```
Далее необходимо вывести метрики качества аналогичным образом и сделать сравнение результатов.


#### Задания для самостоятельного выполнения

1. Проверьте работу модели с другими значениями скорости обучения. Найдите значение, при котором градиентный спуск расходится.
2. Модифицируйте код модели таким образом, чтобы фиктивный столбец единиц добавлялся к матрице признаков внутри класса.
3. Сгенерируйте датасет с большим числом признаков и примените к нему созданную модель.
4. Выведите значения вероятностей для каждого объекта принадлежать тому или иному классу для библиотечной модели `LogisticRegression`?
5. Постройте ROC кривую и найдите площадь под этой кривой, используя функции `roc_curve`, `roc_auc_score` из библиотеки `sklearn.metrics`. Оцените качество модели по этой кривой.

#### Контрольные вопросы

1. Сформулируйте, в чем состоит задача классификации, придумайте несколько примеров.
2. Что такое шаг градиентного спуска?
3. Какая функция используется в качестве функции ошибки в модели логистической регрессии?
4. Зачем при реализации логистической регрессии к матрице признаков добавлялся столбец из единиц?

### Дополнительные задания:
Сгенерируйте датасет с большим числом классов. Решите задачу множественной классификации средствами sklearn.