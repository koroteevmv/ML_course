### Работа с целевой переменной

#### Цель работы

Познакомиться с основными приемами обработки данных в отношении к целевой переменной: дискретизация, отбор признаков, устранение дисбаланса классов.

#### Содержание работы

1. Загрузите первый датасет для регрессии и познакомьтесь с его структурой.
1. Постройте простую модель регрессии и оцените ее качество.
1. Отберите признаки, наиболее сильно влияющие на значение целевой переменной.
1. Постройте модель на оставшихся данных и оцените ее качество.
1. Загрузите второй датасет для регрессии и постройте распределение целевой переменной.
1. Сгруппируйте значения целевой переменной в категории. Постройте получившееся распределение.
1. Загрузите датасет для классификации. Постройте распределение целевой переменной.
1. Разделите датасет на тестовую и обучающую выборки, постройте и оцените baseline модель классификации.
1. Постройте ту же модель с применением весов классов. Сравните ее качество, сделайте выводы.
1. Выравняйте распределение классов путем оверсемплинга с повторением

#### Методические указания

В предыдущих работах мы занимались предобработкой разных типов данных, ориентируясь на сами характеристики этих данных: их распределения, аномалии, шкалы и так далее. То есть, мы ориентировались на информацию, которую содержат эти переменные изолированно. Однако, для целей моделирования нас в первую очередь интересует то, как тот или иной признак влияет на значение целевой переменной. 

В данной работе мы познакомимся с основными операциями обработки данных, которые принимают во внимание соотношение признаков и целевой переменной. Среди них преобразования самой целевой переменной, отбор признаков, работа с несбалансированными датасетами. 

##### Знакомство с датасетом

```py
from sklearn.datasets import fetch_openml
```

```py
df = fetch_openml("mtp", version=1)

df.data.head()
```

```py
plt.hist(df.target, 100)
_ = plt.plot()
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-1.png?raw=true)

```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.25, random_state=42)
```

###### Построение базовой (baseline) модели

```py
baseline = LinearRegression()
baseline.fit(X_train, y_train)
bl_score = baseline.score(X_test, y_test)
bl_score
```

```
-1.6511340762242592
```

```py
y_pred = baseline.predict(X_test)
plt.plot(y_pred, y_pred, c='r')
plt.scatter(y_pred, y_test)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-2.png?raw=true)

##### Определение относительной важности признаков

```py
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=15).fit(X_train, y_train)
```

```py
sort = rf.feature_importances_.argsort()
plt.barh(df.data.columns[sort], rf.feature_importances_[sort])
plt.xlabel("Feature Importance")
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-3.png?raw=true)

```py
rf.feature_importances_[sort][-10:]
```

```
array([0.01009865, 0.01032001, 0.01242211, 0.01312193, 0.01436147,
       0.02522823, 0.02923168, 0.04488688, 0.07439497, 0.14930889])
```

```py
df.data.columns[sort][-10:]
```

```
Index(['oz160', 'oz155', 'oz197', 'oz137', 'oz158', 'oz18', 'oz35', 'oz48',
       'oz15', 'oz141'],
      dtype='object')
```

```py
trimmed = df.data[df.data.columns[sort][-20:]]
trimmed.head()
```

##### Построение новой модели

```py
X_train, X_test, y_train, y_test = train_test_split(trimmed, df.target, test_size=0.25, random_state=42)

better = LinearRegression()
better.fit(X_train, y_train)

print(bl_score)
better.score(X_test, y_test)
```

```py
-1.6511340762242592
0.3885997152790919
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-4.png?raw=true)

##### Автоматизация отбора признаков

```py
df.data.shape
```

```
(4450, 202)
```

```py
from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(RandomForestRegressor(n_estimators=15)).fit(df.data, df.target)
X_trimmed = sfm.transform(df.data)
X_trimmed.shape
```

```py
(4450, 55)
```

```py
X_train, X_test, y_train, y_test = train_test_split(X_trimmed, df.target, test_size=0.25, random_state=42)

better = LinearRegression()
better.fit(X_train, y_train)

print(bl_score)
better.score(X_test, y_test)
```

```py
-1.6511340762242592
0.4314653462618252
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-5.png?raw=true)

##### Дискретизация целевой переменной

```py
df = fetch_openml("CPMP-2015-regression", version=1)
df.data.drop(["instance_id"], inplace=True, axis=1)
df.data = pd.get_dummies(df.data)
df.data.head()
```

```py
plt.hist(df.target, 100)
_ = plt.plot()
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-6.png?raw=true)

```py
X_train, X_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.25, random_state=42)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-7.png?raw=true)

```py
from sklearn.preprocessing import KBinsDiscretizer

y_binned = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform").fit_transform(pd.DataFrame(y_train))
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-8.png?raw=true)

```py
y_binned
```

```py
array([[3.],
       [2.],
       [3.],
       ...,
       [2.],
       [1.],
       [3.]])
```

```py
y_binned = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile").fit_transform(pd.DataFrame(y_train))
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-9.png?raw=true)


##### Устранение дисбаланса классов

```py
df = fetch_openml("balance-scale", version=1)

df.data.head()
```


![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-10.png?raw=true)

```py
X_train, X_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.25, random_state=42, stratify=df.target)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-11.png?raw=true)

```py
baseline = LogisticRegression()
baseline.fit(X_train, y_train)
bl_score = baseline.score(X_test, y_test)
bl_score
```

```
0.8598726114649682
```

```py
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, baseline.predict(X_test)))
```

```
              precision    recall  f1-score   support

           B       0.00      0.00      0.00        12
           L       0.87      0.92      0.89        73
           R       0.86      0.94      0.90        72

    accuracy                           0.86       157
   macro avg       0.58      0.62      0.60       157
weighted avg       0.80      0.86      0.83       157
```

```py
print(confusion_matrix(y_test, baseline.predict(X_test)))
```

```py
[[ 0  6  6]
 [ 1 67  5]
 [ 0  4 68]]
```

```py
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))
class_weights
```

```
{'B': 4.216216216216216, 'L': 0.7255813953488373, 'R': 0.7222222222222222}
```

```py
              precision    recall  f1-score   support

           B       0.61      0.92      0.73        12
           L       0.97      0.92      0.94        73
           R       0.96      0.93      0.94        72

    accuracy                           0.92       157
   macro avg       0.85      0.92      0.87       157
weighted avg       0.94      0.92      0.93       157
```

```py
y_train.value_counts()
```

```
R    216
L    215
B     37
```

```py
X_train["target"] = y_train
```

```py
oversampled = X_train[X_train.target == "B"].sample(n=216-37, replace=True, ignore_index=True)
```

```py
oversampled = pd.concat([X_train, oversampled])
print(oversampled.shape)
oversampled.head()
```

|index|left-weight|left-distance|right-weight|right-distance|target|
|---|---|---|---|---|---|
|262|3|1|3|3|R|
|401|4|2|1|2|L|
|174|2|2|5|5|R|
|81|1|4|2|2|B|
|36|1|2|3|2|R|

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.4%20target/img/ml54-11.png?raw=true)

```py
y_train_OS = oversampled.target
X_train_OS = oversampled.drop(["target"], axis=1)
```

```py
OSmodel = LogisticRegression()
OSmodel.fit(X_train_OS, y_train_OS)
print(bl_score)
OSmodel.score(X_test, y_test)
```

```
              precision    recall  f1-score   support

           B       0.61      0.92      0.73        12
           L       0.97      0.92      0.94        73
           R       0.96      0.93      0.94        72

    accuracy                           0.92       157
   macro avg       0.85      0.92      0.87       157
weighted avg       0.94      0.92      0.93       157
```

#### Задания для самостоятельного выполнения

1. Исследуйте связь между количеством самых важных признаков, которые использует модель для обучения и тестовой точностью получившейся модели. Обучите несколько моделей с разным количеством наиболее важных признаков. Постройте график зависимости точности модели от количества признаков. Сделайте вывод.
1. Используйте другие методы отбора признаков:
	1. Исключение низкодисперсных признаков;
	1. Исключение по парным стаистическим критериям (хи-квадрат, тест Фишера, коэффициент корреляции, информационный критерий);
	1. Рекурсивное исключение признаков;
	1. Последовательное включение признаков;
	1. Исключение по L1-норме (гребневой регрессии).
1. Исследуйте влияние дискретизации целевой переменной на качество модели. Используйте уже продемострированный подход - построение базовой модели (baseline) и сравнение модели после обработки данных с базовой. Проверьте разное количество категорий, а также разные стратегии группировки. Сделайте выводы. Обратите внимание, что после биннинга целевой переменной она стала категориальной. А значит, задача превратилась в задачу классификации.
1. Изучите возможности библиотеки [imbalanced-learn](https://imbalanced-learn.org/stable/). Примените на данном примере возможности данной библиотеки для оверсемплинга и андерсемплинга выборки.

#### Контрольные вопросы

1. Какие модели лучше всего можно использовать для отбора признаков? Почему другие нельзя или нежелательно?
1. Зачем нужен этап отбора признаков? В каких случаях без него не обобйтись? А в каких его можно пропустить?
1. Какие есть методы отбора признаков? Найдите и опишите не менее пяти.
1. Зачем использовать дискретизацию непрерывной целевой переменной? В каких случаях это оправданно, а в каких - нет?
1. Почему дискретизацию целевой переменной нужно делать только после разделения на тестовую и обучающую подвыборки? Что такое утечка данных?
1. Какие есть стратегии устранения дисбаланса классов? В каких случаях стоит применять их и от чего зависит выбор стратегии?

#### Дополнительные задания
1. Повторите приведенный в данной работе анализ полностью на другом датасете. Сделайте вывод.
1. Используйте продвинутые алгоритмы дискретизации целевой переменной, например, CART.
1. Оформите алгоритм обработки данных как конвейер (pipeline) sklearn.
1. Для второго датасета модифицируйте предсказание второй модели так, чтобы вернуть постановку задачи регрессии. Для этого каждой категории присвойте численное значение. Это можно сделать, вычислив, например, медиану. Теперь можно считать, что модель предсказывает не метку категории, а конкретное численное значение. Как следствие, для оценки такой модели можно использовать метрики качества регрессии. Сравните метрики до и после преобразования.
1. Изучите и примените продвинутые стратегии оверсемплинга выборки: SMOTE, ASMO, ADASYN. Поясните механизм их работы и применимость в разных задачах.
1. Изучите и примените метод андерсемплинга выборки, основанный на Tomek Links. Сделайте вывод о его применимости к разным задачам.