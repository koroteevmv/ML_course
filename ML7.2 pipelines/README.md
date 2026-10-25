### Конвейеры в машинном обучении

#### Цель работы

На практике познакомиться с конвейерами (pipeline) как инструментом построения воспроизводимого и поддерживаемого кода машинного обучения. Освоить `Pipeline` и `ColumnTransformer` из `sklearn`, научиться объединять в один объект предобработку, извлечение признаков и модель, писать собственные трансформеры под конкретную задачу, подбирать гиперпараметры всего конвейера целиком, сохранять и загружать обученные конвейеры. В самостоятельных заданиях — применить конвейеры к задачам пакетной (batch) и потоковой (streaming) обработки данных.

#### Содержание работы

1. Зачем нужны конвейеры: воспроизводимость, защита от утечки данных, чистый код.
2. `Pipeline` — простейший конвейер «преобразование → модель».
3. `ColumnTransformer` — разные преобразования для разных типов столбцов.
4. Полный конвейер для таблицы со смешанными типами признаков.
5. Кастомные трансформеры: `FunctionTransformer`, `TransformerMixin`.
6. `FeatureUnion` — объединение параллельных ветвей обработки.
7. Подбор гиперпараметров всего конвейера через `GridSearchCV`.
8. Сохранение и загрузка конвейера.
9. Применение одного конвейера к разным задачам.

#### Подготовка данных

Работаем с тремя наборами данных:

- **`mixed_churn.csv`** — синтетический датасет интернет-магазина: числовые, категориальные, датовые и текстовые признаки, с пропусками и выбросами. Основной датасет для методических указаний.
- **`timeseries_stream.csv`** — временной ряд с трендом, сезонностью и аномалиями. Для потоковой обработки в самостоятельных заданиях.
- **`fetch_california_housing`** — реальный датасет из `sklearn` для сравнения и для заданий. Скачивается автоматически, интернет нужен только при первом запуске.

Загружаем данные:

```python
df = pd.read_csv('mixed_churn.csv')
print(df.shape)
print(df.dtypes)
print('\nПропуски:')
print(df.isna().sum())
print('\nДоля оттока:', df.churn.mean().round(3))
```

Посмотрим на распределение числовых признаков и на «выбросы»:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df['income'].dropna(), bins=50, ax=axes[0])
axes[0].set_title('Распределение дохода (с выбросами)')
sns.histplot(df['age'].dropna(), bins=30, ax=axes[1])
axes[1].set_title('Распределение возраста')
plt.tight_layout(); plt.show()
```

Доход имеет тяжёлый правый хвост: обычный `StandardScaler` будет «раздавлен» выбросами, нужен либо логарифм, либо робастный скейлер. Это одна из причин, по которым удобно собирать конвейер: преобразование дохода можно «упаковать» в общий пайплайн вместе со всем остальным.

![Исходные данные](https://github.com/koroteevmv/ML_course/blob/2025/ML7.2%20pipelines/img/ml72-1.png?raw=true)

Разделяем на train/test один раз. **Все дальнейшие преобразования должны настраиваться только на train** — это ключевая идея, которую мы будем демонстрировать.

```python
X = df.drop(columns='churn')
y = df['churn']

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(X_tr.shape, X_te.shape, y_tr.mean(), y_te.mean())
```

#### Методические указания

##### Зачем нужны конвейеры

Представим, что вы пишете код обучения модели без конвейеров. Типичный сценарий:

```python
# Плохо: ручные преобразования
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# 1) числовые признаки
scaler = StandardScaler().fit(X_tr[['age', 'income', 'days_since_signup', 'num_purchases']])
X_tr_num = scaler.transform(X_tr[['age', 'income', 'days_since_signup', 'num_purchases']])
X_te_num = scaler.transform(X_te[['age', 'income', 'days_since_signup', 'num_purchases']])

# 2) категориальные
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X_tr[['country', 'device', 'subscription']])
X_tr_cat = enc.transform(X_tr[['country', 'device', 'subscription']])
X_te_cat = enc.transform(X_te[['country', 'device', 'subscription']])

# 3) склеиваем
X_tr_proc = np.hstack([X_tr_num, X_tr_cat])
X_te_proc = np.hstack([X_te_num, X_te_cat])

# 4) обучаем
model = LogisticRegression(max_iter=1000).fit(X_tr_proc, y_tr)
print(model.score(X_te_proc, y_te))
```

Что здесь не так:

- **Ошибка 1: утечка данных.** Если по ошибке вызвать `scaler.fit(X)` на **всех** данных (включая test), информация из test «протечёт» в обучение. Качество будет завышено, а на новых данных — плохое. Отследить это глазами сложно: код большой, `fit` разбросан по разным местам.
- **Ошибка 2: рассинхронизация.** Если в одном месте применили `fit_transform` к train, а в другом — `transform` к test, но забыли сделать то же самое для нового преобразования (например, добавили логарифмирование), train и test разъедутся.
- **Ошибка 3: неудобно подбирать гиперпараметры.** `GridSearchCV` не сможет «перебрать» преобразования и модель как единое целое — придётся вручную писать циклы.
- **Ошибка 4: неудобно сохранять.** В продакшене нужно передать не только модель, но и все скейлеры/энкодеры. Легко забыть один — и результат на проде будет отличаться от ноутбука.

**Конвейер решает все эти проблемы.** Идея простая: обернуть всю цепочку «преобразование → ... → модель» в один объект, у которого есть `fit` и `predict`. Внутри `fit` конвейер последовательно вызывает `fit_transform` для каждого преобразователя и `fit` для финальной модели. Внутри `predict` — `transform` и `predict`. При кросс-валидации и подборе гиперпараметров `sklearn` работает с конвейером как с единой моделью, и **никогда не настраивает преобразования на данных, отличных от обучающих**.

##### Простейший Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

simple_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000)),
])

simple_pipe.fit(X_tr[['age', 'income']], y_tr)
score = simple_pipe.score(X_te[['age', 'income']], y_te)
print(f'Accuracy: {score:.3f}')
```

Что происходит внутри `fit`:

1. `StandardScaler.fit_transform(X_tr[['age', 'income']])` → масштабированные данные.
2. `LogisticRegression.fit(масштабированные, y_tr)`.

Внутри `score` (то есть `predict` + проверка):

1. `StandardScaler.transform(X_te[['age', 'income']])` — тот же скейлер, обученный на train, применяется к test.
2. `LogisticRegression.predict(...)`.

Прелесть в том, что мы физически **не можем** случайно вызвать `fit` на тесте: конвейер делает это только для train, а для test — только `transform`.

**Синтаксис параметров.** Чтобы обратиться к параметру шага, используется `__` (двойное подчёркивание):

```python
# Гиперпараметры конвейера
simple_pipe.set_params(clf__C=0.1, scaler__with_mean=False)
print(simple_pipe)
```

Этот синтаксис пригодится в `GridSearchCV`.

##### ColumnTransformer — разные преобразования для разных столбцов

В реальных таблицах столбцы разных типов: числовые нужно масштабировать, категориальные — кодировать one-hot, даты — разложить на компоненты, текст — векторизовать. Для числовых и категориальных столбцов нужны **разные** преобразователи. `ColumnTransformer` позволяет применить разные цепочки к разным столбцам и склеить результат.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_cols = ['age', 'days_since_signup', 'num_purchases']
cat_cols = ['country', 'device', 'subscription']

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         cat_cols),
    ],
    remainder='drop',  # все остальные столбцы выбросить
)

X_tr_proc = preprocess.fit_transform(X_tr)
X_te_proc = preprocess.transform(X_te)
print(f'До обработки: {X_tr.shape}, после: {X_tr_proc.shape}')
```

**Что важно понять:**

- `ColumnTransformer` работает как «параллельные ветви»: каждая ветвь обрабатывает свой набор столбцов.
- `remainder='drop'` — выбросить все столбцы, не упомянутые в `transformers`. Альтернативы: `'passthrough'` (оставить как есть) или `'drop'`. По умолчанию `'drop'`.
- `handle_unknown='ignore'` в `OneHotEncoder` — если в test встретится категория, которой не было в train (например, новый город), энкодер не упадёт с ошибкой, а закодирует нулями. Это критично для продакшена.

**Обратите внимание на масштаб.** `income` мы **не включили** в числовые признаки — потому что у него тяжёлый хвост, и обычный `StandardScaler` с ним плохо справляется. Это подводит нас к кастомным трансформерам.

##### Кастомные трансформеры: `FunctionTransformer` и `TransformerMixin`

`sklearn` предоставляет два основных способа добавить свою логику в конвейер.

###### `FunctionTransformer` — для простых функций

Если преобразование — это простая функция без параметров, которую нужно вычислить по каждому элементу или по всему столбцу, удобно использовать `FunctionTransformer`:

```python
from sklearn.preprocessing import FunctionTransformer

# Логарифмирование дохода (с константой 1, чтобы избежать log(0))
log_income = FunctionTransformer(np.log1p, validate=True)

# Извлечение длины текста
text_length = FunctionTransformer(
    lambda s: s.str.len().values.reshape(-1, 1),
    validate=False,
)
```

`validate=True` заставляет преобразователь проверить, что на вход подан массив numpy, а не DataFrame. Если вы работаете с Series (например, с текстом), ставьте `validate=False`.

**Пример: логарифм дохода.** Чтобы применить `FunctionTransformer` к одному столбцу внутри `ColumnTransformer`, передают имя столбца как «функцию-селектор»:

```python
preprocess = ColumnTransformer(
    transformers=[
        ('log_income', FunctionTransformer(np.log1p, validate=True), ['income']),
        ('num', StandardScaler(),
         ['age', 'days_since_signup', 'num_purchases']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         ['country', 'device', 'subscription']),
    ],
)
X_tr_proc = preprocess.fit_transform(X_tr)
print(X_tr_proc.shape)
```

После логарифмирования `income` можно масштабировать — уже обычным `StandardScaler`, и выбросы не «сломают» его так сильно. Добавим эту ветку:

```python
from sklearn.pipeline import Pipeline

income_pipe = Pipeline([
    ('log', FunctionTransformer(np.log1p, validate=True)),
    ('scale', StandardScaler()),
])

preprocess = ColumnTransformer([
    ('income', income_pipe, ['income']),
    ('num', StandardScaler(),
     ['age', 'days_since_signup', 'num_purchases']),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     ['country', 'device', 'subscription']),
])
```

Обратите внимание: `Pipeline` можно вкладывать в `ColumnTransformer` — это позволяет делать для отдельного столбца целую цепочку преобразований, а не одно.

###### `TransformerMixin` — для параметризованных трансформеров

Если трансформация зависит от данных (например, вычисляет статистику на train и потом применяет её к test), удобнее написать класс-наследник `BaseEstimator` и `TransformerMixin`:

```python
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Извлекает из дат день недели, месяц, год и признак выходного."""
    
    def __init__(self, date_columns):
        self.date_columns = date_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        result = []
        for col in self.date_columns:
            dt = pd.to_datetime(X_out[col])
            result.append(np.column_stack([
                dt.dt.dayofweek.values,
                dt.dt.month.values,
                dt.dt.year.values,
                dt.dt.dayofweek.isin([5, 6]).astype(int).values,
            ]))
        return np.hstack(result)
    
    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.date_columns:
            names += [
                f'{col}_dayofweek',
                f'{col}_month',
                f'{col}_year',
                f'{col}_is_weekend',
            ]
        return np.array(names)
```

Ключевые моменты:

- Наследуемся от `BaseEstimator` (даёт `get_params`/`set_params`, нужные для `GridSearchCV`) и `TransformerMixin` (даёт `fit_transform` по умолчанию).
- `__init__` принимает только гиперпараметры и **ничего не вычисляет**. Это требование `sklearn`: все вычисления — в `fit`.
- `fit` возвращает `self`, `transform` возвращает массив.
- `get_feature_names_out` не обязателен, но полезен: с ним можно передать имена признаков в `feature_importances_` или в графики важности.

Ещё один типовой трансформер — **извлечение признаков из текста**:

```python
class TextStats(BaseEstimator, TransformerMixin):
    """Простые статистики по текстовому столбцу."""
    
    def __init__(self, text_column):
        self.text_column = text_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        s = X[self.text_column].fillna('')
        return np.column_stack([
            s.str.len().values,
            s.str.split().str.len().values,
            s.str.count(r'good|great|excellent').values,
            s.str.count(r'bad|terrible|awful').values,
        ])
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['text_len', 'text_n_words',
                          'text_positive_hits', 'text_negative_hits'])
```

И трансформер **клиппинга выбросов** (обрезка по квантилям) — полезен для дохода, если не хочется логарифмировать:

```python
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Обрезает значения по квантилям, вычисленным на train."""
    
    def __init__(self, lower=0.01, upper=0.99, columns=None):
        self.lower = lower
        self.upper = upper
        self.columns = columns
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.columns)
        self.lower_ = X_df.quantile(self.lower).values
        self.upper_ = X_df.quantile(self.upper).values
        return self
    
    def transform(self, X):
        X_arr = np.asarray(X)
        return np.clip(X_arr, self.lower_, self.upper_)
```

`fit` запоминает границы по train, `transform` применяет их к train и test. Это важно: если пересчитать квантили на test, получится утечка.

**Собираем полный конвейер.** Объединим все кусочки:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Числовой блок: импутация + логарифм/скейлер
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

income_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p, validate=True)),
    ('scale', StandardScaler()),
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocess = ColumnTransformer([
    ('income', income_pipe, ['income']),
    ('num', num_pipe, ['age', 'days_since_signup', 'num_purchases']),
    ('cat', cat_pipe, ['country', 'device', 'subscription']),
    ('dt', DatetimeFeatures(['last_login', 'first_purchase']),
     ['last_login', 'first_purchase']),
    ('text', TextStats('review'), ['review']),
])

full_pipe = Pipeline([
    ('prep', preprocess),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
])

full_pipe.fit(X_tr, y_tr)
from sklearn.metrics import roc_auc_score
y_proba = full_pipe.predict_proba(X_te)[:, 1]
print(f'ROC-AUC: {roc_auc_score(y_te, y_proba):.3f}')
```

**Обратите внимание:** конвейер принимает на вход **исходный** `X`, в котором есть пропуски, выбросы, категории, даты и текст. Всё преобразование — внутри. На тесте `full_pipe.predict_proba(X_te)` даёт корректные предсказания без каких-либо ручных манипуляций.

##### Подбор гиперпараметров конвейера через `GridSearchCV`

Один из главных плюсов конвейера: гиперпараметры **всех** его шагов можно подбирать одновременно, и `sklearn` будет корректно «переобучать» препроцессинг на каждом фолде кросс-валидации.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'prep__num__imputer__strategy': ['mean', 'median'],
    'prep__cat__onehot__drop': [None, 'first'],
    'clf__C': [0.1, 1.0, 10.0],
    'clf__class_weight': [None, 'balanced'],
}

grid = GridSearchCV(
    full_pipe, param_grid, cv=5,
    scoring='roc_auc', n_jobs=-1, verbose=1,
)
grid.fit(X_tr, y_tr)

print(f'Лучшие параметры: {grid.best_params_}')
print(f'CV ROC-AUC: {grid.best_score_:.3f}')
print(f'Test ROC-AUC: {roc_auc_score(y_te, grid.predict_proba(X_te)[:, 1]):.3f}')
```

**Синтаксис.** Путь к параметру строится как `<шаг>__<подшаг>__...__<параметр>`. Например:

- `prep__num__imputer__strategy` — параметр `strategy` у `imputer` внутри ветки `num` внутри `prep`.
- `clf__C` — параметр `C` у `clf`.

Если ошибиться в имени, `GridSearchCV` сообщит `Invalid parameter` и подскажет ближайший вариант.

**Почему важно, что конвейер обучается целиком на фолде?** Внутри `GridSearchCV` для каждой комбинации параметров и каждого фолда вызывается `full_pipe.fit(X_tr_fold, y_tr_fold)`. Внутри этого `fit` все `imputer`, `scaler`, `encoder` настраиваются **только** на тренировочной части фолда. Если бы мы делали это вручную, пришлось бы отдельно оборачивать каждое преобразование в кросс-валидацию — легко ошибиться. С конвейером это получается автоматически.

##### Сохранение и загрузка конвейера

Обученный конвейер — это единый объект, который содержит все настройки препроцессинга и модель. Его удобно сохранять и передавать в продакшен.

```python
import joblib

# Сохранение
joblib.dump(grid.best_estimator_, 'churn_pipeline.joblib')

# Загрузка
loaded = joblib.load('churn_pipeline.joblib')

# Применение к новым данным (в том же формате, что и X_tr)
sample = X_te.iloc[:5]
pred = loaded.predict(sample)
proba = loaded.predict_proba(sample)[:, 1]
print('Предсказания:', pred)
print('Вероятности:', proba.round(3))
```

**Три вещи, которые важно запомнить:**

1. **Сохраняйте конвейер целиком, а не только модель.** Если сохранить только `clf`, при загрузке в продакшене не будет `scaler`/`encoder`, и результат будет отличаться.
2. **Новые данные должны иметь ту же структуру.** Конвейер ожидает те же столбцы, что и на обучении. Лишние столбцы можно игнорировать (`remainder='drop'`), но отсутствующих быть не должно.
3. **Версионируйте конвейеры.** Если модель переобучается с другими гиперпараметрами, старый файл лучше не перезаписывать, а хранить с версией (`churn_pipeline_v2.joblib`). Иначе откат на предыдущую версию будет невозможен.

##### Применение одного конвейера к разным задачам

Конвейер — универсальный механизм. Тот же `ColumnTransformer` можно применить к регрессии, заменив только последний шаг:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_openml

cal = fetch_openml(name='california_housing', version=1, parser='auto').frame
y_cal = cal.median_house_value
X_cal = cal.drop(['median_house_value', 'ocean_proximity'], axis=1)

# Тот же preprocess, но для регрессии
preprocess_reg = ColumnTransformer([
    ('num', StandardScaler(), X_cal.columns.tolist()),
])
reg_pipe = Pipeline([
    ('prep', preprocess_reg),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
])
reg_pipe.fit(X_cal, y_cal)
print(f'R²: {reg_pipe.score(X_cal, y_cal):.3f}')
```

Схема «один `ColumnTransformer` → разные модели → разные задачи» — стандарт в реальных проектах. Препроцессинг переиспользуется, а модель можно менять без переписывания всего кода.

#### Задания для самостоятельного выполнения

Все задания выполняются в ноутбуке. Для каждого пункта — код + краткий вывод.

##### Часть 1. Больше примеров сложной обработки

1. **Конвейер для Titanic.** Загрузите `sns.load_dataset('titanic')`. Постройте конвейер, который:
   - заполняет пропуски в `age` медианой по полу (`SimpleImputer` не умеет группировать, нужен кастомный трансформер или `ColumnTransformer` с отдельными `SimpleImputer` для мужчин и женщин);
   - заполняет `embarked` модой;
   - кодирует `sex`, `embarked`, `pclass` через `OneHotEncoder`;
   - извлекает из `name` титул (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`) и добавляет как категориальный признак;
   - из `cabin` извлекает первую букву (палуба) и добавляет как категориальный признак;
   - обучает `LogisticRegression` с `class_weight='balanced'`.
   
   Оцените ROC-AUC по 5-fold CV. Сравните с «наивным» конвейером без извлечения титула и палубы.

2. **Конвейер для текста.** Загрузите 20 Newsgroups (`sklearn.datasets.fetch_20newsgroups`, 4 класса). Постройте конвейер:
   - `TfidfVectorizer` с биграммами и `min_df=2`;
   - `TruncatedSVD` для уменьшения размерности (например, до 100 компонент);
   - `LogisticRegression`. 
   
   Сравните три варианта: без `TruncatedSVD`, с `TruncatedSVD(50)`, с `TruncatedSVD(200)`. Что даёт уменьшение размерности?

3. **Смешанный конвейер для регрессии.** Сгенерируйте данные с числовыми, категориальными и датовыми признаками (можно расширить `mixed_churn.csv`, заменив `churn` на числовую целевую переменную). Постройте конвейер, который обрабатывает все три типа и обучает `GradientBoostingRegressor`. Сравните с конвейером, где даты обрабатываются «в лоб» — кодируются как числа (ordinal) без извлечения компонент. Какой вариант лучше?

4. **Кастомный трансформер: циклические признаки.** Временные признаки (день недели, месяц) по своей природе цикличны: декабрь «близок» к январю. Стандартное числовое кодирование этого не учитывает. Напишите трансформер `CyclicDatetime`, который для месяца и дня недели возвращает **sin и cos** от угла: $\sin(2\pi k / K), \cos(2\pi k / K)$. Добавьте его в конвейер и сравните качество с обычным `DatetimeFeatures`.

5. **Кастомный трансформер: отбор признаков по важности.** Напишите `TopKByImportance`, который:
   - обучает `RandomForestClassifier` на входных признаках,
   - запоминает индексы `top_k` самых важных,
   - в `transform` оставляет только их.
   
   Вставьте в конвейер перед финальной моделью. Сравните с `SelectKBest` из `sklearn`.

6. **Конвейер с кросс-валидацией внутри.** Используйте `sklearn.model_selection.cross_validate` с конвейером и `return_estimator=True`. Для каждого фолда выведите значение метрики и **лучший гиперпараметр**, найденный на этом фолде (например, если внутри конвейера — `GridSearchCV`, обёрнутый в `Pipeline`). Убедитесь, что гиперпараметры различаются от фолда к фолду — это нормально и показывает, что выбор гиперпараметров сам по себе нестабилен.

##### Часть 2. Пакетная обработка (batch)

Пакетная обработка — это когда данные приходят **блоками** и каждый блок обрабатывается независимо. Это нужно, если данных слишком много для памяти, или если они лежат в разных файлах.

7. **Чтение CSV по частям.** Используйте `pd.read_csv('mixed_churn.csv', chunksize=500)`. Пройдитесь по чанкам и для каждого посчитайте: количество строк, средний доход, долю оттока. Соберите результаты в один DataFrame. В чём преимущество такого подхода перед `pd.read_csv('mixed_churn.csv')` в память?

8. **Предсказания по чанкам.** Обучите конвейер на `mixed_churn.csv` (без целевой переменной). Сохраните. Затем загрузите конвейер в отдельном ноутбуке и пройдитесь по чанкам большого файла (можно сгенерировать несколько копий `mixed_churn.csv` в один большой файл ~50 000 строк). Для каждого чанка сделайте `predict_proba`, сохраните результаты в список. Склейте всё в один файл предсказаний.

9. **Параллельная обработка чанков.** Ускорьте предыдущее задание через `joblib.Parallel`:

```python
from joblib import Parallel, delayed

def process_chunk(chunk, pipeline):
    return pipeline.predict_proba(chunk)[:, 1]

results = Parallel(n_jobs=-1)(
    delayed(process_chunk)(chunk, loaded_pipeline)
    for chunk in pd.read_csv('big_file.csv', chunksize=5000)
)
```

Сравните время с последовательным циклом. Когда параллелизм помогает, а когда — нет (намек: `n_jobs=-1` на маленьких файлах может работать медленнее)?

10. **Пакетная обработка с разными типами файлов.** Предположим, что данные приходят в трёх файлах разного формата: `part1.csv`, `part2.parquet`, `part3.json`. Напишите функцию `process_files(paths)`, которая для каждого файла определяет формат, читает данные, применяет общий конвейер и возвращает объединённый результат.

##### Часть 3. Потоковая обработка (streaming)

Потоковая обработка — это когда данные приходят **по одному объекту** (или маленькими порциями) в реальном времени, и модель должна сразу выдавать предсказание. Полноценный streaming (Kafka, Flink) мы не рассматриваем; имитируем его с помощью генераторов Python и `yield`.

11. **Генератор потока.** Напишите генератор `stream_rows(csv_path)`:

```python
def stream_rows(csv_path):
    for chunk in pd.read_csv(csv_path, chunksize=1):
        yield chunk
```

Он выдаёт по одному объекту за раз. Пройдитесь по нему на `mixed_churn.csv` и для каждого объекта вызовите `pipeline.predict` (один объект). Замерьте общее время. Сравните с обработкой всего файла «пакетом». Почему по одному — намного медленнее?

12. **Мини-батчи.** Модифицируйте генератор: пусть он выдаёт мини-батчи по 100 объектов:

```python
def stream_batches(csv_path, batch_size=100):
    buffer = []
    for _, row in pd.read_csv(csv_path).iterrows():
        buffer.append(row.to_dict())
        if len(buffer) == batch_size:
            yield pd.DataFrame(buffer)
            buffer = []
    if buffer:
        yield pd.DataFrame(buffer)
```

Сравните скорость с `batch_size=1` и `batch_size=100`. Найдите оптимальный размер батча для вашего конвейера.

13. **Инкрементальное обучение.** `sklearn` поддерживает частичное обучение (`partial_fit`) для некоторых моделей — `SGDClassifier`, `MultinomialNB`. Постройте конвейер, в котором финальная модель — `SGDClassifier(loss='log_loss')`. Настройте препроцессинг **один раз** на первой порции данных, а затем для каждой новой порции вызывайте `partial_fit` только для модели. Сравните качество с обучением «с нуля» на всех данных. В чём ограничение такого подхода?

14. **Потоковая обработка с адаптацией.** В реальном потоке распределение данных может меняться (concept drift). Реализуйте простую адаптацию: обучите конвейер на первой половине потока, а на второй — переобучайте `StandardScaler` (или `SimpleImputer`) каждые 1000 объектов. Сравните качество на второй половине с моделью без адаптации. Обсудите в текстовой ячейке, какие риски есть у адаптации (намек: если модель адаптируется слишком быстро, она может «забыть» старое распределение).

15. **Онлайн-метрики.** Постройте график зависимости ROC-AUC (или accuracy) от номера батча. Используйте поток `timeseries_stream.csv`: для каждого батча обучите простую модель (например, `SGDClassifier`) на **всех предыдущих** батчах и оцените на текущем. Как меняется качество со временем? Есть ли «провалы» на аномалиях?

##### Часть 4. Комбинированные сценарии

16. **Гибридный конвейер для временного ряда.** Постройте конвейер, который:
   - принимает временной ряд;
   - извлекает лаговые признаки (сдвиги на 1, 2, 3, 7 шагов) — кастомный трансформер;
   - извлекает статистики по скользящему окну (среднее, std за 7 и 30 шагов) — кастомный трансформер;
   - обучает `GradientBoostingRegressor` для предсказания следующего значения.
   
   Оцените на `timeseries_stream.csv` (разделите на train/test по времени, не случайно). Какой набор лагов/окон даёт лучшее качество?

17. **Конвейер как сервис.** Сохраните лучший конвейер в `joblib`. Напишите функцию `predict_single(row_dict, pipeline_path)`:
   - загружает конвейер из файла;
   - превращает один словарь в DataFrame (один ряд);
   - возвращает предсказание и вероятность.
   
   Протестируйте на 10 новых примерах, сгенерированных вручную. Что произойдёт, если в одном из словарей отсутствует столбец, который был в обучении? А если есть лишний столбец? Проверьте оба сценария.

#### Контрольные вопросы

1. Зачем нужны конвейеры? Назовите не менее трёх причин.
2. Что такое «утечка данных» и как конвейер от неё защищает?
3. В чём разница между `Pipeline` и `ColumnTransformer`? Когда применять каждый?
4. Как внутри `ColumnTransformer` задать, к каким столбцам применять преобразование?
5. Что делает `remainder='drop'`, `'passthrough'`? Когда какой нужен?
6. Зачем нужен `handle_unknown='ignore'` в `OneHotEncoder`?
7. Что такое `FunctionTransformer`? В каких случаях он удобен, а в каких — нет?
8. Зачем наследоваться от `BaseEstimator` и `TransformerMixin` при написании своего трансформера?
9. Почему `__init__` трансформера не должен ничего вычислять? Где должны происходить вычисления?
10. Чем `FeatureUnion` отличается от `ColumnTransformer`? Приведите пример, когда нужен каждый.
11. Как в `GridSearchCV` задать гиперпараметр внутри вложенного `Pipeline`? Приведите пример.
12. Почему важно сохранять весь конвейер целиком, а не только финальную модель?
13. Почему потоковая обработка «по одному объекту» работает медленнее, чем обработка батчами? В чём накладные расходы?
14. Что такое инкрементальное обучение? Какие модели его поддерживают, а какие — нет?
15. В чём отличие пакетной обработки от потоковой? Какие задачи лучше решать каким способом?
16. Что такое concept drift? Почему он важен для потоковых моделей? Как можно с ним бороться?

#### Дополнительные задания

1. **Визуализация конвейера.** `sklearn` умеет строить диаграмму конвейера: `sklearn.utils.estimator_html_repr(pipe)` или `from sklearn import set_config; set_config(display='diagram')`. Постройте диаграмму для вашего конвейера и вставьте в отчёт. Объясните, как на ней читаются вложенные ветви.

2. **Свой конвейер «с нуля».** Реализуйте класс `MyPipeline`, который принимает список шагов `[(name, transformer_or_model), ...]` и реализует `fit`, `predict`, `predict_proba` по аналогии с `sklearn.pipeline.Pipeline`. Не используйте `sklearn.pipeline.Pipeline` внутри. Сравните с библиотечной реализацией — результаты должны совпасть.

3. **Кэширование шагов конвейера.** `Pipeline` поддерживает аргумент `memory=...`: он кэширует результаты `fit_transform` на диск (через `joblib.Memory`). Это ускоряет повторные запуски `GridSearchCV`, если препроцессинг долгий. Продемонстрируйте: используйте `Pipeline(..., memory='./cache')` с `TfidfVectorizer` на большом тексте и покажите, что повторный запуск идёт быстрее.

4. **Свой трансформер для отбора признаков по взаимной информации.** Напишите `MutualInfoSelector(TransformerMixin, BaseEstimator)`, который вычисляет `mutual_info_classif` между признаками и целевой переменной и оставляет `top_k`. Сравните с `SelectKBest(mutual_info_classif)` из `sklearn`. Какой результат совпадает?

5. **Конвейер для multi-label классификации.** Используйте `MultiLabelBinarizer` и `OneVsRestClassifier` внутри `Pipeline`. Постройте конвейер для 20 Newsgroups с несколькими метками (например, объедините близкие темы). Оцените `hamming_loss` и `f1_score(average='micro')`.

6. **Обнаружение concept drift.** На потоке `timeseries_stream.csv` реализуйте простой детектор дрейфа: если среднее значение признака за последние 100 объектов отличается от среднего за предыдущие 100 более чем на 2 стандартных отклонения — считаем, что произошёл дрейф. Постройте график «номер объекта vs индикатор дрейфа». Сопоставьте с известными аномалиями.

7. **Потоковый конвейер с очередью.** Используйте `queue.Queue` и `threading.Thread` для имитации потоковой обработки: один поток «поставляет» данные в очередь (как источник), другой — «забирает» и применяет конвейер. Это простая модель реального продакшен-стриминга. Замерьте пропускную способность (объектов в секунду) для разных размеров батча.

8. **Сравнение с `pyfunc`.** Познакомьтесь с `mlflow.pyfunc`. Оберните свой конвейер в `PythonModel` и сохраните через `mlflow`. Продемонстрируйте, что загруженная модель выдаёт те же предсказания. Объясните, зачем нужна такая обёртка в продакшене (намек: единый интерфейс, версионирование, трекинг экспериментов).
