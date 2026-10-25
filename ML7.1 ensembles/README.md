### Ансамбли моделей

#### Цель работы

На практике познакомиться с механизмом работы разных типов ансамблей моделей машинного обучения. Понять, как голосование, бэггинг, случайный лес, бустинг и стекинг влияют на смещение (bias) и разброс (variance) предсказаний. Реализовать каждый простой ансамбль своими руками, сравнить его с библиотечной реализацией, а затем на реальных данных применить продвинутые промышленные алгоритмы (XGBoost, CatBoost, LightGBM).

#### Подготовка данных

В работе используются три набора данных:

- **Синтетическая регрессия** — небольшая одномерная задача с известной истинной зависимостью. Удобно для визуализации bias/variance и для того, чтобы видеть, как ансамбль «сглаживает» предсказания.
- **Синтетическая классификация** — двумерная задача с нелинейной границей. Показывает, как ансамбли справляются с нелинейными зависимостями.
- **Реальный датасет** — `fetch_california_housing` из `sklearn` для практических заданий.

Датасеты небольшие (200 и 500 точек). Этого достаточно, чтобы быстро обучать десятки моделей в циклах (для бэггинга нам, например, понадобится обучить 50 деревьев).

Загружаем данные:

```python
reg = pd.read_csv('synthetic_regression.csv')
clf = pd.read_csv('synthetic_classification.csv')

X_reg = reg[['x']].values
y_reg = reg['y'].values
y_true_reg = reg['y_true'].values

X_clf = clf[['x1', 'x2']].values
y_clf = clf['y'].values

# Разделение для проверки обобщения
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf)

print('Регрессия:', Xr_tr.shape, Xr_te.shape)
print('Классификация:', Xc_tr.shape, Xc_te.shape)
```

Посмотрим на данные:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(reg['x'], reg['y'], s=20, alpha=0.6, label='Наблюдения')
axes[0].plot(reg['x'].sort_values().values,
             reg.sort_values('x')['y_true'].values,
             'r-', linewidth=2, label='Истинная зависимость')
axes[0].set_title('Синтетическая регрессия')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].legend()

for label, color in [(0, 'tab:blue'), (1, 'tab:red')]:
    m = clf['y'] == label
    axes[1].scatter(clf.loc[m, 'x1'], clf.loc[m, 'x2'],
                    c=color, s=20, alpha=0.7, label=f'Класс {label}')
axes[1].set_title('Синтетическая классификация')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2'); axes[1].legend()

plt.tight_layout(); plt.show()
```

![Исходные данные](url "https://github.com/koroteevmv/ML_course/blob/2025/ML7.1%20ensembles/img/ml71-0.png?raw=true")

#### Методические указания

##### Зачем нужны ансамбли: bias-variance tradeoff

Любая модель ошибается на тестовых данных. Ошибку можно разложить на три компоненты:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\text{Bias}^2[\hat{f}(x)]}_{\text{систематическая ошибка}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{разброс}} + \underbrace{\sigma^2}_{\text{шум}}$$

- **Смещение (bias)** — насколько в среднем предсказания модели отличаются от истины. Высокое смещение — модель слишком простая (например, линейная регрессия для нелинейной зависимости). Это **недообучение**.
- **Разброс (variance)** — насколько сильно предсказания меняются при изменении обучающей выборки. Высокий разброс — модель слишком чувствительна к данным (например, глубокое решающее дерево). Это **переобучение**.

Идея ансамблей:

- **Bagging** (и Random Forest) **уменьшает разброс**, обучая много «независимых» моделей и усредняя их предсказания. Смещение при этом обычно не увеличивается или увеличивается незначительно.
- **Boosting** (AdaBoost, Gradient Boosting) **уменьшает смещение**, последовательно добавляя модели, каждая из которых исправляет ошибки предыдущих. Но при большом числе итераций может **увеличить разброс** и переобучиться.
- **Stacking** комбинирует разные типы моделей, пытаясь взять лучшее от каждой, и в теории может уменьшить и смещение, и разброс.
- **Voting** — самый простой вариант: несколько моделей «голосуют», итоговый ответ — усреднение. Хорошо работает, когда модели разнородные и ошибаются по-разному.

Наглядно посмотрим на это на практике. Ниже мы реализуем каждый ансамбль «руками», а потом сравним с библиотечными реализациями.

##### Voting: простое голосование

**Voting** объединяет предсказания нескольких независимо обученных моделей:

- **Жёсткое голосование (hard voting)** для классификации — каждый базовый классификатор голосует за класс, побеждает класс с наибольшим числом голосов.
- **Мягкое голосование (soft voting)** — усредняются вероятности классов, победитель — класс с наибольшей средней вероятностью. Обычно работает лучше жёсткого, но требует, чтобы все модели умели выдавать вероятности (`predict_proba`).
- **Voting для регрессии** — простое усреднение предсказаний.

Реализуем voting регрессор руками:

```python
class VotingRegressorManual:
    """Усреднение предсказаний нескольких моделей."""
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.array([m.predict(X) for m in self.models])
        return predictions.mean(axis=0)
```

Сравним с библиотечной реализацией:

```python
# Базовые модели — разнородные по типу и сложности
base_models_manual = [
    DecisionTreeRegressor(max_depth=3, random_state=0),
    DecisionTreeRegressor(max_depth=5, random_state=0),
    DecisionTreeRegressor(max_depth=10, random_state=0),
    LinearRegression(),
]

voting_manual = VotingRegressorManual([
    DecisionTreeRegressor(max_depth=3, random_state=0),
    DecisionTreeRegressor(max_depth=5, random_state=0),
    DecisionTreeRegressor(max_depth=10, random_state=0),
    LinearRegression(),
]).fit(Xr_tr, yr_tr)

voting_sklearn = VotingRegressor([
    ('dt3',  DecisionTreeRegressor(max_depth=3, random_state=0)),
    ('dt5',  DecisionTreeRegressor(max_depth=5, random_state=0)),
    ('dt10', DecisionTreeRegressor(max_depth=10, random_state=0)),
    ('lin',  LinearRegression()),
]).fit(Xr_tr, yr_tr)

print('MSE (руками):  ', mean_squared_error(yr_te, voting_manual.predict(Xr_te)))
print('MSE (sklearn): ', mean_squared_error(yr_te, voting_sklearn.predict(Xr_te)))
```

Результаты должны совпасть — библиотечный `VotingRegressor` делает ровно то же самое.

**Когда voting помогает:** когда базовые модели ошибаются по-разному, и их ошибки «гасят» друг друга при усреднении. Voting **не уменьшает** ошибку, если все модели одинаковы.

##### Bagging: bootstrap aggregating

**Bagging** (Bootstrap AGGregating) обучает $N$ копий одной и той же базовой модели на $N$ случайных подвыборках с возвращением (bootstrap samples), а затем усредняет предсказания (для регрессии) или голосует (для классификации).

Ключевая идея: каждая отдельная модель видит «свою» часть данных и переобучается по-своему. Но при усреднении **разброс** (variance) снижается, потому что случайные ошибки отдельных моделей «сглаживаются».

Математически, если $N$ моделей имеют одинаковый разброс $\sigma^2$ и попарно некоррелированы, разброс среднего:

$$\text{Var}\left(\frac{1}{N}\sum_{i=1}^N \hat{f}_i(x)\right) = \frac{\sigma^2}{N}$$

На практике модели коррелированы (видят перекрывающиеся подвыборки), поэтому разброс снижается не в $N$ раз, но всё равно заметно.

Реализация bagging-регрессора:

```python
class BaggingRegressorManual:
    def __init__(self, base_model, n_estimators=50, random_state=42):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        self.models_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True)
            m = clone(self.base_model)
            m.fit(X[idx], y[idx])
            self.models_.append(m)
        return self
    
    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models_])
        return preds.mean(axis=0)
```

Обучим bagging с базовой моделью — глубоким деревом решений (оно переобучается: высокий variance, низкий bias):

```python
deep_tree = DecisionTreeRegressor(max_depth=15, random_state=0)
deep_tree.fit(Xr_tr, yr_tr)
print('Одиночное глубокое дерево, MSE:',
      mean_squared_error(yr_te, deep_tree.predict(Xr_te)))

bagging_manual = BaggingRegressorManual(
    DecisionTreeRegressor(max_depth=15, random_state=0),
    n_estimators=100, random_state=42
).fit(Xr_tr, yr_tr)
print('Bagging (руками), MSE:',
      mean_squared_error(yr_te, bagging_manual.predict(Xr_te)))

bagging_sklearn = BaggingRegressor(
    DecisionTreeRegressor(max_depth=15, random_state=0),
    n_estimators=100, random_state=42, n_jobs=-1
).fit(Xr_tr, yr_tr)
print('Bagging (sklearn), MSE:',
      mean_squared_error(yr_te, bagging_sklearn.predict(Xr_te)))
```

Убедитесь, что bagging даёт заметно меньшую ошибку, чем одно глубокое дерево. Это и есть эффект снижения разброса.

**Визуализация эффекта bagging.** Обучим несколько деревьев на разных bootstrap-подвыборках, нарисуем их предсказания серым и среднее — жирным. Видно, как усреднение «сглаживает» разброс отдельных деревьев.

```python
X_grid = np.linspace(-3, 3, 200).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(12, 5))
rng = np.random.RandomState(0)
n = len(Xr_tr)
for _ in range(30):
    idx = rng.choice(n, size=n, replace=True)
    t = DecisionTreeRegressor(max_depth=8, random_state=0).fit(Xr_tr[idx], yr_tr[idx])
    ax.plot(X_grid, t.predict(X_grid), color='gray', alpha=0.25, linewidth=0.8)

# Среднее по всем моделям (на глаз по тем же 30 моделям)
# Здесь для наглядности обучим bagging ещё раз и нарисуем его предсказание
bag30 = BaggingRegressorManual(
    DecisionTreeRegressor(max_depth=8, random_state=0),
    n_estimators=30, random_state=0).fit(Xr_tr, yr_tr)
ax.plot(X_grid, bag30.predict(X_grid), 'tab:blue', linewidth=2.5,
        label='Bagging (усреднение)')
ax.plot(X_grid, np.sin(X_grid) + 0.5 * X_grid, 'r--', linewidth=2,
        label='Истинная зависимость')
ax.scatter(Xr_tr, yr_tr, s=15, c='k', alpha=0.4, label='Обучающие точки')
ax.set_title('Bagging сглаживает разброс отдельных деревьев')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend()
plt.show()
```

![Усреднение деревьев](url "https://github.com/koroteevmv/ML_course/blob/2025/ML7.1%20ensembles/img/ml71-1.png?raw=true")

##### Random Forest: bagging + случайные признаки

**Random Forest** — это bagging деревьев решений с одним важным дополнением: **на каждом разбиении дерева рассматривается не всё множество признаков, а случайное подмножество** (обычно $\sqrt{p}$ для классификации и $p/3$ для регрессии, где $p$ — число признаков).

Зачем? Bagging сам по себе снижает разброс, но деревья получаются сильно коррелированными, потому что они видят одни и те же данные и часто выбирают одни и те же признаки для разбиений. Если базовые модели коррелированы, усреднение снижает разброс гораздо слабее: $\text{Var} \propto (\rho + \frac{1-\rho}{N})$, где $\rho$ — корреляция между моделями. Рандомизация признаков снижает $\rho$ и делает ансамбль эффективнее.

Реализовать Random Forest «руками» с полной поддержкой рандомизации признаков на каждом узле сложно (это требует модификации самого дерева). Поэтому на практике используют `sklearn.ensemble.RandomForestRegressor`. Но идею можно продемонстрировать: bagging с деревьями, обученными на случайных подмножествах признаков.

```python
class RandomForestRegressorManual:
    """Bagging + случайное подмножество признаков на каждой модели.
    Это упрощённая версия Random Forest: рандомизация по признакам
    происходит на уровне всей модели, а не на каждом разбиении."""
    def __init__(self, base_model, n_estimators=50, max_features=0.7,
                 random_state=42):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
    
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n, p = X.shape
        n_feats = max(1, int(p * self.max_features))
        self.models_ = []
        self.features_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True)
            feats = rng.choice(p, size=n_feats, replace=False)
            m = clone(self.base_model)
            m.fit(X[idx][:, feats], y[idx])
            self.models_.append(m)
            self.features_.append(feats)
        return self
    
    def predict(self, X):
        preds = np.array([
            m.predict(X[:, feats])
            for m, feats in zip(self.models_, self.features_)
        ])
        return preds.mean(axis=0)
```

На нашем одномерном синтетическом датасете рандомизация признаков не имеет смысла (признак один). Поэтому Random Forest по-настоящему раскрывается на многомерных данных — например, на California Housing в самостоятельных заданиях.

Сравните `RandomForestRegressor` с bagging на многомерном датасете — RF обычно немного лучше.

##### AdaBoost: адаптивный бустинг

**AdaBoost** (Adaptive Boosting) — первый практический алгоритм бустинга. Идея: последовательно обучать «слабые» модели (например, пни — деревья глубины 1), причём каждая новая модель фокусируется на тех объектах, которые предыдущие модели плохо классифицировали.

Алгоритм для бинарной классификации ($y \in \{-1, +1\}$):

1. Инициализировать веса объектов: $w_i = 1/n$.
2. Повторить $T$ раз:
   - Обучить базовую модель $h_t$ на данных с весами $w_i$.
   - Вычислить взвешенную ошибку: $\varepsilon_t = \sum_i w_i \cdot \mathbb{1}[h_t(x_i) \ne y_i] / \sum_i w_i$.
   - Вычислить вес модели: $\alpha_t = \frac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}$.
   - Обновить веса объектов: $w_i \leftarrow w_i \cdot e^{-\alpha_t y_i h_t(x_i)}$, затем нормировать.
3. Итоговое предсказание: $\hat{y}(x) = \text{sign}\left(\sum_t \alpha_t h_t(x)\right)$.

Модели с меньшей ошибкой получают больший вес $\alpha_t$. Объекты, которые неправильно классифицированы, получают больший вес и сильнее влияют на следующую модель.

Реализация:

```python
class AdaBoostClassifierManual:
    def __init__(self, base_model, n_estimators=50, learning_rate=1.0):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Приводим метки к {-1, +1}
        y_signed = np.where(y == self.classes_[0], -1, 1)
        n = len(y)
        w = np.ones(n) / n
        self.models_ = []
        self.alphas_ = []
        for _ in range(self.n_estimators):
            m = clone(self.base_model)
            m.fit(X, y, sample_weight=w)
            pred = m.predict(X)
            pred_signed = np.where(pred == self.classes_[0], -1, 1)
            err = np.sum(w * (pred_signed != y_signed)) / np.sum(w)
            if err >= 0.5:
                break
            alpha = self.learning_rate * 0.5 * np.log(
                (1 - err + 1e-10) / (err + 1e-10))
            w = w * np.exp(-alpha * y_signed * pred_signed)
            w = w / w.sum()
            self.models_.append(m)
            self.alphas_.append(alpha)
        return self
    
    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models_])
        preds_signed = np.where(preds == self.classes_[0], -1, 1)
        result = (preds_signed * np.array(self.alphas_)[:, None]).sum(axis=0)
        return np.where(result >= 0, self.classes_[1], self.classes_[0])
```

В качестве базовой модели используем «пни» — деревья глубины 1:

```python
stump = DecisionTreeClassifier(max_depth=1, random_state=0)

ada_manual = AdaBoostClassifierManual(
    DecisionTreeClassifier(max_depth=1, random_state=0),
    n_estimators=50, learning_rate=1.0
).fit(Xc_tr, yc_tr)

ada_sklearn = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
    n_estimators=50, learning_rate=1.0, random_state=0
).fit(Xc_tr, yc_tr)

print('Accuracy (руками): ', accuracy_score(yc_te, ada_manual.predict(Xc_te)))
print('Accuracy (sklearn):', accuracy_score(yc_te, ada_sklearn.predict(Xc_te)))
```

Результаты должны быть близки. Небольшое расхождение может быть из-за деталей реализации (например, sklearn использует `SAMME.R` — модификацию для моделей с вероятностями).

**Как AdaBoost уменьшает bias.** Каждая новая модель корректирует ошибки предыдущих. В итоге ансамбль описывает более сложную границу, чем одна модель. Это видно визуально:

```python
def plot_decision_boundary(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=15, cmap='coolwarm', edgecolor='k')
    ax.set_title(title)

# Сравним 1 пень vs AdaBoost
single_stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xc_tr, yc_tr)
ada_10 = AdaBoostClassifierManual(
    DecisionTreeClassifier(max_depth=1, random_state=0),
    n_estimators=10).fit(Xc_tr, yc_tr)
ada_50 = AdaBoostClassifierManual(
    DecisionTreeClassifier(max_depth=1, random_state=0),
    n_estimators=50).fit(Xc_tr, yc_tr)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_decision_boundary(axes[0], single_stump, Xc_tr, yc_tr, 'Один пень (bias высокий)')
plot_decision_boundary(axes[1], ada_10, Xc_tr, yc_tr, 'AdaBoost, 10 моделей')
plot_decision_boundary(axes[2], ada_50, Xc_tr, yc_tr, 'AdaBoost, 50 моделей')
plt.tight_layout(); plt.show()
```

С ростом числа моделей граница становится всё более сложной и лучше описывает данные.

![Адаптивный бустинг](url "https://github.com/koroteevmv/ML_course/blob/2025/ML7.1%20ensembles/img/ml71-2.png?raw=true")

##### Gradient Boosting: бустинг на градиентах

**Gradient Boosting** — обобщение AdaBoost. Ключевая идея: каждая новая модель обучается **на остатках** (residuals) предыдущего ансамбля, то есть предсказывает то, что ансамбль пока не смог объяснить.

Для задачи регрессии с квадратичной ошибкой алгоритм выглядит так:

1. Инициализировать ансамбль константой: $F_0(x) = \bar{y}$.
2. Повторить $T$ раз:
   - Вычислить остатки: $r_i = y_i - F_{t-1}(x_i)$.
   - Обучить базовую модель $h_t$ на остатках: $h_t \approx r$.
   - Добавить её в ансамбль с шагом $\eta$ (learning rate): $F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$.

С точки зрения оптимизации, $r_i = -\partial \mathcal{L} / \partial F$ — это антиградиент функции потерь по предсказанию, поэтому метод и называется градиентным бустингом. Обобщение на другие функции потерь (MAE, Huber) и на классификацию — просто замена остатков на соответствующий антиградиент.

Реализация:

```python
class GradientBoostingRegressorManual:
    def __init__(self, base_model, n_estimators=100, learning_rate=0.1):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        self.init_ = y.mean()
        F = np.full(len(y), self.init_)
        self.models_ = []
        for _ in range(self.n_estimators):
            residual = y - F
            m = clone(self.base_model)
            m.fit(X, residual)
            F = F + self.learning_rate * m.predict(X)
            self.models_.append(m)
        return self
    
    def predict(self, X):
        F = np.full(len(X), self.init_)
        for m in self.models_:
            F = F + self.learning_rate * m.predict(X)
        return F
```

Сравним с `sklearn`:

```python
gb_manual = GradientBoostingRegressorManual(
    DecisionTreeRegressor(max_depth=3, random_state=0),
    n_estimators=100, learning_rate=0.1
).fit(Xr_tr, yr_tr)

gb_sklearn = GradientBoostingRegressor(
    max_depth=3, n_estimators=100, learning_rate=0.1, random_state=0
).fit(Xr_tr, yr_tr)

print('MSE (руками):  ', mean_squared_error(yr_te, gb_manual.predict(Xr_te)))
print('MSE (sklearn): ', mean_squared_error(yr_te, gb_sklearn.predict(Xr_te)))
```

**Как Gradient Boosting уменьшает bias.** Каждая новая модель «дотягивает» предсказание до истины. Проследим это по кривой обучения — ошибка на обучающей выборке падает с числом итераций.

```python
train_mse = []
test_mse = []
for n_est in range(1, 101, 5):
    gb = GradientBoostingRegressorManual(
        DecisionTreeRegressor(max_depth=3, random_state=0),
        n_estimators=n_est, learning_rate=0.1
    ).fit(Xr_tr, yr_tr)
    train_mse.append(mean_squared_error(yr_tr, gb.predict(Xr_tr)))
    test_mse.append(mean_squared_error(yr_te, gb.predict(Xr_te)))

plt.plot(range(1, 101, 5), train_mse, 'o-', label='Train MSE')
plt.plot(range(1, 101, 5), test_mse, 's-', label='Test MSE')
plt.xlabel('Количество итераций бустинга')
plt.ylabel('MSE')
plt.title('Gradient Boosting: ошибка падает с числом итераций')
plt.legend()
plt.show()
```

Кривая Test MSE сначала падает (bias уменьшается), затем выходит на плато, а при очень большом числе итераций может начать расти (variance увеличивается, переобучение). Оптимальное число итераций подбирается по кросс-валидации.

![Кривая обучения градиентного бустинга](url "https://github.com/koroteevmv/ML_course/blob/2025/ML7.1%20ensembles/img/ml71-3.png?raw=true")

##### Stacking: мета-обучение

**Stacking** (stacked generalization) — техника, при которой предсказания нескольких базовых моделей подаются на вход **мета-модели**, которая учится их правильно комбинировать.

Алгоритм:

1. Разбить данные на $K$ фолдов (кросс-валидация).
2. Для каждого фолда обучить базовые модели на $K-1$ фолдах, сделать предсказания на оставшемся. Так получаются **out-of-fold (OOF) предсказания** — каждый объект предсказан моделью, которая его не видела.
3. Обучить мета-модель на OOF-предсказаниях базовых моделей.
4. Переобучить базовые модели на всех данных (чтобы потом делать предсказания на новых объектах).

OOF-предсказания нужны, чтобы избежать утечки информации: если обучать мета-модель на предсказаниях, сделанных на тех же данных, на которых обучались базовые модели, мета-модель «переоценит» их качество.

Реализация:

```python
class StackingRegressorManual:
    def __init__(self, base_models, meta_model, n_folds=5, random_state=42):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.random_state = random_state
    
    def fit(self, X, y):
        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)
        oof = np.zeros((len(y), len(self.base_models)))
        for tr_idx, va_idx in kf.split(X):
            for j, model in enumerate(self.base_models):
                m = clone(model)
                m.fit(X[tr_idx], y[tr_idx])
                oof[va_idx, j] = m.predict(X[va_idx])
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(oof, y)
        # Переобучаем базовые модели на всех данных
        self.models_ = []
        for model in self.base_models:
            m = clone(model)
            m.fit(X, y)
            self.models_.append(m)
        return self
    
    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models_])
        return self.meta_model_.predict(preds)
```

Базовые модели — разнородные, чтобы их предсказания были «разными взглядами» на данные. Мета-модель — простая (например, линейная регрессия), чтобы не переобучиться на предсказаниях.

```python
stack_manual = StackingRegressorManual(
    base_models=[
        DecisionTreeRegressor(max_depth=3, random_state=0),
        DecisionTreeRegressor(max_depth=10, random_state=0),
        LinearRegression(),
    ],
    meta_model=LinearRegression(),
    n_folds=5
).fit(Xr_tr, yr_tr)

print('MSE (stacking руками):',
      mean_squared_error(yr_te, stack_manual.predict(Xr_te)))
```

Библиотечная реализация:

```python
stack_sklearn = StackingRegressor(
    estimators=[
        ('dt3', DecisionTreeRegressor(max_depth=3, random_state=0)),
        ('dt10', DecisionTreeRegressor(max_depth=10, random_state=0)),
        ('lin', LinearRegression()),
    ],
    final_estimator=LinearRegression(),
    cv=5
).fit(Xr_tr, yr_tr)

print('MSE (stacking sklearn):',
      mean_squared_error(yr_te, stack_sklearn.predict(Xr_te)))
```

Stacking обычно даёт небольшое, но стабильное улучшение по сравнению с лучшей из базовых моделей. Плата — время обучения (нужно обучить $N \times K + N$ моделей).

##### Сравнение: как ансамбли влияют на bias и variance

Соберём все модели в одну таблицу и сравним их ошибку на обучающей и тестовой выборках. Разница между train и test error — это индикатор variance, а высокий test error при низком train error — признак переобучения.

```python
models = {
    'Одиночное глубокое дерево (depth=15)': DecisionTreeRegressor(
        max_depth=15, random_state=0),
    'Одиночное мелкое дерево (depth=3)': DecisionTreeRegressor(
        max_depth=3, random_state=0),
    'Bagging (100 деревьев depth=15)': BaggingRegressorManual(
        DecisionTreeRegressor(max_depth=15, random_state=0),
        n_estimators=100, random_state=42),
    'AdaBoost-регрессор (пни, 100)': AdaBoostClassifierManual(
        DecisionTreeClassifier(max_depth=1, random_state=0),
        n_estimators=100),  # для регрессии не подходит, только для классификации
}

# Для регрессии соберём список отдельно
reg_models = {
    'Мелкое дерево (depth=3)': DecisionTreeRegressor(max_depth=3, random_state=0),
    'Глубокое дерево (depth=15)': DecisionTreeRegressor(max_depth=15, random_state=0),
    'Bagging (100 × depth=15)': BaggingRegressorManual(
        DecisionTreeRegressor(max_depth=15, random_state=0),
        n_estimators=100, random_state=42),
    'Gradient Boosting (100 × depth=3)': GradientBoostingRegressorManual(
        DecisionTreeRegressor(max_depth=3, random_state=0),
        n_estimators=100, learning_rate=0.1),
    'Stacking (dt3 + dt15 + lin)': StackingRegressorManual(
        base_models=[
            DecisionTreeRegressor(max_depth=3, random_state=0),
            DecisionTreeRegressor(max_depth=15, random_state=0),
            LinearRegression(),
        ],
        meta_model=LinearRegression(), n_folds=5),
}

rows = []
for name, model in reg_models.items():
    model.fit(Xr_tr, yr_tr)
    train_mse = mean_squared_error(yr_tr, model.predict(Xr_tr))
    test_mse = mean_squared_error(yr_te, model.predict(Xr_te))
    rows.append({'Модель': name,
                 'Train MSE': round(train_mse, 3),
                 'Test MSE': round(test_mse, 3),
                 'Разница': round(test_mse - train_mse, 3)})

pd.DataFrame(rows).set_index('Модель')
```

Типичная картина:

| Модель | Train MSE | Test MSE | Разница |
|---|---|---|---|
| Мелкое дерево (depth=3) | высокий | высокий | маленькая (**высокий bias**) |
| Глубокое дерево (depth=15) | очень низкий | высокий | большая (**высокий variance**) |
| Bagging (100 × depth=15) | низкий | **ниже**, чем у глубокого | умеренная (**variance снижен**) |
| Gradient Boosting | низкий | **низкий** | умеренная (**bias снижен**) |
| Stacking | низкий | **низкий** | умеренная (**bias и variance сбалансированы**) |

**Выводы:**
- **Одиночное мелкое дерево** — высокий bias, недообучение.
- **Одиночное глубокое дерево** — низкий bias, но огромный variance: разница train/test максимальна.
- **Bagging** снижает test error за счёт усреднения — variance уменьшается, bias остаётся на уровне базовой модели.
- **Boosting** снижает test error за счёт последовательного исправления ошибок — bias уменьшается, но при большом числе итераций может вырасти variance (переобучение).
- **Stacking** комбинирует сильные стороны разных моделей, давая стабильно низкую ошибку.

##### Что выбрать: практические рекомендации

- **Табличные данные, много признаков, нелинейные зависимости** — Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost).
- **Нужны интерпретируемость и скорость** — Random Forest (можно смотреть feature importance), либо ограниченный бустинг с небольшим числом итераций.
- **Много шума в данных** — Bagging и Random Forest (устойчивы к шуму), а не AdaBoost (он фокусируется на шумных объектах и переобучается).
- **Есть разнородные модели** — Stacking.
- **Нужен быстрый baseline** — Voting из 2–3 простых моделей.
- **Очень большой датасет (>10⁶ строк)** — LightGBM или XGBoost (по скорости).
- **Категориальные признаки с большим числом категорий** — CatBoost (из коробки работает с категориями).

#### Задания для самостоятельного выполнения

Все задания выполняются в ноутбуке. Для каждого пункта нужен код + краткий текстовый вывод.

##### Часть 1. Продвинутые ансамбли (XGBoost, CatBoost, LightGBM)

Установите библиотеки (если ещё не установлены):

```bash
pip install xgboost catboost lightgbm
```

1. **Знакомство с API.** Загрузите `fetch_california_housing` из `sklearn` и разделите его на train/test (70/30). Обучите по одной модели каждого из трёх фреймворков в режиме регрессии с параметрами по умолчанию:
   - `XGBRegressor(n_estimators=100)`,
   - `LGBMRegressor(n_estimators=100)`,
   - `CatBoostRegressor(n_estimators=100, verbose=0)`.
   
   Сравните время обучения и RMSE на тестовой выборке. Обратите внимание на разные интерфейсы (например, `eval_metric` у XGBoost и `loss_function` у CatBoost).

2. **Сравнение с Random Forest и Gradient Boosting из sklearn.** Добавьте в таблицу из пункта 1 модели `RandomForestRegressor` и `GradientBoostingRegressor` с сопоставимым числом деревьев. Сравните RMSE и время. Сделайте вывод, какие модели выигрывают по скорости, а какие — по качеству.

3. **Подбор гиперпараметров.** Выберите одну модель (например, LightGBM) и подберите гиперпараметры `n_estimators`, `learning_rate`, `num_leaves`/`max_depth` с помощью `GridSearchCV` или `RandomizedSearchCV` и кросс-валидации по 5 фолдам. Сравните качество с дефолтной моделью. Насколько удалось улучшить RMSE?

4. **Early stopping.** Используя LightGBM или XGBoost, реализуйте early stopping на валидационной выборке (передайте `eval_set` в `fit`). Постройте кривые обучения (train и validation RMSE от числа итераций). При каком числе итераций модель перестаёт улучшаться? Согласуется ли это с вашим пониманием бустинга?

5. **Feature importance.** Для одной из бустинговых моделей выведите и визуализируйте важность признаков (`feature_importances_` или `get_feature_importance()` у CatBoost). Какие признаки оказались самыми важными? Совпадает ли их порядок между XGBoost, LightGBM и CatBoost? Сделайте вывод об устойчивости важности.

6. **Категориальные признаки с CatBoost.** Загрузите датасет Titanic (`sns.load_dataset('titanic')`), почистите его от явных пропусков и передайте категориальные столбцы (`Sex`, `Embarked`, `Pclass`) в CatBoost как `cat_features`. Сравните с вариантом, где категориальные признаки закодированы вручную через `OneHotEncoder`. Что даёт CatBoost «из коробки»?

##### Часть 2. Понимание механизма ансамблей

7. **Bagging и variance.** На синтетической регрессии обучите 100 деревьев глубины 5 на bootstrap-подвыборках. Для сетки точек `X_grid` постройте доверительный интервал: среднее предсказание ± стандартное отклонение по 100 моделям. Сравните ширину интервала (variance) для одного дерева, bagging из 10 деревьев и bagging из 100 деревьев. Постройте график: ширина интервала vs число деревьев в bagging.

8. **Boosting и bias.** На той же синтетической регрессии обучите Gradient Boosting с числом итераций 1, 5, 20, 100, 500. Для каждого варианта постройте предсказание на `X_grid` и наложите на график вместе с истинной зависимостью и обучающими точками. Наглядно покажите, что с ростом числа итераций ансамбль всё точнее приближает истину (bias падает).

9. **AdaBoost и выбросы.** На синтетической классификации добавьте 10 «выбросов»: точки одного класса, находящиеся глубоко в области другого класса. Сравните поведение AdaBoost на исходных данных и на данных с выбросами (число моделей = 50). Визуализируйте границы решений. Сделайте вывод, почему AdaBoost чувствителен к выбросам.

10. **Stacking vs Voting.** На `California Housing` соберите два ансамбля с одинаковым набором базовых моделей (например, DecisionTree, RandomForest, LinearRegression):
    - `VotingRegressor` (усреднение),
    - `StackingRegressor` с линейной мета-моделью.
    
    Сравните RMSE. В каком случае stacking выигрывает? Почему?

##### Часть 3. Сравнительный анализ

11. **Сводная таблица.** На одном и том же датасете (например, `California Housing`) обучите:
    - одиночное дерево,
    - Random Forest,
    - Bagging,
    - Gradient Boosting (sklearn),
    - XGBoost,
    - LightGBM,
    - CatBoost.
    
    Для каждой модели подберите разумные гиперпараметры (не обязательно оптимальные — главное, чтобы они были сопоставимы). Заполните таблицу: время обучения, RMSE на train, RMSE на test, RMSE при 5-fold кросс-валидации. Сделайте вывод: какие модели лучше по качеству, какие — по скорости, какие — самые стабильные.

12. **Классификация.** Повторите пункт 11, но на задаче классификации. Подойдёт встроенный `load_breast_cancer` или `fetch_covtype` (последний большой — можно взять подвыборку). Используйте метрики `accuracy`, `f1_score`, `roc_auc_score`. Оцените также время обучения.

13. **Продвинутые техники.** (по желанию) Изучите и продемонстрируйте:
    - **Stacking с бустингами** — мета-модель поверх XGBoost, LightGBM и CatBoost;
    - **Blending** — разновидность stacking с выделенной валидационной выборкой вместо OOF;
    - **Multi-level stacking** — несколько уровней мета-моделей.
    
    Сравните с одноуровневым stacking.

##### Часть 4. Bias-variance decomposition

14. **Экспериментальная оценка bias и variance.** Реализуйте функцию, которая по заданной фабрике моделей оценивает bias² и variance предсказаний:
    
    ```python
    def bias_variance(model_factory, X, y, X_eval, y_true_eval, n_bootstrap=100):
        preds = []
        for i in range(n_bootstrap):
            rng = np.random.RandomState(i)
            idx = rng.choice(len(X), size=len(X), replace=True)
            m = model_factory()
            m.fit(X[idx], y[idx])
            preds.append(m.predict(X_eval))
        preds = np.array(preds)
        mean_pred = preds.mean(axis=0)
        bias2 = np.mean((mean_pred - y_true_eval)**2)
        variance = np.mean(preds.var(axis=0))
        return bias2, variance, bias2 + variance
    ```
    
    Примените её к:
    - одному дереву глубины 15,
    - bagging из 50 деревьев глубины 15,
    - gradient boosting с 50 итерациями (глубина 3).
    
    Сравните компоненты. Убедитесь, что bagging снижает variance, boosting — bias.

#### Контрольные вопросы

1. Чем отличается voting от bagging? Когда применяется каждый?
2. В чём разница между жёстким и мягким голосованием? Почему мягкое обычно точнее?
3. Как bagging снижает variance? Почему снижение не такое сильное, как $\sigma^2/N$?
4. Зачем Random Forest добавляет рандомизацию признаков поверх bagging? Что произойдёт, если её убрать?
5. Как работает AdaBoost? Почему он фокусируется на «сложных» объектах?
6. Чем AdaBoost отличается от Gradient Boosting? В чём преимущества второго?
7. Что такое learning rate в бустинге? Как он связан с числом итераций?
8. Почему бустинг может переобучиться, а bagging — нет?
9. В чём идея stacking? Зачем нужны out-of-fold предсказания?
10. Почему мета-модель в stacking должна быть простой?
11. Как вы диагностируете, что модель недообучена, а как — что переобучена, по значениям train и test ошибок?
12. Почему AdaBoost чувствителен к выбросам? Как можно снизить его чувствительность?
13. Какие ансамбли лучше подходят для больших датасетов? Почему?
14. Что такое feature importance в деревьях и ансамблях? Чем она полезна и в чём её ограничения?

#### Дополнительные задания

1. **Ручной XGBoost.** Реализуйте упрощённую версию градиентного бустинга с регуляризацией листьев (XGBoost использует формулу $\frac{G}{H + \lambda}$ для значения листа, где $G$ и $H$ — сумма градиентов и вторых производных в листе). Сравните с обычным gradient boosting.

2. **Бэггинг для регрессии с OOB-оценкой.** В bagging каждый bootstrap-набор оставляет около 37% объектов вне обучающей подвыборки (out-of-bag). Реализуйте OOB-оценку: для каждого объекта усредните предсказания только тех моделей, для которых он был OOB. Сравните с ошибкой на отложенной выборке.

3. **Stacking с разными мета-моделями.** Сравните линейную регрессию, гребневую регрессию и очень мелкое дерево как мета-модели. Какая мета-модель даёт лучший результат? Почему?

4. **CatBoost с нестандартными потерями.** Обучите CatBoost на задаче регрессии с функцией потерь MAE (`loss_function='MAE'`) и сравните с RMSE. Как меняются предсказания?

5. **Влияние гиперпараметров LightGBM.** Исследуйте влияние `num_leaves`, `min_child_samples`, `feature_fraction`, `bagging_fraction` на качество. Для каждого параметра постройте график зависимости RMSE от значения (на фиксированном остальном наборе параметров). Сделайте вывод, какие параметры критичны.

6. **Сравнение с простыми моделями.** На одном и том же датасете сравните лучший из ваших ансамблей с простыми моделями: линейной регрессией, одиночным деревом, KNN. Насколько ансамбль лучше? Оцените выигрыш в RMSE и во времени обучения.

7. **Онлайн-обучение.** XGBoost и LightGBM поддерживают инкрементальное обучение (`xgb_model` в `fit`, `init_model` в `lightgbm`). Продемонстрируйте: обучите модель на первых 50% данных, затем дообучите на следующих 50% и сравните с моделью, обученной сразу на всех данных.
