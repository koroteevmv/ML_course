### Полиномиальные модели

#### Цель работы

Познакомиться с полиномиальными моделями: понять, как из линейной регрессии и логистической регрессии «получаются» нелинейные зависимости за счёт добавления полиномиальных признаков, как степень полинома влияет на качество и переобучение, и как с этим бороться с помощью регуляризации. Освоить `PolynomialFeatures` из `sklearn` и научиться подбирать степень полинома, опираясь на графики и метрики.

#### Подготовка данных

Работаем с двумя небольшими синтетическими датасетами. Их преимущество — мы знаем истинную зависимость и можем визуально оценить, насколько хорошо каждая модель её восстанавливает.

- **`poly_reg.csv`** — нелинейная регрессия: $y = 0.5 x^3 - x^2 + 2 + \varepsilon$.
- **`poly_clf.csv`** — бинарная классификация в форме «двух лун» (`make_moons`): классы линейно неразделимы.

Скрипт ниже создаёт оба файла. Запускать достаточно один раз.

```python
# prepare_data.py
import numpy as np
import pandas as pd


def generate_regression(n=200, noise=1.0, random_state=42,
                        path='poly_reg.csv'):
    """Кубическая зависимость + шум."""
    rng = np.random.RandomState(random_state)
    x = rng.uniform(-3, 3, n)
    y_true = 0.5 * x ** 3 - x ** 2 + 2
    y = y_true + rng.randn(n) * noise
    df = pd.DataFrame({'x': x, 'y': y, 'y_true': y_true})
    df = df.sort_values('x').reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f'{path}: {df.shape}')
    return df


def generate_classification(n=400, noise=0.20, random_state=42,
                             path='poly_clf.csv'):
    """Две луны — линейно неразделимые классы."""
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n, noise=noise, random_state=random_state)
    df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y})
    df.to_csv(path, index=False)
    print(f'{path}: {df.shape}, классы: {df.y.value_counts().to_dict()}')
    return df


if __name__ == '__main__':
    generate_regression()
    generate_classification()
```

Библиотеки:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Загружаем данные:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

reg = pd.read_csv('poly_reg.csv')
clf = pd.read_csv('poly_clf.csv')

print('Регрессия:', reg.shape)
print('Классификация:', clf.shape)
```

Посмотрим на регрессионные данные:

```python
plt.figure(figsize=(10, 5))
plt.scatter(reg.x, reg.y, s=20, alpha=0.6, label='Наблюдения')
plt.plot(reg.x, reg.y_true, 'r-', linewidth=2, label='Истинная зависимость')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Нелинейная зависимость: y = 0.5·x³ − x² + 2')
plt.legend()
plt.grid(True)
plt.show()
```

Зависимость явно **нелинейная**: обычная линейная регрессия её не опишет. Именно для таких случаев нужны полиномиальные признаки.

И на классификационные:

```python
plt.figure(figsize=(8, 6))
plt.scatter(clf.x1[clf.y == 0], clf.x2[clf.y == 0],
            c='tab:blue', s=30, alpha=0.7, label='Класс 0')
plt.scatter(clf.x1[clf.y == 1], clf.x2[clf.y == 1],
            c='tab:red', s=30, alpha=0.7, marker='x', label='Класс 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Две луны: линейно неразделимые классы')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
```

Прямой линией эти классы не разделить: нужна кривая, а значит, и логистической регрессии понадобятся нелинейные признаки.

Разделим данные на train/test:

```python
from sklearn.model_selection import train_test_split

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    reg[['x']], reg['y'], test_size=0.3, random_state=42
)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    clf[['x1', 'x2']], clf['y'], test_size=0.3,
    random_state=42, stratify=clf['y']
)

print(Xr_tr.shape, Xr_te.shape)
print(Xc_tr.shape, Xc_te.shape)
```

#### Методические указания

##### Идея полиномиальной регрессии

**Линейная регрессия** предполагает, что $y$ линейно зависит от признаков:

$$\hat{y} = w_0 + w_1 x$$

Если же истинная зависимость квадратичная или кубическая, прямая линия плохо опишет данные. Но ключевая идея: **линейная по коэффициентам, но нелинейная по исходному $x$** зависимость — это всё ещё «линейная регрессия» с точки зрения оптимизации. Например:

$$\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3$$

Это **линейная модель относительно $w$** — мы просто добавили новые признаки $x^2, x^3, \dots$ и обучили на них обычную линейную регрессию. Никакого нового алгоритма не нужно.

То же самое работает и для нескольких исходных признаков. Если у нас два признака $x_1, x_2$, то полиномиальные признаки степени 2 — это:

$$x_1, x_2, x_1^2, x_1 x_2, x_2^2$$

а степени 3 — к ним добавляются ещё $x_1^3, x_1^2 x_2, x_1 x_2^2, x_2^3$ и т.д. Количество признаков растёт как $O(p^d)$, где $p$ — число исходных, $d$ — степень.

> **Замечание.** Полиномиальные признаки — это самый простой способ дать линейной модели «увидеть» нелинейность и взаимодействия между признаками. Дальше в курсе будут более мощные модели (SVM с ядром, деревья, ансамбли), но полиномиальные признаки остаются полезными и в них.

##### `PolynomialFeatures`

`sklearn.preprocessing.PolynomialFeatures` автоматически строит все полиномиальные признаки до заданной степени:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(Xr_tr[['x']])
print(f'Один признак, степень 2 → {X_poly.shape[1]} признаков:',
      poly.get_feature_names_out(['x']))

poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly3.fit_transform(Xr_tr[['x']])
print(f'Один признак, степень 3 → {X_poly3.shape[1]} признаков:',
      poly3.get_feature_names_out(['x']))
```

**Параметры:**

- `degree` — максимальная степень.
- `include_bias` — добавлять ли столбец единиц. По умолчанию `True`. Обычно ставят `False`, потому что линейная регрессия и так добавляет intercept.
- `interaction_only=True` — оставить только «смешанные» признаки (без чистых $x_i^2$, $x_i^3$). Полезно, если нужны только взаимодействия.

**Важно:** `PolynomialFeatures` нужно обучать (`fit`) **только на train**, а к test применять `transform`. Иначе получится утечка данных — формально признаки те же, но лучше всегда следовать правилу «fit на train, transform на test».

В `sklearn` это удобно делать через `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('linreg', LinearRegression()),
])
pipe.fit(Xr_tr, yr_tr)
```

##### Полиномиальная регрессия: степени 1, 2, 3, 10

Построим модели разной степени и посмотрим, как они описывают зависимость:

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

degrees = [1, 2, 3, 10]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
X_grid = np.linspace(-3, 3, 300).reshape(-1, 1)

for ax, d in zip(axes.ravel(), degrees):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('lin', LinearRegression()),
    ])
    pipe.fit(Xr_tr, yr_tr)

    r2_tr = r2_score(yr_tr, pipe.predict(Xr_tr))
    r2_te = r2_score(yr_te, pipe.predict(Xr_te))

    ax.scatter(Xr_tr['x'], yr_tr, s=15, alpha=0.5, label='train')
    ax.scatter(Xr_te['x'], yr_te, s=15, alpha=0.5, marker='^', label='test')
    ax.plot(X_grid, pipe.predict(pd.DataFrame({'x': X_grid.ravel()})),
            'r-', linewidth=2, label=f'Степень {d}')
    ax.set_title(f'Степень {d}: R²(train)={r2_tr:.3f}, R²(test)={r2_te:.3f}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(); ax.grid(True)

plt.suptitle('Полиномиальная регрессия разных степеней', fontsize=14)
plt.tight_layout(); plt.show()
```

**Что вы увидите:**

- **Степень 1:** прямая линия. Плохо описывает и train, и test — это **недообучение**.
- **Степень 2:** парабола, уже ближе к истине, но форма истинной зависимости кубическая, поэтому всё ещё не идеально.
- **Степень 3:** почти идеально повторяет истинную зависимость. Хороший R² и на train, и на test.
- **Степень 10:** кривая «извивается», ловит шум. R² на train почти 1, а на test — может быть хуже, чем у степени 3. Это **переобучение**.

##### Степень и переобучение

Построим график «R² от степени полинома» для train и test — он наглядно показывает компромисс между сложностью и обобщением:

```python
degrees = range(1, 16)
r2_train, r2_test = [], []

for d in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('lin', LinearRegression()),
    ]).fit(Xr_tr, yr_tr)
    r2_train.append(r2_score(yr_tr, pipe.predict(Xr_tr)))
    r2_test.append(r2_score(yr_te, pipe.predict(Xr_te)))

plt.figure(figsize=(10, 5))
plt.plot(list(degrees), r2_train, 'o-', label='R² (train)')
plt.plot(list(degrees), r2_test, 's-', label='R² (test)')
plt.xlabel('Степень полинома')
plt.ylabel('R²')
plt.title('Степень полинома и переобучение')
plt.legend()
plt.grid(True)
plt.xticks(list(degrees))
plt.show()
```

**Как читать график:**

- На малых степенях (1–2) обе кривые низко — **недообучение**.
- На средних (3–5) обе кривые высоко и близко друг к другу — **хорошая модель**.
- На больших (≥ 8) train R² продолжает расти (или уже 1.0), а test — падает или «дёргается» — **переобучение**.

Оптимальная степень — там, где test R² максимален, но при этом train и test ещё не разошлись сильно. Обратите внимание: **test R² может колебаться не монотонно**, поэтому «лучшая степень» — это не всегда та, где тестовая метрика формально выше на одну тысячную. Обычно выбирают **самую простую модель из близких к лучшей** (правило «один стандартный шаг»).

> **Помните.** Высокая степень полинома — это не «лучше», а «сложнее». Для задачи с $n$ точками и степенью $d$ эффективное число параметров растёт как $O(d)$, и при $d \approx n$ модель может «пройти через все точки» — но на новых данных она будет плохой.

##### Регуляризация: Ridge для высоких степеней

Если очень хочется использовать высокую степень (или если данных мало), но при этом не переобучаться, помогает **регуляризация**. Мы уже видели её в лабораторных про регрессию; здесь она работает так же.

**Ridge-регрессия** (L2-регуляризация) добавляет к функции ошибки штраф за большие коэффициенты:

$$\text{Loss} = \text{MSE} + \alpha \sum_j w_j^2$$

Это заставляет коэффициенты «сжиматься» к нулю и делает модель проще. Параметр $\alpha$ регулирует силу штрафа: чем больше $\alpha$, тем проще модель.

```python
from sklearn.linear_model import Ridge

degrees = [10]
alphas = [0.001, 0.1, 10, 1000]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, a in zip(axes.ravel(), alphas):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=10, include_bias=False)),
        ('ridge', Ridge(alpha=a)),
    ]).fit(Xr_tr, yr_tr)

    r2_tr = r2_score(yr_tr, pipe.predict(Xr_tr))
    r2_te = r2_score(yr_te, pipe.predict(Xr_te))

    ax.scatter(Xr_tr['x'], yr_tr, s=15, alpha=0.5, label='train')
    ax.scatter(Xr_te['x'], yr_te, s=15, alpha=0.5, marker='^', label='test')
    ax.plot(X_grid, pipe.predict(pd.DataFrame({'x': X_grid.ravel()})),
            'r-', linewidth=2)
    ax.set_title(f'degree=10, α={a}: R²(train)={r2_tr:.3f}, R²(test)={r2_te:.3f}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(); ax.grid(True)

plt.suptitle('Ridge-регуляризация для полинома степени 10', fontsize=14)
plt.tight_layout(); plt.show()
```

**Что видно:**

- При малом $\alpha$ (0.001) модель практически не регуляризована — «извивается» как обычная степень 10.
- При $\alpha \approx 0.1$–$10$ коэффициенты сжимаются, кривая становится гладкой и по форме близка к истинной зависимости.
- При очень большом $\alpha$ (1000) модель становится слишком простой — все коэффициенты «прижаты» к нулю, и кривая вырождается почти в горизонтальную линию (недообучение).

Оптимальное $\alpha$ находят через кросс-валидацию (`RidgeCV`, `GridSearchCV`). Это подробно разбиралось в лабораторных про регрессию и кросс-валидацию — здесь просто напомним, что регуляризация — **стандартный инструмент** при работе с полиномиальными признаками высокой степени.

##### Полиномиальные признаки в классификации

С тем же успехом полиномиальные признаки можно применять и к логистической регрессии. Тогда граница решения (линейная по признакам) в исходном пространстве становится **нелинейной**.

На данных «две луны» линейная логистическая регрессия не справится. Добавим полиномиальные признаки — и граница станет кривой.

```python
from sklearn.linear_model import LogisticRegression

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_boundary(ax, pipe, X, y, title):
    x_min, x_max = X['x1'].min() - 0.5, X['x1'].max() + 0.5
    y_min, y_max = X['x2'].min() - 0.5, X['x2'].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = pd.DataFrame({'x1': xx.ravel(), 'x2': yy.ravel()})
    Z = pipe.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X['x1'][y == 0], X['x2'][y == 0],
               c='tab:blue', s=20, alpha=0.7, label='Класс 0')
    ax.scatter(X['x1'][y == 1], X['x2'][y == 1],
               c='tab:red', s=20, alpha=0.7, marker='x', label='Класс 1')
    acc_te = pipe.score(Xc_te, yc_te)
    ax.set_title(f'{title}\nAccuracy (test) = {acc_te:.3f}')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.legend(loc='best', fontsize=9)

for ax, degree in zip(axes, [1, 2, 5]):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('logreg', LogisticRegression(max_iter=1000)),
    ]).fit(Xc_tr, yc_tr)
    plot_boundary(ax, pipe, Xc_tr, yc_tr, f'Степень {degree}')

plt.suptitle('Полиномиальная логистическая регрессия на «двух лунах»', fontsize=14)
plt.tight_layout(); plt.show()
```

**Как читать графики:**

- **Степень 1:** прямая граница решения. Классы сильно перепутаны, accuracy около 0.85–0.88.
- **Степень 2:** граница становится параболой/эллипсом, accuracy растёт до 0.93–0.95. Модель лучше отделяет луны.
- **Степень 5:** граница «изгибается» вокруг точек, accuracy может быть ещё чуть выше, но есть риск переобучения (на других данных может быть хуже).

На этом датасете степени 2–3 обычно достаточно. Дальнейшее повышение степени — уже не про качество, а про переобучение. Это та же логика, что и в регрессии: **сложность модели должна соответствовать сложности зависимости**.

> **Замечание.** `LogisticRegression` внутри себя **не** добавляет нелинейные признаки — она всегда линейна по входным. Вся «нелинейность» появляется именно из-за `PolynomialFeatures`, который стоит **перед** логистической регрессией в конвейере. Это важный момент: модель остаётся простой и интерпретируемой, но мы «даём ей» более богатое пространство признаков.

#### Задания для самостоятельного выполнения

Все задания выполняются на `poly_reg.csv` (задания 1–3) и `poly_clf.csv` (задание 4). Для каждого пункта — код + краткий вывод.

##### Задание 1. Влияние степени на R² (15 минут)

1. Для степеней $d = 1, 2, 3, 4, 5, 6, 8, 10, 12, 15$ обучите полиномиальную регрессию и соберите R² на train и test в одну таблицу.
2. Постройте график «R² от степени» (две линии — train и test).
3. Ответьте на вопросы:
   - При какой степени test R² максимален?
   - Начиная с какой степени train R² превышает 0.99?
   - Начиная с какой степени test R² начинает заметно падать?
4. **Вывод:** какую степень полинома вы бы выбрали для этой задачи? Обоснуйте.

##### Задание 2. Ridge и высокая степень (15 минут)

1. Для степени 10 обучите три модели: обычную линейную регрессию, Ridge с $\alpha = 0.1$ и Ridge с $\alpha = 10$.
2. Заполните таблицу:

   | Модель | R² (train) | R² (test) | Число признаков |
   |---|---|---|---|
   | Linear (degree=10) | … | … | … |
   | Ridge (α=0.1) | … | … | … |
   | Ridge (α=10) | … | … | … |

3. Постройте графики трёх моделей на одних осях (плотная сетка `x`). Какая кривая ближе всего к истинной зависимости $y = 0.5x^3 - x^2 + 2$?
4. **Вывод:** как регуляризация помогает при высокой степени? Почему?

##### Задание 3. Сколько нужно данных? (10 минут)

1. Возьмите степень 5. Обучите модель на **10** объектах train и посмотрите R² на train и test.
2. Постепенно увеличивайте train до 30, 60, 100, всех. Заполните таблицу:

   | Размер train | R² (train) | R² (test) |
   |---|---|---|
   | 10 | … | … |
   | 30 | … | … |
   | 60 | … | … |
   | Все | … | … |

3. Постройте график «R² от размера train».
4. **Вывод:** как размер выборки связан с переобучением? Почему полиномиальные модели высокой степени «требуют» больше данных?

##### Задание 4. Полиномиальная классификация (15 минут)

1. Обучите логистическую регрессию (без полиномиальных признаков) на `poly_clf.csv`. Посчитайте accuracy на test. Постройте границу решения.
2. Обучите полиномиальную логистическую регрессию степени 2 и степени 5. Для каждой модели:
   - постройте границу решения;
   - посчитайте accuracy на train и на test.
3. Заполните таблицу:

   | Степень | Accuracy (train) | Accuracy (test) |
   |---|---|---|
   | 1 | … | … |
   | 2 | … | … |
   | 5 | … | … |

4. **Вывод:** при какой степени модель лучше всего разделяет «луны»? Наблюдается ли переобучение при степени 5 (сравните train/test accuracy)?

#### Контрольные вопросы

1. В чём идея полиномиальной регрессии? Почему она остаётся «линейной» моделью, хотя зависимость от $x$ — нелинейная?
2. Какие признаки создаёт `PolynomialFeatures(degree=2)` из одного исходного признака $x$? А из двух ($x_1, x_2$)?
3. Сколько признаков получится из 5 исходных при степени 3? (Подсказка: посчитайте вручную или через `PolynomialFeatures`.)
4. Почему `PolynomialFeatures` нужно обучать (`fit`) только на train, а к test применять `transform`?
5. Что происходит с моделью при слишком низкой степени полинома? А при слишком высокой?
6. Как определить, что модель переобучена, по значениям R² на train и test?
7. Почему высокая степень полинома может привести к переобучению, даже если модель линейна по коэффициентам?
8. Как регуляризация (Ridge) помогает бороться с переобучением полиномиальной модели? Что делает параметр $\alpha$?
9. Как полиномиальные признаки помогают в задаче классификации? Что меняется в границе решения?
10. Всегда ли повышение степени полинома улучшает качество модели? Почему?

#### Дополнительные задания

Задания не входят в основную оценку. Их цель — показать, что полиномиальные признаки — это лишь один из способов моделировать нелинейности; есть и другие.

1. **Сравнение с другими моделями.** Обучите на тех же данных (`poly_reg.csv`):
   - `KNeighborsRegressor(n_neighbors=5)` — модель на основе ближайших соседей;
   - `DecisionTreeRegressor(max_depth=3)` — дерево решений;
   - `RandomForestRegressor(n_estimators=100)` — случайный лес.
   
   Для каждой модели посчитайте R² на train/test и постройте предсказание на плотной сетке. Сравните с полиномиальной регрессией степени 3. Какая модель лучше описывает зависимость? В чём принципиальное отличие полиномиальной регрессии от деревьев и kNN?

2. **Полиномиальная регрессия с несколькими признаками.** Загрузите `sklearn.datasets.fetch_california_housing`. Обучите `LinearRegression` на одном признаке `MedInc`. Затем — `PolynomialFeatures(degree=2)` + `LinearRegression` на нём же. Сравните R². Улучшила ли полиномиальная модель качество?

3. **Взаимодействия признаков.** `PolynomialFeatures(interaction_only=True)` создаёт только «смешанные» признаки ($x_1 x_2$), но не $x_1^2$ и $x_2^2$. Попробуйте этот режим на `California Housing` с признаками `MedInc`, `HouseAge`, `AveRooms`. Сравните с полным `degree=2` и с обычной линейной регрессией.

4. **RidgeCV для подбора α.** Замените ручной перебор $\alpha$ в задании 2 на `RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)`. Какое $\alpha$ подбирает кросс-валидация? Совпадает ли с вашим ручным выбором?

5. **Полиномиальные признаки для классификации на реальных данных.** Загрузите `load_breast_cancer` и обучите логистическую регрессию с `PolynomialFeatures(degree=2)` и без него. Сравните ROC-AUC. Заметно ли улучшение? Почему (или почему нет)?
