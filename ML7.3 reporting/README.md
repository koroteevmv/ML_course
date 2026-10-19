### Представление результатов моделирования

#### Цель работы

На практике познакомиться с методами визуализации и представления результатов работы моделей машинного обучения. Научиться переводить технические метрики на язык бизнеса, выбирать порог классификации с учётом стоимости ошибок, анализировать ошибки модели и интерпретировать её предсказания — как глобально (какие признаки важны для всей модели), так и локально (почему модель приняла конкретное решение для конкретного клиента). Освоить SHAP и LIME — два основных инструмента интерпретации.

#### Подготовка данных

Работаем с синтетическим, но реалистичным датасетом по оттоку клиентов телеком-компании. Задача — предсказать, уйдёт ли клиент, чтобы служба удержания могла заранее предложить ему персональную скидку или бонус.

Скрипт ниже создаёт `churn.csv`. Датасет небольшой (3000 строк), обучение занимает секунды, но при этом в нём достаточно «сигнала» для содержательной интерпретации.

```python
import numpy as np
import pandas as pd


def generate_churn_data(n=3000, random_state=42, path='churn.csv'):
    """
    Генерирует синтетический набор данных об оттоке клиентов
    телеком-компании с осмысленной бизнес-логикой.
    """
    rng = np.random.RandomState(random_state)

    # --- Демография и услуги ---
    age = rng.randint(18, 76, n)
    tenure_months = rng.randint(1, 73, n)
    monthly_charges = np.round(rng.uniform(20, 120, n), 2)

    contract = rng.choice(
        ['month-to-month', 'one-year', 'two-year'],
        n, p=[0.55, 0.25, 0.20]
    )
    payment_method = rng.choice(
        ['electronic_check', 'mailed_check',
         'bank_transfer', 'credit_card'],
        n, p=[0.35, 0.15, 0.25, 0.25]
    )
    internet_service = rng.choice(
        ['DSL', 'Fiber optic', 'None'],
        n, p=[0.35, 0.45, 0.20]
    )
    tech_support = rng.choice(
        ['Yes', 'No', 'No internet service'],
        n, p=[0.35, 0.45, 0.20]
    )
    support_calls = rng.poisson(0.6, n)
    has_partner = rng.choice(['Yes', 'No'], n, p=[0.45, 0.55])

    # --- Логика вероятности оттока ---
    # (основана на типичных бизнес-паттернах телекома)
    logit = (
        -1.8
        + 1.1 * (contract == 'month-to-month').astype(int)
        - 0.6 * (contract == 'two-year').astype(int)
        + 0.05 * (monthly_charges - 60) / 20.0
        - 0.04 * (tenure_months - 30) / 12.0
        + 0.4 * (internet_service == 'Fiber optic').astype(int)
        - 0.55 * (tech_support == 'Yes').astype(int)
        + 0.45 * (support_calls >= 2).astype(int)
        + 0.35 * (payment_method == 'electronic_check').astype(int)
        - 0.2 * (has_partner == 'Yes').astype(int)
    )
    p_churn = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, p_churn)

    df = pd.DataFrame({
        'age': age,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'contract': contract,
        'payment_method': payment_method,
        'internet_service': internet_service,
        'tech_support': tech_support,
        'support_calls': support_calls,
        'has_partner': has_partner,
        'churn': churn,
    })
    df.to_csv(path, index=False)
    print(f'Сохранено {len(df)} записей в {path}')
    print(f'Доля оттока: {df.churn.mean():.3f}')
    return df


if __name__ == '__main__':
    generate_churn_data()
```

После запуска вы получите файл `churn.csv` с 3000 строк и 10 столбцами. Целевая переменная — `churn` (1 = клиент ушёл, 0 = остался). Доля оттока около 25–30%, что реалистично для телекома.

**Бизнес-контекст.** Клиент, который ушёл, теряет для компании в среднем $500 (CLV, customer lifetime value). Предложение удержания (скидка, бонус) стоит компании $50 и делается только тем клиентам, кого модель считает «риском оттока».

Загружаем и готовим данные:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, roc_auc_score,
                              precision_recall_curve, average_precision_score,
                              brier_score_loss, precision_score, recall_score,
                              f1_score)
from sklearn.calibration import calibration_curve

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

churn = pd.read_csv('churn.csv')

X = churn.drop(columns='churn')
y = churn['churn']

# Признаки: числовые и категориальные
num_cols = ['age', 'tenure_months', 'monthly_charges', 'support_calls']
cat_cols = ['contract', 'payment_method', 'internet_service',
            'tech_support', 'has_partner']

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(X_tr.shape, X_te.shape)
```

#### Методические указания

##### Зачем представлять результаты моделирования

Обучить модель — это только половина работы. Другая половина — **убедить бизнес-заказчика**, что модель полезна, и **объяснить**, как её использовать. Тимлид, продакт-менеджер, маркетолог, юрист — все они хотят видеть разные аспекты:

- **Продакт-менеджер**: сколько клиентов модель отнесёт к группе риска, какая доля из них действительно уйдёт, как это повлияет на бюджет удержания.
- **Маркетолог**: как выбрать порог, чтобы кампания была экономически выгодной.
- **Юрист и compliance**: почему модель отказала конкретному клиенту в кредите / скидке (требование GDPR, «право на объяснение»).
- **Аналитик данных**: где модель ошибается и как её улучшить.

Технические метрики (`accuracy = 0.83`, `AUC = 0.81`) для бизнеса бесполезны. Нужно переводить их на язык денег, клиентов и решений.

##### Матрица ошибок в бизнес-терминах

Базовая модель — `RandomForestClassifier` с препроцессингом. Обучим и получим предсказания:

```python
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         cat_cols),
    ]
)

model = Pipeline([
    ('prep', preprocess),
    ('clf', RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    )),
])

model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)
y_proba = model.predict_proba(X_te)[:, 1]

print(classification_report(y_te, y_pred,
                             target_names=['Остался', 'Ушёл']))
```

Стандартная матрица ошибок:

```python
cm = confusion_matrix(y_te, y_pred)
print(cm)
```

Но для бизнеса привычнее таблица с **смыслом** каждой ячейки:

|  | Модель сказала «останется» | Модель сказала «уйдёт» |
|---|---|---|
| **Клиент остался** | ✅ True Negative — правильно не тронули, сэкономили $50 | ❌ False Positive — зря потратили $50 на удержание |
| **Клиент ушёл** | ❌ False Negative — потеряли $500 | ✅ True Positive — удержали, сохранили $500 |

Стоимость ошибок:

- **FP**: потратили деньги на удержание клиента, который и так не собирался уходить → **−$50**.
- **FN**: не предложили удержание, клиент ушёл → **−$500**.
- **TP**: предложили удержание, клиент остался (предполагаем, что удержание сработало) → **+$450** (сохранили $500, потратили $50).
- **TN**: никого не тронули, клиент остался → **$0**.

Стоимость ошибок при таком раскладе **асимметрична**: пропустить уходящего клиента в 10 раз дороже, чем зря потратить на удержание.

Визуализируем матрицу ошибок в бизнес-терминах:

```python
import matplotlib.patches as mpatches

labels = np.array([
    [f'TN\n{cm[0,0]}\n$0',
     f'FP\n{cm[0,1]}\n−${50*cm[0,1]:,}'],
    [f'FN\n{cm[1,0]}\n−${500*cm[1,0]:,}',
     f'TP\n{cm[1,1]}\n+${450*cm[1,1]:,}'],
])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    np.array([[cm[0,0], cm[0,1]],[cm[1,0], cm[1,1]]]),
    annot=labels, fmt='', cmap='RdYlGn', cbar=False,
    xticklabels=['Сказал: останется', 'Сказал: уйдёт'],
    yticklabels=['Остался', 'Ушёл'],
    ax=ax, annot_kws={'fontsize': 11},
)
ax.set_title('Матрица ошибок в терминах бизнеса')
plt.tight_layout(); plt.show()
```

На такой матрице сразу видно, что каждый FN «стоит» в 10 раз дороже FP. Это и определяет, какой порог выбирать.

##### Выбор порога на основе стоимости ошибок

По умолчанию `predict` использует порог 0.5: если вероятность оттока > 0.5, клиент считается «уходящим». Но при асимметричных ошибках оптимальный порог сдвигается.

Определим **функцию суммарной стоимости** предсказаний при заданном пороге:

$$\text{Cost}(t) = 50 \cdot FP(t) + 500 \cdot FN(t) - 450 \cdot TP(t)$$

Мы ищем порог $t$, который минимизирует стоимость (или, что эквивалентно, максимизирует прибыль).

```python
def total_cost(y_true, y_proba, threshold,
               cost_fp=50, cost_fn=500, gain_tp=450):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return cost_fp * fp + cost_fn * fn - gain_tp * tp

thresholds = np.linspace(0.05, 0.95, 91)
costs = [total_cost(y_te, y_proba, t) for t in thresholds]
best_idx = int(np.argmin(costs))
best_threshold = thresholds[best_idx]

plt.figure(figsize=(10, 5))
plt.plot(thresholds, costs, 'b-')
plt.axvline(0.5, color='gray', linestyle='--', label='Порог по умолчанию (0.5)')
plt.axvline(best_threshold, color='red', linestyle='--',
            label=f'Оптимальный порог ({best_threshold:.2f})')
plt.xlabel('Порог вероятности оттока')
plt.ylabel('Суммарная стоимость, $')
plt.title('Зависимость бизнес-стоимости от порога классификации')
plt.legend()
plt.grid(True)
plt.show()

print(f'Порог 0.50: стоимость = ${costs[45]:,.0f}')
print(f'Порог {best_threshold:.2f}: стоимость = ${costs[best_idx]:,.0f}')
```

Обычно оптимальный порог оказывается **ниже 0.5** — мы склонны «перестраховаться» и предложить удержание большему числу клиентов, потому что пропустить уходящего дороже.

Сравним метрики при двух порогах:

```python
def print_metrics_at_threshold(y_true, y_proba, threshold, label=''):
    y_pred = (y_proba >= threshold).astype(int)
    print(f'--- {label} (порог = {threshold:.2f}) ---')
    print(f'  Precision: {precision_score(y_true, y_pred):.3f}')
    print(f'  Recall:    {recall_score(y_true, y_pred):.3f}')
    print(f'  F1:        {f1_score(y_true, y_pred):.3f}')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f'  TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}')

print_metrics_at_threshold(y_te, y_proba, 0.5, 'Порог по умолчанию')
print_metrics_at_threshold(y_te, y_proba, best_threshold, 'Бизнес-оптимальный')
```

Обратите внимание: при оптимальном пороге **precision падает** (мы чаще ошибаемся, обещая удержание), но **recall растёт** (ловит больше уходящих), и это экономически выгодно. Это важный момент для объяснения бизнесу: «формально точность стала хуже, но прибыль выросла».

##### Кривые Lift и Gain

**Кривая Gain (кумулятивная выгода)** показывает, какую долю всех уходящих клиентов мы поймаем, если обойдём с предложением удержания первых $N$% клиентов из списка, отсортированного по убыванию предсказанной вероятности.

**Кривая Lift** показывает, во сколько раз лучше случайного выбора мы работаем в каждой точке списка.

Эти графики — стандарт в маркетинге и CRM. Они отвечают на вопрос «сколько клиентов нужно обзвонить, чтобы поймать 80% уходящих».

```python
def gain_lift_chart(y_true, y_proba):
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    cum_positives = np.cumsum(y_sorted)
    total_positives = y_sorted.sum()
    n = len(y_sorted)
    population_pct = np.arange(1, n + 1) / n * 100
    gain_pct = cum_positives / total_positives * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Gain ---
    axes[0].plot(population_pct, gain_pct, 'b-', linewidth=2,
                 label='Модель')
    axes[0].plot([0, 100], [0, 100], 'k--', label='Случайный выбор')
    axes[0].set_xlabel('Доля клиентов, охваченных кампанией, %')
    axes[0].set_ylabel('Доля выявленных уходящих, %')
    axes[0].set_title('Кумулятивная кривая Gain')
    axes[0].legend()
    axes[0].grid(True)

    # --- Lift ---
    baseline = population_pct / 100
    lift = (gain_pct / 100) / np.where(baseline > 0, baseline, np.nan)
    axes[1].plot(population_pct, lift, 'g-', linewidth=2, label='Lift')
    axes[1].axhline(1.0, color='k', linestyle='--', label='Случайный (Lift = 1)')
    axes[1].set_xlabel('Доля клиентов, охваченных кампанией, %')
    axes[1].set_ylabel('Lift')
    axes[1].set_title('Кривая Lift')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

gain_lift_chart(y_te.values, y_proba)
```

**Как читать график.** Если модель идеальна, кривая Gain сразу идёт в правый верхний угол (все уходящие пойманы на первых нескольких процентах списка). Если модель бесполезна, кривая Gain лежит на диагонали. Чем «выпуклее» кривая, тем лучше модель.

Lift в первых 10% списка обычно 3–5× — это означает, что при обзвоне 10% клиентов мы ловим в 3–5 раз больше уходящих, чем при случайном выборе тех же 10%.

Полезно вывести и **децильный анализ** — таблицу с долей оттока по децилям предсказанной вероятности:

```python
def decile_table(y_true, y_proba):
    df = pd.DataFrame({'y': y_true, 'p': y_proba})
    df['decile'] = pd.qcut(-df['p'], q=10, labels=range(1, 11))
    tbl = df.groupby('decile').agg(
        n=('y', 'size'),
        churn_rate=('y', 'mean'),
    ).reset_index()
    tbl['lift'] = tbl['churn_rate'] / df['y'].mean()
    return tbl

decile_table(y_te.values, y_proba)
```

Такая таблица хорошо воспринимается бизнесом: «в топ-10% по риску оттока реальный отток составляет 78%, что в 3.1 раза выше среднего».

##### Калибровка вероятностей

Модель выдаёт `predict_proba` — «вероятность» оттока. Но не все модели дают **калиброванные** вероятности: если модель говорит «70%», это не всегда значит, что из 100 таких клиентов уйдёт ровно 70. Это важно, если вероятности используются для расчёта ожидаемой прибыли.

**Кривая калибровки** показывает, насколько средняя предсказанная вероятность совпадает с реальной частотой:

```python
def plot_calibration(y_true, y_proba, label='Модель'):
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=10, strategy='uniform'
    )
    plt.plot(prob_pred, prob_true, marker='o', label=label)
    plt.plot([0, 1], [0, 1], 'k--', label='Идеальная калибровка')
    plt.xlabel('Средняя предсказанная вероятность')
    plt.ylabel('Реальная частота оттока')
    plt.title('Кривая калибровки')
    plt.legend()
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_calibration(y_te.values, y_proba, 'RandomForest')
plt.show()

print(f'Brier score: {brier_score_loss(y_te, y_proba):.4f}')
```

Brier score — среднеквадратичная ошибка между предсказанной вероятностью и реальным исходом. Меньше — лучше. 0 — идеальная калибровка, 0.25 — эквивалент случайного угадывания при равных классах.

Random Forest и особенно деревья решений часто плохо калиброваны (склонны выдавать «уверенные» вероятности около 0 или 1). Логистическая регрессия калибрована лучше, но менее точна. Если бизнесу нужны именно вероятности (для расчёта ожидаемого дохода), можно применить `CalibratedClassifierCV`.

##### Анализ ошибок модели

Ошибки — это не только цифры. Важно понять, **на каких сегментах** модель ошибается чаще всего. Это подсказывает, где улучшать модель и где она может быть опасной.

```python
def error_analysis(X_df, y_true, y_proba, threshold, segment_col):
    df = X_df.copy()
    df['y_true'] = y_true.values if hasattr(y_true, 'values') else y_true
    df['y_pred'] = (y_proba >= threshold).astype(int)
    df['error'] = (df['y_true'] != df['y_pred']).astype(int)
    df['fn'] = ((df['y_true'] == 1) & (df['y_pred'] == 0)).astype(int)
    df['fp'] = ((df['y_true'] == 0) & (df['y_pred'] == 1)).astype(int)

    return df.groupby(segment_col).agg(
        n=('y_true', 'size'),
        error_rate=('error', 'mean'),
        fn_rate=('fn', 'mean'),
        fp_rate=('fp', 'mean'),
    ).sort_values('error_rate', ascending=False)

# По типу контракта
print(error_analysis(X_te, y_te, y_proba, best_threshold, 'contract'))

# По наличию техподдержки
print(error_analysis(X_te, y_te, y_proba, best_threshold, 'tech_support'))
```

Таблицы вроде этих хорошо работают в разговоре с бизнесом: «Модель чаще ошибается на клиентах с помесячной оплатой, особенно на молодых. Возможно, стоит дообучить её на этом сегменте или применить к нему отдельную модель».

Для **регрессии** аналогичный анализ строится по остаткам (residual = $y - \hat{y}$): нужно проверить, что остатки не имеют систематического смещения в каком-то сегменте.

##### Глобальная интерпретация: `feature_importances_` и permutation importance

**Встроенная важность признаков** (`feature_importances_`) в деревьях показывает, насколько часто и насколько эффективно признак использовался для разбиений. Но у неё есть недостатки: она смещена в пользу признаков с большим числом уникальных значений (например, числовых).

**Permutation importance** — более надёжный и модельно-независимый метод: признак перемешивается случайным образом, и мы смотрим, насколько упадёт качество модели. Если качество не изменилось — признак не важен.

```python
from sklearn.inspection import permutation_importance

# Передаём исходный (не трансформированный) X_te
# Модель внутри пайплайна сама применит препроцессинг
perm = permutation_importance(
    model, X_te, y_te, n_repeats=10, random_state=42,
    scoring='roc_auc', n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': X_te.columns,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std,
}).sort_values('importance_mean', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(perm_df['feature'][::-1],
         perm_df['importance_mean'][::-1],
         xerr=perm_df['importance_std'][::-1])
plt.xlabel('Снижение ROC-AUC при перемешивании')
plt.title('Permutation importance')
plt.tight_layout()
plt.show()

print(perm_df)
```

Обратите внимание: permutation importance можно посчитать на **исходных** столбцах, даже если модель обучена на трансформированных данных. Это большое преимущество — бизнесу не нужно разбираться в one-hot-кодировании.

Сравните с встроенной важностью Random Forest:

```python
# Имена признаков после препроцессинга
feature_names = model.named_steps['prep'].get_feature_names_out()
rf_importance = model.named_steps['clf'].feature_importances_

rf_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_importance,
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(data=rf_df, y='feature', x='importance')
plt.title('Встроенная важность Random Forest (топ-15)')
plt.tight_layout(); plt.show()
```

Как правило, порядок признаков согласован (в нашем примере самое важное — `contract`, затем `tenure_months` и `monthly_charges`), но permutation importance даёт более честную картину.

##### Partial Dependence Plots (PDP)

PDP показывает, как **в среднем** предсказание модели зависит от значения одного признака при фиксированных остальных. Это удобный способ объяснить бизнесу форму зависимости.

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    model, X_te, features=['tenure_months', 'monthly_charges'],
    kind='average', ax=ax
)
plt.suptitle('Partial Dependence для ключевых признаков')
plt.tight_layout(); plt.show()
```

**Как читать график.** По оси X — значение признака, по оси Y — средняя предсказанная вероятность оттока. Например, для `tenure_months` мы обычно видим монотонно убывающую кривую: чем дольше клиент с нами, тем ниже вероятность оттока. Это совпадает с бизнес-интуицией и хорошо воспринимается заказчиком.

PDP бывает обманчив, когда признаки сильно коррелированы (например, `tenure_months` и `monthly_charges × tenure = total_revenue`). В таких случаях лучше использовать SHAP.

##### глобальная и локальная интерпретация

**SHAP (SHapley Additive exPlanations)** — теоретико-игровой подход к интерпретации. Для каждого предсказания он раскладывает его на вклад каждого признака, причём сумма вкладов равна отклонению предсказания от среднего по выборке:

$$f(x) = \phi_0 + \sum_{i=1}^{p} \phi_i(x)$$

где $\phi_0$ — среднее предсказание по данным, $\phi_i(x)$ — вклад $i$-го признака для конкретного объекта $x$.

Главные преимущества SHAP: **согласованность** (если признак важнее, его вклад не может уменьшиться) и **локальная точность** (вклады точно складываются в предсказание).

Установка:

```bash
pip install shap
```

Для деревьев решений и ансамблей есть быстрый `TreeExplainer`:

```python
import shap

# Для пайплайна нужно передать модель и трансформированные данные
X_te_transformed = model.named_steps['prep'].transform(X_te)
feature_names = model.named_steps['prep'].get_feature_names_out()

explainer = shap.TreeExplainer(model.named_steps['clf'])
shap_values = explainer.shap_values(X_te_transformed)
```

`shap_values` — это массив вкладов. Для бинарной классификации `shap_values` обычно список из двух массивов (по одному на класс) или массив формы `(n_samples, n_features, 2)`. Нам нужен вклад в вероятность **оттока** (класс 1):

```python
# Зависит от версии shap; проверим тип
if isinstance(shap_values, list):
    sv = shap_values[1]  # вклад в класс «уйдёт»
else:
    sv = shap_values
```

**Глобальная интерпретация: `summary_plot(bar)`**

Показывает средний абсолютный вклад каждого признака — по сути, глобальная важность:

```python
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_te_transformed,
                  feature_names=feature_names, plot_type='bar', show=False)
plt.title('SHAP: глобальная важность признаков')
plt.tight_layout(); plt.show()
```

**Глобальная интерпретация: `summary_plot(beeswarm)`**

Показывает не только важность, но и **направление** влияния каждого признака:

```python
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_te_transformed,
                  feature_names=feature_names, show=False)
plt.title('SHAP: вклад признаков и направление влияния')
plt.tight_layout(); plt.show()
```

**Как читать beeswarm:**

- Каждая точка — один клиент.
- Положение по горизонтали — вклад признака в предсказание (влево — снижает вероятность оттока, вправо — повышает).
- Цвет — значение признака (красный — высокое, синий — низкое).
- Признаки отсортированы по убыванию важности.

Например, для `contract_month-to-month`: если точка красная (клиент с помесячной оплатой), она смещена вправо — повышает риск оттока. Это идеальное объяснение для бизнеса.

**Локальная интерпретация: `waterfall_plot`**

Показывает, как модель пришла к предсказанию для **одного конкретного клиента**:

```python
# Найдём клиента с высоким риском оттока
idx_high_risk = int(np.argmax(y_proba))

shap.waterfall_plot(
    shap.Explanation(
        values=sv[idx_high_risk],
        base_values=explainer.expected_value[1]
                    if isinstance(explainer.expected_value, list)
                    else explainer.expected_value,
        data=X_te_transformed[idx_high_risk],
        feature_names=feature_names,
    ),
    max_display=10, show=True
)
```

На этом графике видно, как стартовое значение `E[f(X)]` (средняя вероятность оттока по выборке) под действием конкретных признаков конкретного клиента превращается в его индивидуальное предсказание. Это и есть ответ на вопрос «почему модель считает, что этот клиент уйдёт».

**Зависимость от одного признака: `dependence_plot`**

Показывает, как вклад признака меняется с его значением:

```python
plt.figure(figsize=(10, 5))
shap.dependence_plot(
    'num__tenure_months', sv, X_te_transformed,
    feature_names=feature_names, show=False
)
plt.title('SHAP: зависимость вклада от стажа клиента')
plt.tight_layout(); plt.show()
```

В отличие от PDP, `dependence_plot` показывает **индивидуальные вклады** всех клиентов, а не среднее — это позволяет увидеть гетерогенность и нелинейности.

**Компактная визуализация для отчёта: `force_plot`**

`force_plot` показывает вклады в виде «стрелок», толкающих предсказание вправо (к оттоку) или влево (к удержанию). Удобно для интерактивных отчётов, но статичный вариант плохо читается в презентации. Покажем, как его построить для одного клиента:

```python
shap.initjs()  # только для Jupyter

shap.force_plot(
    explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value,
    sv[idx_high_risk],
    X_te_transformed[idx_high_risk],
    feature_names=feature_names,
)
```

В обычном скрипте (не Jupyter) `force_plot` вернёт интерактивный объект, который можно сохранить в HTML и открыть в браузере:

```python
html = shap.force_plot(
    explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value,
    sv[idx_high_risk],
    X_te_transformed[idx_high_risk],
    feature_names=feature_names,
    matplotlib=False,
).html()
with open('shap_force_example.html', 'w', encoding='utf-8') as f:
    f.write(html)
```

##### LIME: локальная интерпретация

**LIME (Local Interpretable Model-agnostic Explanations)** — другой подход к объяснению одного предсказания. Идея: обучим простую **локальную** модель (например, линейную регрессию) на искусственно сгенерированных точках **вокруг** интересующего объекта и посмотрим на её коэффициенты.

В отличие от SHAP, LIME не претендует на теоретическую согласованность и не разлагает предсказание точно. Но он работает с любой моделью (в том числе «чёрным ящиком») и очень удобен для быстрых объяснений «на лету».

Установка:

```bash
pip install lime
```

Применение:

```python
from lime.lime_tabular import LimeTabularExplainer

# LIME хочет числовые данные — применим препроцессинг
X_tr_transformed = model.named_steps['prep'].transform(X_tr)
X_te_transformed = model.named_steps['prep'].transform(X_te)

lime_explainer = LimeTabularExplainer(
    training_data=X_tr_transformed,
    feature_names=feature_names,
    class_names=['Остался', 'Ушёл'],
    mode='classification',
    discretize_continuous=True,
    random_state=42,
)

exp = lime_explainer.explain_instance(
    X_te_transformed[idx_high_risk],
    model.named_steps['clf'].predict_proba,
    num_features=10,
)

# Вывод в ноутбуке
exp.show_in_notebook(show_table=True)

# Или в виде списка
for feature, weight in exp.as_list():
    print(f'{feature}: {weight:+.4f}')
```

LIME показывает вклад каждого признака в предсказание. Например, «contract=month-to-month > 0.5: +0.12» означает, что этот признак «толкает» предсказание к оттоку на 0.12.

**Сравнение SHAP и LIME.** На одном и том же клиенте постройте оба объяснения. Как правило, топ-5 вкладов совпадают, но численные значения и порядок могут различаться. Это нормально: SHAP даёт теоретически обоснованные вклады, LIME — локальную аппроксимацию, которая может «видеть» данные по-другому.

**Рекомендации по использованию:**

- **SHAP** — когда нужна строгая интерпретация и согласованность. Хорошо работает с деревьями (быстро через `TreeExplainer`), медленнее с другими моделями (`KernelExplainer`).
- **LIME** — когда нужна быстрая локальная интерпретация произвольной модели и не критична строгая согласованность.
- На практике часто используют **оба**: SHAP — для глобального анализа, LIME — для быстрых объяснений «на лету» (например, в интерфейсе оператора колл-центра).

##### Сборка «дашборда» для бизнес-заказчика

Все предыдущие графики и таблицы хороши по отдельности, но заказчику нужен **единый документ**. Соберём минимальный «дашборд» из четырёх блоков:

1. Ключевые бизнес-метрики (одна таблица).
2. Lift-кривая с точкой, соответствующей размеру кампании.
3. Матрица ошибок с бизнес-интерпретацией.
4. SHAP summary (глобальная важность признаков).

```python
def business_dashboard(y_true, y_proba, threshold, y_true_churn_rate):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    # --- Блок 1: KPI ---
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    cost = 50 * fp + 500 * fn - 450 * tp
    revenue_saved = 450 * tp
    spent = 50 * (tp + fp)
    kpi_text = (
        f"Бизнес-показатели кампании удержания\n\n"
        f"Целевая группа (риск оттока):     {tp + fp:>5d}\n"
        f"Реально ушли из группы:           {tp:>5d}\n"
        f"Удержано (по модели):             {tp:>5d}\n"
        f"Пропущено уходящих (FN):          {fn:>5d}\n"
        f"Зря потрачено на FP:              {fp:>5d}\n\n"
        f"Затраты на кампанию:              ${spent:>10,.0f}\n"
        f"Сохранённая выручка:              ${revenue_saved:>10,.0f}\n"
        f"Чистый эффект:                    ${-cost:>10,.0f}\n"
        f"Порог:                            {threshold:.2f}"
    )
    ax.text(0.02, 0.98, kpi_text, va='top', fontsize=11,
            family='monospace')
    ax.set_title('Ключевые показатели', fontsize=12, fontweight='bold')
    
    # --- Блок 2: Gain curve ---
    ax = fig.add_subplot(gs[0, 1])
    order = np.argsort(-y_proba)
    y_sorted = np.asarray(y_true)[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = y_sorted.sum()
    pop_pct = np.arange(1, len(y_sorted) + 1) / len(y_sorted) * 100
    gain = cum_pos / total_pos * 100
    ax.plot(pop_pct, gain, 'b-', linewidth=2, label='Модель')
    ax.plot([0, 100], [0, 100], 'k--', label='Случайно')
    coverage_pct = (tp + fp) / len(y_sorted) * 100
    ax.axvline(coverage_pct, color='red', linestyle=':',
               label=f'Кампания: {coverage_pct:.0f}%')
    ax.set_xlabel('Доля клиентов в кампании, %')
    ax.set_ylabel('Поймано уходящих, %')
    ax.set_title('Кумулятивный Gain')
    ax.legend()
    ax.grid(True)
    
    # --- Блок 3: Confusion matrix ---
    ax = fig.add_subplot(gs[1, 0])
    labels = np.array([
        [f'TN\n{tn}\n$0',
         f'FP\n{fp}\n−${50*fp:,}'],
        [f'FN\n{fn}\n−${500*fn:,}',
         f'TP\n{tp}\n+${450*tp:,}'],
    ])
    sns.heatmap(np.array([[tn, fp], [fn, tp]]),
                annot=labels, fmt='', cmap='RdYlGn', cbar=False,
                xticklabels=['Сказал: останется', 'Сказал: уйдёт'],
                yticklabels=['Остался', 'Ушёл'],
                ax=ax, annot_kws={'fontsize': 10})
    ax.set_title('Матрица ошибок в деньгах')
    
    # --- Блок 4: SHAP bar (глобальная важность) ---
    ax = fig.add_subplot(gs[1, 1])
    # Упрощённый top-10 по среднему |SHAP|
    importance = np.abs(sv).mean(axis=0)
    order_idx = np.argsort(importance)[-10:]
    ax.barh(np.array(feature_names)[order_idx],
            importance[order_idx], color='tab:blue')
    ax.set_xlabel('Средний |SHAP|')
    ax.set_title('Топ-10 признаков по SHAP')
    
    fig.suptitle(f'Дашборд модели: отток клиентов\n(порог = {threshold:.2f}, '
                 f'средний отток = {y_true_churn_rate:.1%})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

business_dashboard(y_te.values, y_proba, best_threshold, y_te.mean())
```

Такой дашборд можно защищать перед бизнесом: слева сверху — цифры, которые интересуют менеджера; справа сверху — кривая, показывающая, как модель помогает кампании; слева снизу — что стоит каждая ошибка; справа снизу — какие признаки влияют на модель в целом.

#### Задания для самостоятельного выполнения

Все задания выполняются в ноутбуке. Для каждого пункта нужен код + краткий текстовый вывод (что видно на графике, какие цифры получились, что они означают для бизнеса).

1. **Своя матрица стоимости.** Предположим, что в вашей компании предложение удержания стоит $100, а CLV клиента — $800. Пересчитайте оптимальный порог и стоимость кампании. Как изменилось число FP и FN? Сравните с базовым вариантом.

2. **Кривые Lift и Gain для двух моделей.** Обучите две модели: логистическую регрессию и Random Forest. Постройте для обеих кривые Gain и Lift на одном графике. Сделайте вывод, какая модель лучше «ловит» уходящих в первых децилях списка.

3. **Децильный анализ.** Постройте децильную таблицу (10 групп по предсказанной вероятности) с долей оттока, cumulative capture rate и lift. Оформите её как таблицу, готовую для презентации. Добавьте график столбчатой диаграммы `churn_rate` по децилям.

4. **Калибровка.** Сравните калибровку Random Forest и логистической регрессии на одних и тех же данных. Постройте обе кривые калибровки на одном графике, посчитайте Brier score. Примените `CalibratedClassifierCV` к Random Forest и покажите, что калибровка улучшилась (кривая ближе к диагонали, Brier score меньше). Почему бизнесу может быть важно, чтобы вероятности были откалиброваны?

5. **Анализ ошибок по сегментам.** Разбейте клиентов по типу контракта и по наличию техподдержки. Для каждого сегмента посчитайте recall, precision и F1 при выбранном пороге. В каком сегменте модель работает хуже всего? Предложите, что можно сделать (например, отдельная модель или дообучение).

6. **Permutation importance против встроенной.** Сравните permutation importance (на исходных признаках) и `feature_importances_` (на трансформированных) для Random Forest. Совпадает ли топ-5? Для каких признаков расхождение особенно сильное и почему?

7. **PDP для двух признаков.** Постройте PDP для `monthly_charges` и `support_calls`. Опишите форму зависимости словами: «При увеличении числа обращений в поддержку с 0 до 2 вероятность оттока растёт в среднем на X п.п.». Согласуется ли это с бизнес-логикой?

8. **SHAP: глобальная интерпретация.** Постройте `summary_plot(bar)` и `summary_plot(beeswarm)`. Для трёх самых важных признаков сформулируйте по одному предложению, объясняющему направление влияния (например, «клиенты с помесячной оплатой имеют более высокий риск оттока»).

9. **SHAP: локальная интерпретация.** Выберите одного клиента с высокой предсказанной вероятностью оттока и одного — с низкой. Постройте для обоих `waterfall_plot`. Объясните словами: «Этот клиент считается рискованным, потому что…». Сравните объяснения.

10. **SHAP: зависимость от признака.** Постройте `dependence_plot` для `tenure_months` и `monthly_charges`. Опишите, как меняется вклад признака с его значением. Для `monthly_charges` проверьте, есть ли нелинейность (например, резкий скачок после определённой суммы).

11. **LIME: локальная интерпретация.** Для того же клиента с высоким риском постройте LIME-объяснение. Сравните с SHAP-объяснением. Совпадает ли топ-5 признаков? Какие расхождения и почему?

12. **Сборка дашборда.** Соберите свой «дашборд» для бизнеса по образцу из методических указаний. Адаптируйте его под выбранный вами порог (из задания 1). Проверьте, что все цифры согласованы: KPI, матрица ошибок и Gain-кривая должны «сходиться».

13. **Сценарий презентации.** Напишите короткий текст (10–15 предложений) — «речь для презентации бизнес-заказчику», в которой вы объясняете:
    - зачем нужна модель,
    - какой порог выбран и почему,
    - какие клиенты попадут в кампанию,
    - какие признаки определяют риск,
    - что может пойти не так (риски модели).

Это задание проверяет умение переводить технические результаты на язык бизнеса.

#### Контрольные вопросы

1. Почему accuracy — плохая метрика для задач с дисбалансом классов? Какие метрики лучше подходят?
2. Что показывает матрица ошибок? Как её перевести на язык бизнес-стоимости?
3. Как выбрать порог классификации, если стоимость FP и FN разная? Приведите пример расчёта.
4. Что показывает кривая Gain? Чем она отличается от ROC-кривой?
5. Как читать кривую Lift? Что означает Lift = 3 в первых 10% списка?
6. Зачем нужна калибровка вероятностей? В каких случаях модель может быть хорошо откалибрована, а в каких — нет?
7. Какие методы калибровки вы знаете? (Подсказка: `CalibratedClassifierCV` с `method='sigmoid'` и `method='isotonic'`.)
8. Что показывает permutation importance и почему она предпочтительнее встроенной `feature_importances_` для интерпретации?
9. В чём отличие PDP от SHAP dependence plot?
10. Как работает SHAP? Почему его вклады называют «справедливыми»?
11. Как читать SHAP beeswarm plot? Что означают цвет, положение и размер точек?
12. Как объяснить бизнесу предсказание для одного клиента с помощью SHAP waterfall?
13. Что такое LIME и чем он отличается от SHAP? В каких случаях LIME предпочтительнее?
14. Почему SHAP хорошо работает с деревьями решений, а для других моделей может быть медленным?
15. Почему интерпретация модели важна не только для аналитика, но и для юристов / compliance (GDPR, «право на объяснение»)?
16. Какие ограничения у интерпретационных методов? Может ли SHAP «обмануть»?

#### Дополнительные задания

1. **SHAP для регрессии.** Загрузите `fetch_california_housing` из `sklearn`, обучите `GradientBoostingRegressor` и постройте все основные SHAP-графики (summary bar, beeswarm, waterfall для одного объекта, dependence plot). Сравните с тем, что вы видели на классификации: что общего, что отличается?

2. **Kernel SHAP vs Tree SHAP.** Для одной и той же модели (Random Forest) постройте объяснения через `TreeExplainer` и `KernelExplainer`. Сравните время работы и результаты. Объясните, почему `TreeExplainer` работает быстрее.

3. **SHAP и взаимодействия.** Постройте `shap.dependence_plot` для `monthly_charges` с цветовой кодировкой по `tenure_months` (параметр `interaction_index`). Видны ли взаимодействия между признаками? Опишите их.

4. **LIME: выбор количества признаков.** Для одного и того же клиента постройте LIME-объяснения с `num_features = 5, 10, 20`. Как меняется картина? Какие признаки «всплывают» только при большом `num_features`, и что это значит?

5. **Кастомизация дашборда.** Соберите дашборд в `plotly` или `dash` с интерактивными графиками: при движении порога вручную пересчитываются KPI и перерисовывается матрица ошибок. Такие дашборды часто используются в бизнес-аналитике.

6. **Отчёт для бизнеса.** Оформите результаты работы в виде короткого PDF-отчёта (например, через `jupyter nbconvert --to webpdf` или `weasyprint`). Отчёт должен содержать: постановку задачи, ключевые метрики, один главный график, одну таблицу с рекомендациями и раздел «Ограничения и риски модели».

7. **Сравнение с интерпретацией «как есть».** Постройте один и тот же объяснительный график (например, важность признаков) тремя способами: встроенный `feature_importances_`, permutation importance, SHAP summary bar. Оцените, насколько они согласованы. Какой график вы бы показали бизнесу и почему?

8. **Ответ на «право на объяснение».** Представьте, что клиент, которому модель отказала в скидке (отнесла к группе «не уйдёт»), требует объяснения. Составьте письменный ответ (5–7 предложений) на основе SHAP waterfall для этого клиента. Ответ должен быть понятен человеку без технического бэкграунда.