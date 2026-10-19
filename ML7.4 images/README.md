### Анализ изображений

#### Цель работы

На практике познакомиться с тем, как общие модели машинного обучения работают с изображениями, и какие специфические приёмы используются при извлечении признаков из визуальных данных. Освоить полный конвейер обработки изображений: предобработка, извлечение признаков классическими дескрипторами (HOG, LBP, цветовые гистограммы, границы), применение стандартных моделей (SVM, RandomForest, kNN) к задачам классификации и идентификации, а также базовую сегментацию изображений. Свёрточные нейронные сети и другие методы глубокого обучения в этой работе не рассматриваются — они вынесены в следующий курс.

#### Подготовка данных

Работаем с четырьмя наборами данных:

- **`load_digits`** (встроен в `sklearn`) — 8×8 изображения рукописных цифр, 1797 штук, 10 классов. Малый размер делает его идеальным для базовых экспериментов и быстрой итерации.
- **`fetch_olivetti_faces`** (встроен в `sklearn`, скачивается один раз) — 400 фотографий лиц 64×64, 40 человек по 10 снимков. Классическая задача идентификации.
- **`skimage.data`** (встроен в `scikit-image`) — образцы реальных фотографий: `camera()` (512×512, серое), `coins()` (303×384, монеты), `astronaut()` (512×512, цветное). Не требуют загрузки.
- **Синтетическое изображение с известной сегментацией** — генерируется скриптом, чтобы можно было численно оценить качество сегментации (IoU, Dice).

Скрипт ниже подготавливает все данные и сохраняет их локально. Его достаточно запустить один раз.

```python
# prepare_data.py
import numpy as np


def prepare_data():
    print('Готовим данные...')

    # --- 1. Цифры (8x8, встроены в sklearn) ---
    from sklearn.datasets import load_digits
    digits = load_digits()
    np.save('digits_images.npy', digits.images)   # (1797, 8, 8), значения 0..16
    np.save('digits_labels.npy', digits.target)    # (1797,)
    print(f'  digits: {digits.images.shape}, классы: {np.unique(digits.target)}')

    # --- 2. Лица (64x64, скачиваются один раз ~4 МБ) ---
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces()
    np.save('faces_images.npy', faces.images)     # (400, 64, 64), значения 0..1
    np.save('faces_labels.npy', faces.target)      # (400,)
    print(f'  faces: {faces.images.shape}, '
          f'классов: {len(np.unique(faces.target))}')

    # --- 3. Образцы skimage (встроены, не требуют загрузки) ---
    from skimage import data
    np.save('img_camera.npy', data.camera())       # 512x512 uint8
    np.save('img_coins.npy', data.coins())          # 303x384 uint8
    np.save('img_astronaut.npy', data.astronaut())   # 512x512x3 uint8
    print(f'  camera: {data.camera().shape}, '
          f'coins: {data.coins().shape}, '
          f'astronaut: {data.astronaut().shape}')

    # --- 4. Синтетическая сегментация с известным ground truth ---
    from skimage.draw import disk, rectangle
    n = 256
    img = np.full((n, n), 30, dtype=np.uint8)  # тёмный фон
    gt = np.zeros((n, n), dtype=np.uint8)

    # Красный круг (яркий)
    rr, cc = disk((80, 80), 40, shape=(n, n))
    img[rr, cc] = 210
    gt[rr, cc] = 1

    # Прямоугольник (средний)
    rr, cc = rectangle(start=(140, 100), end=(220, 210), shape=(n, n))
    img[rr, cc] = 150
    gt[rr, cc] = 2

    # Синий круг (тёмный)
    rr, cc = disk((90, 190), 35, shape=(n, n))
    img[rr, cc] = 90
    gt[rr, cc] = 3

    # Немного шума, чтобы задача была реалистичной
    rng = np.random.RandomState(42)
    img = np.clip(
        img.astype(int) + rng.randint(-15, 16, img.shape),
        0, 255
    ).astype(np.uint8)

    np.save('seg_synthetic.npy', img)
    np.save('seg_synthetic_gt.npy', gt)
    print(f'  seg_synthetic: {img.shape}, классы: {np.unique(gt)}')

    print('Готово.')


if __name__ == '__main__':
    prepare_data()
```

Понадобятся библиотеки:

```bash
pip install scikit-learn scikit-image pandas numpy matplotlib seaborn
```

Загружаем данные в ноутбуке:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

digits_X = np.load('digits_images.npy')  # (1797, 8, 8)
digits_y = np.load('digits_labels.npy')  # (1797,)
faces_X  = np.load('faces_images.npy')   # (400, 64, 64)
faces_y  = np.load('faces_labels.npy')   # (400,)
camera   = np.load('img_camera.npy')     # (512, 512)
coins    = np.load('img_coins.npy')      # (303, 384)
astronaut = np.load('img_astronaut.npy') # (512, 512, 3)
seg_syn  = np.load('seg_synthetic.npy')  # (256, 256)
seg_gt   = np.load('seg_synthetic_gt.npy')  # (256, 256)

print('Цифры:', digits_X.shape, '| метки:', digits_y.shape)
print('Лица:', faces_X.shape, '| метки:', faces_y.shape)
print('Изображения:', camera.shape, coins.shape, astronaut.shape)
print('Синтетическая сегментация:', seg_syn.shape, '| GT:', seg_gt.shape)
```

Посмотрим на несколько примеров — это первое, что нужно сделать с новым набором изображений:

```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits_X[i], cmap='gray')
    ax.set_title(f'Метка: {digits_y[i]}')
    ax.axis('off')
plt.suptitle('Примеры из набора рукописных цифр (8×8)')
plt.tight_layout(); plt.show()
```

```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(faces_X[i], cmap='gray')
    ax.set_title(f'Человек №{faces_y[i]}')
    ax.axis('off')
plt.suptitle('Примеры из набора лиц (64×64)')
plt.tight_layout(); plt.show()
```

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(camera, cmap='gray'); axes[0].set_title('camera (серое)'); axes[0].axis('off')
axes[1].imshow(coins, cmap='gray'); axes[1].set_title('coins'); axes[1].axis('off')
axes[2].imshow(astronaut); axes[2].set_title('astronaut (цветное)'); axes[2].axis('off')
plt.tight_layout(); plt.show()
```

#### Методические указания

##### Как изображение становится признаками

Для человека изображение — это картинка. Для модели машинного обучения — это **матрица чисел фиксированной формы**.

- **Оттенки серого**: `(H, W)` — матрица яркостей (обычно 0–255 или 0–1).
- **Цветное (RGB)**: `(H, W, 3)` — три канала: красный, зелёный, синий.
- **Прозрачность (RGBA)**: `(H, W, 4)`.
- **Многоканальное (гиперспектральное)**: `(H, W, C)` с произвольным числом каналов.

Для стандартных моделей (SVM, RandomForest, LogisticRegression) нужен **одномерный вектор фиксированной длины**. Первое, что приходит в голову, — просто «вытянуть» матрицу в вектор:

$$\text{image } (H \times W) \;\longrightarrow\; \text{vector } (H \cdot W)$$

Для цифр 8×8 это даст 64 признака — совсем немного. Для лиц 64×64 — 4096, а для фото 512×512 — уже 262 тысячи. И это только для серого; для RGB — втрое больше.

Здесь начинается проблема: **«проклятие размерности»**. Большинство моделей плохо работают, когда признаков много, а объектов мало. Кроме того, отдельные пиксели **не несут семантики** — модель не «видит» ни краёв, ни форм, ни текстур. Если лицо сдвинется на один пиксель, вектор изменится полностью.

Поэтому вместо сырых пикселей обычно извлекают **признаки**, описывающие изображение компактно и осмысленно. В этой работе мы разберём три подхода:

1. **Простые признаки** — гистограммы яркости, цветовые гистограммы, координаты границ. Дают небольшой вектор, описывающий общую статистику.
2. **Локальные дескрипторы** — HOG и LBP. Разбивают изображение на ячейки, в каждой считают локальную характеристику (направление градиентов или текстуру), затем объединяют в один вектор. Получается признаковый вектор длиной ~100–5000, устойчивый к малым сдвигам и освещению.
3. **Сырые пиксели** — иногда достаточны, если изображения маленькие (как у цифр 8×8) и хорошо выровнены.

**Проверим на цифрах.** Возьмём одну цифру и посмотрим на неё как на матрицу чисел:

```python
img = digits_X[0]
print(f'Форма: {img.shape}, dtype: {img.dtype}')
print(f'Диапазон значений: [{img.min()}, {img.max()}]')
print('Матрица пикселей:')
print(img)
```

Каждое значение — яркость пикселя (0 — чёрный, 16 — белый). Модель будет работать с этими 64 числами так же, как с табличными признаками.

##### Предобработка

Даже если мы используем сырые пиксели, изображения нужно привести к **единому формату**. Типичный конвейер:

1. **Изменение размера** — все изображения должны быть одного размера. Иначе из них нельзя собрать матрицу признаков.
2. **Приведение к серому** — для многих задач цвет не важен, а градиенты в сером считаются проще.
3. **Нормализация** — значения 0–255 приводятся к 0–1 или стандартизируются (`StandardScaler`).
4. **Шумоподавление** — сглаживание (Gaussian blur) убирает мелкий шум и артефакты.
5. **Контрастирование** — усиление контраста (`equalize_hist`) помогает при плохом освещении.

```python
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage import exposure

# Цветное → серое
gray_astronaut = rgb2gray(astronaut)
print('astronaut gray:', gray_astronaut.shape, gray_astronaut.dtype,
      f'[{gray_astronaut.min():.2f}, {gray_astronaut.max():.2f}]')

# Сглаживание
smoothed = gaussian(gray_astronaut, sigma=1.5)

# Выравнивание гистограммы (контрастирование)
equalized = exposure.equalize_hist(gray_astronaut)

# Изменение размера (например, к 128x128)
resized = resize(gray_astronaut, (128, 128), anti_aliasing=True)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (title, im) in zip(axes, [
    ('Серое', gray_astronaut),
    ('Сглаженное (σ=1.5)', smoothed),
    ('Equalize_hist', equalized),
    ('Resize 128×128', resized),
]):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.suptitle('Предобработка изображения')
plt.tight_layout(); plt.show()
```

**Замечание по нормализации.** Для цифр сырые значения 0–16 уже хороши. Для лиц 0–1 — тоже. Но если вы обучаете SVM, **стандартизация** (`StandardScaler`) почти всегда полезна: она выравнивает вклад пикселей, у которых разные диапазоны и распределения. Для HOG и LBP, у которых признаки уже нормированы, стандартизация обычно не нужна.

##### Простые признаки

###### Сырые пиксели как baseline

Самое простое, что можно сделать, — «вытянуть» матрицу в вектор и обучить на нём модель:

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

X_flat = digits_X.reshape(len(digits_X), -1)  # (1797, 64)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_flat, digits_y, test_size=0.3, random_state=42, stratify=digits_y
)

svm_raw = SVC(kernel='rbf', gamma='scale', C=10).fit(X_tr, y_tr)
y_pred = svm_raw.predict(X_te)

print(f'Точность на сырых пикселях: {accuracy_score(y_te, y_pred):.3f}')
print(f'F1-macro: {f1_score(y_te, y_pred, average="macro"):.3f}')
```

Для 8×8 цифр это даёт уже 97–98% точности — задача слишком простая. Для более сложных изображений (лица, реальные фото) сырые пиксели работают заметно хуже. Уже здесь стоит запомнить правило: **чем меньше изображение и чем более стандартизирована сцена, тем лучше работают сырые пиксели**.

###### Гистограммы яркости и цвета

Гистограмма яркости — это распределение значений пикселей по бинам. Она компактно описывает **общую статистику** изображения и не зависит от точного положения объектов.

```python
def intensity_histogram(image, bins=32):
    hist, _ = np.histogram(image, bins=bins, range=(0, 1))
    return hist / hist.sum()  # нормируем в вероятности

# Для набора лиц
hists_faces = np.array([intensity_histogram(img) for img in faces_X])
print('Гистограммы лиц:', hists_faces.shape)  # (400, 32)

# Для набора цифр (нормализуем 0..16 → 0..1)
digits_norm = digits_X / 16.0
hists_digits = np.array([intensity_histogram(img) for img in digits_norm])
print('Гистограммы цифр:', hists_digits.shape)  # (1797, 32)
```

**Цветовые гистограммы** — тот же принцип для цветных изображений, отдельно по каждому каналу (или в пространстве HSV):

```python
from skimage.color import rgb2hsv

def color_histogram(image_rgb, bins=16):
    hsv = rgb2hsv(image_rgb)
    hists = []
    for ch in range(3):
        h, _ = np.histogram(hsv[:, :, ch], bins=bins, range=(0, 1))
        hists.append(h / h.sum())
    return np.concatenate(hists)

# Пример: сегментация астронавта по цветам
h = color_histogram(astronaut)
print('Цветовая гистограмма астронавта:', h.shape)  # (48,)
```

**Когда гистограммы полезны:**

- Быстрая классификация сцен (пляж, лес, город) — по преобладающим цветам.
- Поиск похожих изображений — по форме гистограммы.
- Детектирование «неестественных» изображений — например, полностью тёмных или пересвеченных.

**Когда — нет:**

- Задачи, где важен объект, а не общая яркость (распознавание цифр — бесполезно).
- Задачи, где два разных объекта имеют одинаковую гистограмму (например, изображение с двумя кругами и с одним — если площади совпадают).

###### Границы (Sobel, Canny)

Границы — быстрый способ «убрать всё лишнее» и оставить только контуры объектов. Sobel вычисляет градиент по каждой оси, Canny — более сложный, но качественный детектор границ.

```python
from skimage.filters import sobel, sobel_h, sobel_v
from skimage.feature import canny

edges_sobel = sobel(camera)       # модуль градиента
edges_canny = canny(camera, sigma=1.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(camera, cmap='gray'); axes[0].set_title('Оригинал')
axes[0].axis('off')
axes[1].imshow(edges_sobel, cmap='gray'); axes[1].set_title('Sobel')
axes[1].axis('off')
axes[2].imshow(edges_canny, cmap='gray'); axes[2].set_title('Canny')
axes[2].axis('off')
plt.suptitle('Выделение границ')
plt.tight_layout(); plt.show()
```

Границы сами по себе редко используются как признаки для классификации, но они полезны:

- **Как этап предобработки** — например, для улучшения качества сырых пикселей в тёмных сценах.
- **Как вход для других дескрипторов** — HOG и LBP используют градиенты.
- **Для сегментации** — границы помогают отделить объекты друг от друга.

##### Специализированные дескрипторы: HOG и LBP

Сырые пиксели и гистограммы плохо описывают **структуру** изображения. HOG и LBP извлекают локальные характеристики — направления градиентов и текстуру — и собирают их в компактный вектор.

###### HOG — Histogram of Oriented Gradients

**Идея.** Изображение разбивается на **ячейки** (обычно 8×8 пикселей). В каждой ячейке:

1. Считаются градиенты по X и Y (через Sobel).
2. Каждый градиент квантуется по направлению (обычно 9 бинов, по 20° каждый).
3. Строится гистограмма направлений, взвешенная по модулю градиента.
4. Гистограммы соседних ячеек объединяются в блоки (например, 2×2) и нормируются — это делает дескриптор устойчивым к изменению освещения.
5. Все блоки конкатенируются в один длинный вектор.

Для изображения 64×64 с ячейками 8×8, блоками 2×2 и 9 бинами получается ~1764 признака. Это уже компактнее сырых 4096 пикселей, но главное — **устойчиво к небольшим сдвигам и изменению яркости**.

```python
from skimage.feature import hog

# Параметры HOG
hog_params = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=False,
)

# HOG для одного лица
hog_faces = np.array([
    hog(img, **hog_params) for img in faces_X
])
print(f'HOG для лиц: {hog_faces.shape}')  # (400, ~1764)

# HOG для цифр (ячейки 4x4 для 8x8 изображения маловаты, возьмём 2x2)
hog_params_digits = dict(
    orientations=9,
    pixels_per_cell=(2, 2),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
)
hog_digits = np.array([
    hog(img, **hog_params_digits) for img in digits_X
])
print(f'HOG для цифр: {hog_digits.shape}')  # (1797, 324)
```

Посмотрим, как выглядит HOG-визуализация — она показывает, какие направления градиентов доминируют в каждом регионе:

```python
_, hog_img = hog(faces_X[0], visualize=True, **hog_params)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(faces_X[0], cmap='gray')
axes[0].set_title('Оригинал'); axes[0].axis('off')
axes[1].imshow(hog_img, cmap='gray')
axes[1].set_title('HOG-визуализация'); axes[1].axis('off')
plt.tight_layout(); plt.show()
```

HOG-визуализация — это «портрет» изображения, где каждый пиксель — насколько сильный градиент в данном направлении. Она показывает, что HOG «видит» лицо как совокупность направленных контуров.

###### LBP — Local Binary Patterns

**Идея.** Для каждого пикселя рассматривается его **окрестность** (например, 8 соседей по кругу радиуса 1). Центральный пиксель сравнивается с каждым соседом:

- Если сосед ярче центра — ставим 1, иначе 0.
- Получается 8-битное число (0–255) — локальный шаблон для этого пикселя.
- Гистограмма этих чисел по всему изображению — дескриптор текстуры.

LBP устойчив к изменению освещения (важен только порядок «светлее/темнее», а не абсолютные значения) и хорошо описывает **текстуру** — кожу, ткань, листву.

```python
from skimage.feature import local_binary_pattern

radius = 1
n_points = 8 * radius

lbp_faces = np.array([
    local_binary_pattern(img, n_points, radius, method='uniform')
    for img in faces_X
])
print(f'LBP-изображения лиц: {lbp_faces.shape}')  # (400, 64, 64)

# Гистограмма LBP как признак
def lbp_histogram(lbp_img, n_bins=10):
    hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins), density=True)
    return hist

lbp_features = np.array([lbp_histogram(l) for l in lbp_faces])
print(f'Гистограммы LBP: {lbp_features.shape}')  # (400, 10)
```

Посмотрим, как LBP «видит» текстуру:

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(faces_X[0], cmap='gray')
axes[0].set_title('Оригинал'); axes[0].axis('off')
axes[1].imshow(lbp_faces[0], cmap='gray')
axes[1].set_title('LBP'); axes[1].axis('off')
plt.tight_layout(); plt.show()
```

**Когда HOG, когда LBP, когда оба:**

- **HOG** — для задач, где важна форма и контуры (лица, пешеходы, символы). Работает на относительно гладких изображениях.
- **LBP** — для задач, где важна текстура (материалы, ткани, листья, кожа). Работает на «шумных» изображениях.
- **Оба** — если нужно максимально полное описание. Признаки конкатенируются: `np.hstack([hog_feats, lbp_feats])`. Это часто даёт небольшой прирост качества за счёт большего числа информации, но увеличивает размерность.

##### Классификация: распознавание цифр

Соберём полный конвейер на цифрах и сравним разные признаки и модели:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def compare_features_and_models(X_features_dict, y, cv=5):
    results = []
    for feat_name, X_feat in X_features_dict.items():
        for model_name, model in [
            ('kNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
            ('LogReg (C=1)', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=2000, C=1.0))
            ])),
            ('SVM (RBF)', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='rbf', C=10, gamma='scale'))
            ])),
            ('RandomForest', RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            )),
        ]:
            scores = cross_val_score(
                model, X_feat, y, cv=cv, scoring='accuracy', n_jobs=-1
            )
            results.append({
                'Признаки': feat_name,
                'Модель': model_name,
                'Accuracy': f'{scores.mean():.3f} ± {scores.std():.3f}',
            })
    return pd.DataFrame(results)

features_digits = {
    'Сырые пиксели (64)': digits_X.reshape(len(digits_X), -1),
    'Гистограммы яркости (32)': hists_digits,
    'HOG (324)': hog_digits,
}

pd.set_option('display.max_colwidth', 40)
df_results = compare_features_and_models(features_digits, digits_y)
df_results
```

**Что вы должны увидеть:**

- **Сырые пиксели** работают очень хорошо: цифры уже выровнены и нормализованы, задача — «игрушечная».
- **HOG** даёт сопоставимое качество при **меньшей** размерности (324 против 64? нет, 324 против 64 — HOG больше; но всё равно компактно и часто чуть точнее, потому что устойчив к малым вариациям).
- **Гистограммы яркости** работают заметно хуже, потому что теряют пространственную структуру: разные цифры могут иметь одинаковую гистограмму.
- **RandomForest** немного отстаёт от SVM/kNN на сырых пикселях, но на HOG работает конкурентно.

**Важный вывод:** на простой задаче «игрушечные» признаки выигрывают. На реальных данных, где изображения больше и вариативнее, HOG и LBP начинают выигрывать у сырых пикселей.

##### Идентификация: распознавание лиц

Задача — по фотографии определить, кто из 40 человек изображён. Каждому человеку соответствует по 10 снимков. Это классическая задача **идентификации** (в отличие от верификации — «это один и тот же человек?»).

Базовый конвейер:

```python
from sklearn.model_selection import GridSearchCV

X_faces_hog = hog_faces
y_faces = faces_y

X_tr, X_te, y_tr, y_te = train_test_split(
    X_faces_hog, y_faces, test_size=0.3, random_state=42, stratify=y_faces
)

# SVM с RBF-ядром + grid search
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf'))
])

param_grid = {
    'clf__C': [1, 10, 100],
    'clf__gamma': ['scale', 0.001, 0.01],
}

grid = GridSearchCV(svm_pipe, param_grid, cv=5, scoring='accuracy',
                     n_jobs=-1, verbose=0)
grid.fit(X_tr, y_tr)

print(f'Лучшие параметры: {grid.best_params_}')
print(f'CV accuracy: {grid.best_score_:.3f}')
print(f'Test accuracy: {grid.score(X_te, y_te):.3f}')
```

Сравним разные признаки на задаче распознавания лиц:

```python
features_faces = {
    'Сырые пиксели (4096)': faces_X.reshape(len(faces_X), -1),
    'Гистограммы яркости (32)': hists_faces,
    'HOG (1764)': hog_faces,
    'LBP-гистограммы (10)': lbp_features,
    'HOG + LBP': np.hstack([hog_faces, lbp_features]),
}

df_faces = compare_features_and_models(features_faces, y_faces, cv=5)
df_faces
```

**Ожидаемые результаты:**

- **Сырые пиксели** — 80–90% accuracy. Работают хуже, чем на цифрах, потому что лица имеют большую вариативность (освещение, поза, очки).
- **HOG** — 90–95%. Значительно лучше: HOG «фокусируется» на структуре лица, устойчив к изменению яркости.
- **LBP-гистограммы** — 70–80%. Одного LBP мало: он теряет пространственную структуру лица.
- **HOG + LBP** — часто лучше, чем каждый по отдельности, но не всегда: иногда LBP добавляет шум.
- **Гистограммы яркости** — низкое качество: лица разных людей часто имеют похожее распределение яркости.

**Визуализация ошибок.** Полезно понять, на каких людях модель ошибается чаще:

```python
y_pred = grid.predict(X_te)
cm = pd.crosstab(y_te, y_pred)

# Процент правильных классификаций для каждого человека
correct_rate = (cm.values.diagonal() / cm.values.sum(axis=1))
worst_people = np.argsort(correct_rate)[:5]

print('Люди, которых модель чаще всего путает:')
for p_idx in worst_people:
    person_id = cm.index[p_idx]
    print(f'  Человек №{person_id}: accuracy = {correct_rate[p_idx]:.2f}')
```

**Почему модель путает одних людей чаще, чем других?** Причины:

- Схожая внешность (по структуре лица, по очкам, по причёске).
- Мало разноплановых фотографий (все 10 снимков одного человека могут быть с похожего ракурса).
- Плохое освещение на части фото.

Это демонстрирует важное ограничение классических методов: они работают «по пикселям и градиентам», не «понимая» семантику лица. Глубокие сети (следующий курс) решают эту проблему.

##### 7. Сегментация

Сегментация — это разбиение изображения на области. В отличие от классификации (один ответ на всё изображение), здесь ответ — **маска**, где каждому пикселю присвоен класс.

Разберём три классических подхода.

###### Пороговая сегментация (Otsu)

**Идея.** Если на изображении объекты отделены от фона по яркости, можно подобрать порог: пиксели ярче порога — объект, темнее — фон. **Метод Оцу** выбирает порог автоматически, максимизируя межклассовую дисперсию.

```python
from skimage.filters import threshold_otsu

img = coins
threshold = threshold_otsu(img)
binary = img > threshold

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('Оригинал'); axes[0].axis('off')
axes[1].hist(img.ravel(), bins=64);
axes[1].axvline(threshold, color='r', linestyle='--', label=f'Порог = {threshold}')
axes[1].set_title('Гистограмма яркости'); axes[1].legend()
axes[2].imshow(binary, cmap='gray'); axes[2].set_title('Otsu'); axes[2].axis('off')
plt.tight_layout(); plt.show()
```

**Когда работает:** изображение с двумя чёткими классами яркости (объект и фон), равномерное освещение.

**Когда не работает:** сложный фон, неравномерное освещение, объекты сливаются с фоном.

###### K-means по цвету (или по яркости)

**Идея.** Воспринимаем каждый пиксель как точку в пространстве признаков (цвет или яркость) и кластеризуем все пиксели на $k$ групп. Каждая группа — сегмент изображения.

Хорошо работает для цветных изображений, где объекты отличаются по цвету.

```python
from sklearn.cluster import KMeans

# Для цветного изображения: пиксель = (R, G, B)
pixels = astronaut.reshape(-1, 3) / 255.0  # (n_pixels, 3)
print(f'Пикселей: {len(pixels)}')

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10).fit(pixels)
labels = kmeans.labels_.reshape(astronaut.shape[:2])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(astronaut); axes[0].set_title('Оригинал'); axes[0].axis('off')
axes[1].imshow(labels, cmap='tab10'); axes[1].set_title('K-means (k=6)')
axes[1].axis('off')
plt.tight_layout(); plt.show()

# Восстановленное изображение (каждый пиксель заменён центром своего кластера)
centers = (kmeans.cluster_centers_ * 255).astype(np.uint8)
reconstructed = centers[labels]
plt.figure(figsize=(8, 8))
plt.imshow(reconstructed); plt.title('Изображение, восстановленное по кластерам')
plt.axis('off'); plt.show()
```

**Когда работает:** объекты хорошо выделяются по цвету; можно взять k = число «основных цветов» на сцене.

**Когда не работает:** много мелких объектов, объекты с плавными цветовыми переходами; выбор k субъективен.

###### Водораздел (Watershed)

**Идея.** Воспринимаем изображение как «рельеф»: яркие области — горы, тёмные — долины. «Заливаем водой» из локальных минимумов; где вода из разных минимумов встречается, там проходит граница между объектами. Это позволяет **разделить слипшиеся объекты**, что не получается у пороговых методов.

Классический пример — слипшиеся монеты:

```python
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

img = coins

# Шаг 1. Бинарная маска
th = threshold_otsu(img)
binary = img > th

# Шаг 2. Расстояние до фона — каждому пикселю объекта
# присваиваем расстояние до ближайшего фонового пикселя.
# В центрах объектов это расстояние максимально.
distance = ndi.distance_transform_edt(binary)

# Шаг 3. Локальные максимумы — «семена» для водораздела
coords = peak_local_max(distance, min_distance=20, labels=binary)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)

# Шаг 4. Водораздел
labels_ws = watershed(-distance, markers, mask=binary)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('Оригинал'); axes[0].axis('off')
axes[1].imshow(binary, cmap='gray'); axes[1].set_title('Otsu (монеты слиплись)')
axes[1].axis('off')
axes[2].imshow(distance, cmap='viridis'); axes[2].set_title('Distance transform')
axes[2].axis('off')
axes[3].imshow(labels_ws, cmap='nipy_spectral'); axes[3].set_title('Watershed')
axes[3].axis('off')
plt.tight_layout(); plt.show()

# Сколько объектов нашли
n_coins_otsu = ndi.label(binary)[1]
n_coins_ws = len(np.unique(labels_ws)) - 1
print(f'Otsu: {n_coins_otsu} областей')
print(f'Watershed: {n_coins_ws} монет')
```

**Когда работает:** объекты имеют примерно одинаковый размер, частично перекрываются, фон контрастный.

**Когда не работает:** объекты сильно разного размера, шумный фон, «слипание» по неконтрастной границе.

##### Оценка качества

###### Метрики классификации

Для задач распознавания и идентификации работают стандартные метрики:

- **Accuracy** — доля правильных ответов. Работает, если классы сбалансированы.
- **F1-macro** — усреднённый F1 по классам. Устойчив к дисбалансу.
- **Confusion matrix** — какие классы модель путает.

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = grid.predict(X_te)
print(classification_report(y_te, y_pred))

plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_te, y_pred), cmap='Blues', cbar=False)
plt.title('Матрица ошибок (идентификация лиц)')
plt.xlabel('Предсказано'); plt.ylabel('Истина')
plt.tight_layout(); plt.show()
```

###### Метрики сегментации

Для сегментации стандартные метрики классификации не подходят, потому что важно **попиксельное** совпадение. Основные метрики:

- **IoU (Intersection over Union, индекс Жаккара)**:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

- **Dice (коэффициент Сёренсена)**:

$$\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}$$

Обе лежат в [0, 1], где 1 — идеальное совпадение. Обычно для многоклассовой сегментации считают **среднее по классам**.

```python
def iou(pred_mask, true_mask):
    inter = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return inter / union if union > 0 else 0.0

def dice(pred_mask, true_mask):
    inter = np.logical_and(pred_mask, true_mask).sum()
    return 2 * inter / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) > 0 else 0.0

def evaluate_multiclass_segmentation(pred, gt):
    classes = np.unique(gt)
    classes = classes[classes != 0]  # 0 — фон, пропускаем
    ious, dices = [], []
    for c in classes:
        iou_c = iou(pred == c, gt == c)
        dice_c = dice(pred == c, gt == c)
        ious.append(iou_c)
        dices.append(dice_c)
        print(f'  Класс {c}: IoU = {iou_c:.3f}, Dice = {dice_c:.3f}')
    print(f'  Mean IoU = {np.mean(ious):.3f}, Mean Dice = {np.mean(dices):.3f}')
    return np.mean(ious), np.mean(dices)
```

Применим к синтетическому изображению:

```python
# Пороговая сегментация (3 класса — по 3 уровням яркости)
# Используем встроенный multi-Otsu
from skimage.filters import threshold_multiotsu

thresholds = threshold_multiotsu(seg_syn, classes=4)
seg_pred = np.digitize(seg_syn, bins=thresholds)  # 0,1,2,3

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(seg_syn, cmap='gray'); axes[0].set_title('Оригинал')
axes[0].axis('off')
axes[1].imshow(seg_pred, cmap='tab10'); axes[1].set_title('Multi-Otsu')
axes[1].axis('off')
plt.tight_layout(); plt.show()

print('Оценка сегментации:')
evaluate_multiclass_segmentation(seg_pred, seg_gt)
```

Попробуем k-means на пикселях синтетического изображения:

```python
pixels = seg_syn.reshape(-1, 1).astype(float)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(pixels)
seg_kmeans = kmeans.labels_.reshape(seg_syn.shape)

# Метки KMeans произвольны; сопоставим их с GT по совпадению с большинством
from scipy.optimize import linear_sum_assignment

n_classes = len(np.unique(seg_gt))
confusion = np.zeros((n_classes, n_classes), dtype=int)
for pred_c in range(n_classes):
    for true_c in range(n_classes):
        confusion[pred_c, true_c] = np.sum(
            (seg_kmeans == pred_c) & (seg_gt == true_c)
        )
row_ind, col_ind = linear_sum_assignment(-confusion)
remap = {int(p): int(t) for p, t in zip(row_ind, col_ind)}
seg_kmeans_aligned = np.vectorize(lambda x: remap.get(int(x), x))(seg_kmeans)

print('K-means сегментация:')
evaluate_multiclass_segmentation(seg_kmeans_aligned, seg_gt)
```

На синтетическом изображении с чёткими уровнями яркости оба метода работают неплохо. В реальных задачах всё сложнее: шум, плавные переходы, размытые границы.

#### Задания для самостоятельного выполнения

Все задания выполняются в ноутбуке. Для каждого пункта — код и краткий вывод (2–3 предложения с интерпретацией).

##### Задание 1. Распознавание на более сложных данных

1. Загрузите один из наборов:
   - **Fashion-MNIST** (28×28, 10 классов одежды) — `sklearn.datasets.fetch_openml('Fashion-MNIST')`, ~30 МБ;
   - **CIFAR-10** (32×32×3, 10 классов объектов) — скачивается с официального сайта.
   
   Для CIFAR-10 можно работать с 10 000–20 000 изображений.
2. Сравните три подхода: **сырые пиксели**, **HOG**, **HOG + цветовые гистограммы**. Используйте SVM и RandomForest.
3. Оцените accuracy, F1-macro и время обучения. Сделайте вывод: какой подход лучше и почему.
4. Визуализируйте несколько примеров, которые модель классифицировала неправильно. Что у них общего?

##### Задание 2. Идентификация и верификация лиц

1. На наборе Olivetti обучите модель **идентификации** (кто это?) с SVM + HOG. Сравните качество с моделью на сырых пикселях.
2. Реализуйте задачу **верификации**: по паре фотографий определить, один ли это человек. Сформируйте пары:
   - **положительные** — два разных снимка одного человека,
   - **отрицательные** — снимки разных людей.
   
   Обучите бинарный классификатор на признаках разницы (например, `np.abs(hog_1 - hog_2)`) или на конкатенации. Оцените accuracy, ROC-AUC.
3. Сравните accuracy идентификации и верификации. Какая задача сложнее и почему?

##### Задание 3. Обнаружение объектов (sliding window)

1. Используйте набор цифр. Создайте **сцену** — большое изображение (например, 64×64), в которое случайно вставлены несколько цифр.
2. Реализуйте **обнаружение** через **скользящее окно**: обучите классификатор на цифрах 8×8, затем «пройдите» окном 8×8 по всей сцене (с шагом 1–2 пикселя), классифицируйте каждый патч, оставьте только предсказания с высокой уверенностью.
3. Реализуйте **non-max suppression**: если два предсказания перекрываются и указывают на один объект — оставьте только одно.
4. Оцените точность обнаружения (precision, recall) для 10 сцен. Визуализируйте найденные объекты прямоугольниками на изображении.

*Подсказка:* для non-max suppression можно использовать `skimage.feature.peak_local_max` или написать свою функцию.

##### Задание 4. Сравнение методов сегментации

1. На синтетическом изображении (`seg_synthetic`) и на реальных `coins` сравните три метода сегментации:
   - порог Оцу,
   - k-means по яркости (k = число классов),
   - watershed.
2. Для синтетического изображения (есть ground truth) посчитайте IoU и Dice. Заполните таблицу «метод × метрика».
3. Для `coins` (нет GT) посчитайте **число найденных объектов**. Сравните с тем, сколько монет на изображении (посмотрите глазами). Какой метод точнее?
4. Постройте графики для всех трёх методов в одной фигуре. Сделайте вывод о том, в каких случаях какой метод предпочтителен.

##### Задание 5. Цветовая сегментация

1. Загрузите несколько реальных фотографий (например, из `skimage.data`: `coffee`, `chelsea`, `rocket`, `immunohistochemistry`). Загрузите также свои 3–5 фото из галереи.
2. Реализуйте сегментацию через k-means **в разных цветовых пространствах**: RGB, HSV, Lab.
3. Сравните результаты визуально. В каком пространстве объекты отделяются лучше? Почему?
4. Для одного изображения постройте сегментацию для k = 2, 3, 5, 8. Оцените визуально, при каком k сегментация «осмысленна».

##### Задание 6. Сравнение дескрипторов

1. Для набора лиц постройте матрицу «дескриптор × модель» (HOG, LBP-гистограмма, HOG + LBP, сырые пиксели) × (kNN, LogReg, SVM, RandomForest). Заполните accuracy по 5-fold CV.
2. Повторите то же для набора цифр.
3. Сравните таблицы. На какой задаче дескрипторы дают больше выигрыша? Почему?
4. Оцените **скорость** извлечения признаков: замерьте, сколько времени занимает HOG, LBP, гистограммы на 400 изображениях. Какой дескриптор самый быстрый?

##### Задание 7. Аугментация изображений

1. Реализуйте базовые аугментации:
   - поворот на ±15°,
   - горизонтальное отражение,
   - добавление гауссова шума,
   - изменение яркости,
   - случайный сдвиг.
2. Обучите классификатор на цифрах **без аугментации**. Затем — **с аугментацией** (увеличьте обучающую выборку в 3–5 раз, каждому изображению — несколько случайных преобразований).
3. Сравните accuracy на **неизменённом** тесте и на тесте, где все изображения повёрнуты на 10–15°. Улучшает ли аугментация устойчивость модели к трансформациям?
4. Почему аугментация — важный приём в реальных задачах? Как она связана с переобучением?

##### Задание 8. Уменьшение размерности и визуализация

1. Возьмите набор лиц (400 изображений, 40 классов). Извлеките HOG-признаки.
2. Примените **PCA** (или t-SNE / UMAP) и спроецируйте изображения на 2D-плоскость. Раскрасьте точки по классам.
3. Совпадают ли точки одного человека? Перекрываются ли разные люди? Что это говорит о сложности задачи?
4. Сравните визуализации для **сырых пикселей**, **HOG** и **LBP-гистограмм**. В каком случае классы разделены лучше?

##### Задание 9. Кластеризация изображений

1. Используйте набор цифр **без меток** (представьте, что разметки нет). Примените k-means (k = 10) к HOG-признакам.
2. Сопоставьте найденные кластеры с истинными метками (через матрицу сопоставления или Hungarian algorithm).
3. Какие цифры алгоритм находит «хорошо», а какие — путает?
4. Сравните с кластеризацией на сырых пикселях. Какой признак лучше подходит для обучения без учителя?

##### Задание 10. Полный конвейер

1. Соберите **production-ready** конвейер для классификации изображений в виде `sklearn.Pipeline`. Включите в него:
   - предобработку (resize, grayscale, normalization) — реализуйте как `FunctionTransformer`;
   - извлечение HOG-признаков — тоже как `FunctionTransformer` или кастомный трансформер;
   - классификатор.
2. Подберите гиперпараметры через `GridSearchCV` (C для SVM, размер ячейки HOG, число бинов).
3. Сохраните пайплайн через `joblib`. Загрузите в отдельной ячейке и примените к 5 новым изображениям.
4. Напишите функцию `predict_image(image, pipeline)`, которая принимает `np.array` изображения (произвольного размера!) и возвращает предсказанный класс и топ-3 вероятности.

#### Контрольные вопросы

1. Чем изображение отличается от табличных данных? Какие проблемы это создаёт для классических моделей ML?
2. Что такое «проклятие размерности» и как оно проявляется в задачах анализа изображений?
3. Зачем нужно приведение к серому? В каких задачах цвет, наоборот, полезен?
4. Чем отличается нормализация (0–1) от стандартизации (`StandardScaler`)? Когда применять каждую?
5. Что такое гистограмма яркости? В каких задачах она полезна, а в каких — бесполезна?
6. В чём идея HOG? Почему он устойчив к изменению освещения?
7. В чём идея LBP? Почему он хорошо описывает текстуру?
8. Что общего у HOG и LBP и чем они принципиально различаются?
9. Почему на 8×8 цифрах сырые пиксели работают лучше, чем на 64×64 лицах?
10. Как работает метод Оцу? В каких случаях он даёт плохие результаты?
11. Как работает сегментация через k-means? Что такое «k» в этой задаче?
12. В чём идея алгоритма водораздела? Зачем нужен distance transform?
13. Что такое IoU и Dice? Чем они отличаются и почему обе используются?
14. Почему для сегментации нельзя использовать accuracy? Что она «скрывает»?
15. Что такое аугментация изображений и зачем она нужна?
16. Почему классические дескрипторы (HOG, LBP) хорошо работают, но глубокие сети (следующий курс) работают ещё лучше? В чём принципиальное ограничение ручных дескрипторов?

#### Дополнительные задания

1. **Своя реализация HOG.** Напишите функцию, которая вычисляет HOG-признаки шаг за шагом (Sobel → квантование направлений → гистограммы по ячейкам → нормализация по блокам). Сравните результат с `skimage.feature.hog` — должны совпасть (с точностью до нормировки).

2. **Своя реализация Otsu.** Реализуйте метод Оцу без использования `skimage.filters.threshold_otsu`: переберите все возможные пороги и найдите тот, который максимизирует межклассовую дисперсию. Сравните с библиотечной реализацией.

3. **Сравнение с классическими keypoint-дескрипторами.** Познакомьтесь с SIFT и ORB (`skimage.feature.SIFT`, `cv2.ORB_create`). Извлеките keypoints и дескрипторы из нескольких изображений, визуализируйте ключевые точки. Сравните с HOG: какие преимущества даёт keypoint-подход? Где он применим?

4. **Обнаружение через template matching.** Реализуйте поиск объекта на изображении через `skimage.feature.match_template` (нормированная кросс-корреляция). Работает ли это для разных масштабов? Для повёрнутых объектов? Сравните с методом скользящего окна.

5. **Многомасштабный анализ.** Для задачи обнаружения реализуйте sliding window **на нескольких масштабах**: перебирайте окна разных размеров. Это стандартный приём в классическом CV до появления глубокого обучения.

6. **Сравнение с глубокими признаками.** (Ознакомительно.) Используйте `torchvision.models.resnet18(pretrained=True)` как **feature extractor** (без fine-tuning): пропустите изображение через сеть, возьмите выход предпоследнего слоя (512-мерный вектор), обучите на нём классификатор. Сравните с HOG. Насколько глубокие признаки лучше? Это даёт представление о том, что будет в следующем курсе.

7. **Устойчивость модели к искажениям.** Возьмите обученный классификатор лиц. Оцените его accuracy на:
   - тесте без изменений,
   - тесте с добавленным гауссовым шумом,
   - тесте с поворотом на ±10°,
   - тесте с обрезанными по краям изображениями,
   - тесте с изменённой яркостью.
   
   Постройте график «accuracy vs степень искажения». Насколько модель устойчива? Какие искажения она переносит лучше?

8. **Классификация сцен.** Используйте `skimage.data` (или скачайте датасет сцен, например, `MIT Places365`), чтобы классифицировать сцены по общему виду. Попробуйте:
   - цветовые гистограммы,
   - HOG,
   - комбинацию признаков.
   
   Какой подход даёт лучший результат?
