### Интеграция данных

#### Цель работы

Ознакомиться с основными приемами работы с численными атрибутами в датасетах для машинного обучения.

#### Содержание работы

1. Загрузить прилагающийся к работе датасет PRSA_Data
1. Выведите на экран основную информацию о датасете. Идентифицируйте значения вне разумного диапазона
1. Постройте визуализацию распределения каждого численного атрибута.
1. Исходя из распределения атрибутов по необходимости примените бинаризацию численных признаков.
1. Постройте совместное распределение каждого признака вместе с целевой переменной. Сделайте вывод о необходимости проведения группировки данных.
1. Удалите или ограничьте экстремальные значения атрибутов. 
1. Избавьтесь от пропущенных значений в датасете.
1. При необходимости округлите излишне точные значения атрибутов.
1. Рассмотрите возможность преобразования шкалы атрибута к логарифмической.

#### Методические указания

##### Первоначальное знакомство с данными



```py
prsa_data = pd.read_csv("https://github.com/koroteevmv/ML_course/raw/main/ML5.2%20numeric%20features/PRSA_Data.csv")
prsa_data.head()
```

|index|No|SO2|NO2|CO|O3|PRES|RAIN|wd|WSPM|AQI Label|
|---|---|---|---|---|---|---|---|---|---|---|
|0|1|6\.0|28\.0|400\.0|52\.37124684346539|1023\.0|0\.0|NNW|4\.4|Severely Polluted|
|1|2|6\.0|28\.0|400\.0|50\.433574890423515|1023\.2|0\.0|N|4\.7|Severely Polluted|
|2|3|NaN|19\.0|400\.0|54\.59906675266078|1023\.5|0\.0|NNW|5\.6|Severely Polluted|
|3|4|8\.0|14\.0|NaN|NaN|1024\.5|0\.0|NW|3\.1|Excellent|
|4|5|9\.0|NaN|300\.0|53\.52974321124786|1025\.2|0\.0|N|2\.0|Heavily Polluted|



```py
prsa_data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 35064 entries, 0 to 35063
Data columns (total 10 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   No         35064 non-null  int64  
 1   SO2        35064 non-null  float64
 2   NO2        35064 non-null  float64
 3   CO         35064 non-null  float64
 4   O3         35064 non-null  float64
 5   PRES       35064 non-null  float64
 6   RAIN       35064 non-null  float64
 7   wd         35064 non-null  object 
 8   WSPM       35064 non-null  float64
 9   AQI Label  35064 non-null  object 
dtypes: float64(7), int64(1), object(2)
memory usage: 2.7+ MB
```



```py
prsa_data.describe()
```

|index|No|SO2|NO2|CO|O3|PRES|RAIN|WSPM|
|---|---|---|---|---|---|---|---|---|
|count|35064\.0|35064\.0|35064\.0|35064\.0|35064\.0|35064\.0|35064\.0|35064\.0|
|mean|17532\.5|18\.058733698380106|63\.236860663358435|1251\.1216917636323|45\.876245676104766|1010\.5202498764164|0\.06765343372119553|1\.5002167465206477|
|std|10122\.24925597073|22\.558126329870454|39\.10923159613662|1269\.0335562739235|54\.72944847600266|26\.289217123446083|0\.8968325386377156|1\.105381648786512|
|min|1\.0|-1\.0|-1\.0|-1\.0|-1\.0|-1\.0|-1\.0|-1\.0|
|25%|8766\.75|4\.0|34\.0|500\.0|2\.400026434533582|1002\.5|0\.0|0\.8|
|50%|17532\.5|10\.0|58\.0|900\.0|27\.37830570188505|1010\.8|0\.0|1\.2|
|75%|26298\.25|22\.2768|87\.0|1500\.0|69\.57736619098368|1019\.4|0\.0|2\.0|
|max|35064\.0|282\.0|264\.0|10000\.0|364\.3447496563047|1040\.3|72\.5|11\.2|



##### Идентификация ошибочных значений


```py
prsa_data[prsa_data == -1] = np.nan
prsa_data.head()
```

|index|No|SO2|NO2|CO|O3|PRES|RAIN|wd|WSPM|AQI Label|
|---|---|---|---|---|---|---|---|---|---|---|
|0|1|6\.0|28\.0|400\.0|52\.37124684346539|1023\.0|0\.0|NNW|4\.4|Severely Polluted|
|1|2|6\.0|28\.0|400\.0|50\.433574890423515|1023\.2|0\.0|N|4\.7|Severely Polluted|
|2|3|NaN|19\.0|400\.0|54\.59906675266078|1023\.5|0\.0|NNW|5\.6|Severely Polluted|
|3|4|8\.0|14\.0|NaN|NaN|1024\.5|0\.0|NW|3\.1|Excellent|
|4|5|9\.0|NaN|300\.0|53\.52974321124786|1025\.2|0\.0|N|2\.0|Heavily Polluted|



##### Визуализация распределения атрибутов



```py
sns.histplot(prsa_data.SO2)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-1.png?raw=true)



```py
sns.kdeplot(prsa_data.NO2)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-2.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-3.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-4.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-5.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-6.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-7.png?raw=true)



##### Бинаризация атрибутов

```py
prsa_data.RAIN[prsa_data.RAIN > 0]
```

```
267      0.1
268      0.4
269      0.1
270      0.9
271      0.9
        ... 
34891    0.2
34892    0.7
34893    0.9
34894    0.4
34895    0.2
Name: RAIN, Length: 1436, dtype: float64
```



```py
is_rain = np.array(prsa_data.RAIN)
is_rain[is_rain > 0] = 1
prsa_data['IS_RAIN'] = is_rain
prsa_data.drop(['RAIN'], axis=1, inplace=True)
prsa_data.describe()
```

|index|No|SO2|NO2|CO|O3|PRES|WSPM|IS\_RAIN|
|---|---|---|---|---|---|---|---|---|
|count|35064\.0|34489\.0|33994\.0|33252\.0|32957\.0|35044\.0|35050\.0|35044\.0|
|mean|17532\.5|18\.376480570616717|65\.25878926575278|1319\.3535125706724|48\.8731279663482|1011\.0975357170033|1\.501215406562054|0\.040977057413537264|
|std|10122\.24925597073|22\.609647740471033|37\.99608792528771|1268\.1143306714334|55\.11211847974543|10\.355246504075161|1\.1044721447237704|0\.1982399041559329|
|min|1\.0|0\.2856|1\.6424|100\.0|-0\.24025790735349623|985\.9|0\.0|0\.0|
|25%|8766\.75|4\.0|36\.0|500\.0|3\.5767910527049533|1002\.5|0\.8|0\.0|
|50%|17532\.5|10\.0|60\.0|900\.0|31\.992632810880806|1010\.8|1\.2|0\.0|
|75%|26298\.25|23\.0|88\.0|1600\.0|72\.89152376395555|1019\.4|2\.0|0\.0|
|max|35064\.0|282\.0|264\.0|10000\.0|364\.3447496563047|1040\.3|11\.2|1\.0|

##### Визуализация связи атрибутов с целевой переменной

```py
sns.kdeplot(data=prsa_data, x="SO2", hue="AQI Label")
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-8.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-9.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-10.png?raw=true)



```py
sns.kdeplot(data=prsa_data, x="CO", hue="AQI Label")
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-11.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-12.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-13.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-14.png?raw=true)



```py
sns.kdeplot(data=prsa_data, x="CO", hue="AQI Label", log_scale=True)
```

##### Группировка численных значений

```py
sns.kdeplot(data=prsa_data, x="CO", hue="AQI Label", log_scale=True)
plt.axvline(250, 0,0.17)
plt.axvline(320, 0,0.17)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-15.png?raw=true)



```py
bin_ranges = [0, 250, 320, 10000]
bin_names = [1, 2, 3]
prsa_data['CO_bin_custom_label'] = pd.cut(np.array(prsa_data['CO']), 
                                               bins=bin_ranges, labels=bin_names)
prsa_data.head()
```



```py
prsa_data['CO_bin_custom_label'] = prsa_data['CO_bin_custom_label'].values.add_categories(0)
prsa_data['CO_bin_custom_label'] = prsa_data['CO_bin_custom_label'].fillna(0).astype(int)
prsa_data.head()
```

|index|No|SO2|NO2|CO|O3|PRES|wd|WSPM|AQI Label|IS\_RAIN|CO\_bin\_custom\_label|
|---|---|---|---|---|---|---|---|---|---|---|---|
|0|1|6\.0|28\.0|400\.0|51\.66378173181024|1023\.0|NNW|4\.4|Severely Polluted|0\.0|3|
|1|2|6\.0|28\.0|400\.0|49\.72659979230651|1023\.2|N|4\.7|Severely Polluted|0\.0|3|
|2|3|NaN|19\.0|400\.0|55\.270067369938324|1023\.5|NNW|5\.6|Severely Polluted|0\.0|3|
|3|4|8\.0|14\.0|NaN|NaN|1024\.5|NW|3\.1|Excellent|0\.0|0|
|4|5|9\.0|NaN|300\.0|54\.22212476907075|1025\.2|N|2\.0|Heavily Polluted|0\.0|2|

##### Удаление экстремальных значений

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-16.png?raw=true)

```py
prsa_data.PRES[prsa_data.PRES <= 992] = 992
prsa_data.PRES[prsa_data.PRES >= 1034] = 1034
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-17.png?raw=true)

##### Заполнение пропусков

```py
sns.heatmap(prsa_data.isnull(), yticklabels=False, cbar=False)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-18.png?raw=true)



```py
undef = prsa_data.isnull().sum(axis=1)
undef[undef >= 2]
```

```
3        2
276      3
435      2
459      2
555      2
        ..
34880    4
34883    4
34885    4
35029    2
35030    2
Length: 884, dtype: int64
```



```py
prsa_data = prsa_data.drop(undef[undef >= 2].index, axis=0)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-19.png?raw=true)



```py
prsa_data.isnull().sum()
```

```
No                        0
SO2                     281
NO2                     702
CO                     1028
O3                     1300
PRES                      0
wd                        0
WSPM                      0
AQI Label                 0
IS_RAIN                   0
CO_bin_custom_label       0
dtype: int64
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-20.png?raw=true)



```py
prsa_data.PRES = prsa_data.PRES.fillna(prsa_data.PRES.mean())
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-21.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-22.png?raw=true)



```py
filler = prsa_data.O3[prsa_data.O3.isna()]
```

```
436     NaN
460     NaN
556     NaN
652     NaN
748     NaN
         ..
33124   NaN
33220   NaN
33604   NaN
33892   NaN
35031   NaN
Name: O3, Length: 1300, dtype: float64
```



```py
filler = prsa_data.O3[~prsa_data.O3.isna()].sample(n=len(filler)).set_axis(filler.index)
```

```
436       59.726165
460        9.490829
556       52.339174
652        2.210643
748       17.802473
            ...    
33124     46.354788
33220     56.869096
33604    147.957579
33892      6.986806
35031     30.261312
Name: O3, Length: 1300, dtype: float64
```



```py
prsa_data.O3 = prsa_data.O3.fillna(filler)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-23.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-24.png?raw=true)

##### Округление атрибутов

```py
prsa_data['O3'] = np.array(np.round((prsa_data['O3'])), dtype='int')
prsa_data.head()
```

|index|No|SO2|NO2|CO|O3|PRES|wd|WSPM|AQI Label|IS\_RAIN|CO\_bin\_custom\_label|
|---|---|---|---|---|---|---|---|---|---|---|---|
|0|1|6\.0|28\.0|400\.0|52|1023\.0|NNW|4\.4|Severely Polluted|0\.0|3|
|1|2|6\.0|28\.0|400\.0|50|1023\.2|N|4\.7|Severely Polluted|0\.0|3|
|2|3|NaN|19\.0|400\.0|55|1023\.5|NNW|5\.6|Severely Polluted|0\.0|3|
|4|5|9\.0|NaN|300\.0|54|1025\.2|N|2\.0|Heavily Polluted|0\.0|2|
|5|6|8\.0|17\.0|300\.0|54|1025\.6|N|3\.7|Heavily Polluted|0\.0|2|

##### Логарифмирование атрибутов

```py
prsa_data.SO2 = np.log(prsa_data.SO2)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.2%20numeric%20features/ml52-25.png?raw=true)

#### Задания для самостоятельного выполнения

1. При выполнении 6 задания мы явно подбирали руками границы диапазона для клиппинга. Реализуйте адаптивный клиппинг через процентили.
1. Избавьтесь от оставшихся пропусков в данных. Самостоятельно выберите метод.
1. Проведите нормализацию численных признаков. Выберите наиболее подходящий вид нормализации для каждого признака.
1. Постройте кореллограмму по всем численным столбцам датасета. Сделайте вывод о значимости признаков.
1. Визуализируйте связи между признаками. Сделайте вывод об их взаимозависимости.

#### Контрольные вопросы

1. Как в датасете идентифицировать численные атрибуты?
1. Какие основные виды непрерывных распределений часто встречаются на практике анализа данных?
1. 

#### Дополнительные задания

1. В данной работе проверьте целесообразность каждого необязательного преобразования данных путем проверки, увеличивает ли данное преобразование точность модели. Проверьте на простом виде модели (логит регрессия, дерево решений или случайный лес).
1. Перед началом обработки данных разбейте датасет на тестовую и обучающую выборки. Очистите по методу из работы обучающую выборку. Повторите обработку на тестовой выборке. При этом позаботьтесь, чтобы все параметрические преобразования (клиппинг, нормализация, группировка и так далее).
1. Создайте воспроизводимый код обработки данного датасета. 
1. В датасете Customer_support, который стал результатом выполнения предыдущей работы преобразуйте все даты в абсолютные признаки. Извлеките из дат значимую информацию - день недели, время дня, день месяца. Составьте значимые временные промежутки.
1. Повторите обработку численных параметров в датасете "Титаник".
1. (*) Создайте код, реализующий алгоритм очистки данных автоматически (для любого датасета).

