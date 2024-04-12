### Инжиниринг категориальных признаков

#### Цель работы

Ознакомиться с основными приемами работы с численными атрибутами в датасетах для машинного обучения.

#### Содержание работы

1. Загрузите прилагаемые к этой работе два датасета - титаник и Customer support. Выведите основную информацию по каждому датасету и сделайте выводы.
1. Визуализируйте распределение каждого категориального признака в датасете Customer support. Учитывайте количество уникальных значений.
1. Исследуйте связь каждого признака датасета Customer support с целевой переменной. Сделайте предварительный вывод о значимости признаков.
1. Где целесообразно, проведите укрупнение категорий, путем объединения разных значений в столбце.
1. Добавьте к датасету новый столбец, содержащий агрегированную информацию, которая предположительно будет полезна для моделирования целевой переменной.
1. Заполните отсутствующие значения в датасете.
1. На примере датасета Титаник проведите преобразование категориальных переменных разных шкал в численные.
1. В датасете Customer support удалите лишние столбцы и преобразуйте все категориальные переменные через get\_dummies()

#### Методические указания

#### Первоначальное знакомство с данными



|index|Unique id|channel\_name|category|Sub-category|Customer Remarks|Order\_id|order\_date\_time|Issue\_reported at|issue\_responded|Survey\_response\_Date|Customer\_City|Product\_category|Item\_price|connected\_handling\_time|Agent\_name|Supervisor|Manager|Tenure Bucket|Agent Shift|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|7e9ae164-6a8b-4521-a2d4-58f7c9fff13f|Outcall|Product Queries|Other|NaN|c27c9bb4-fa36-4140-9f1f-21009254ffdb|NaN|01/08/2023 11:13|01/08/2023 11:47|01-Aug-23|NaN|NaN|NaN|NaN|Richard Buchanan|Mason Gupta|Jennifer Nguyen|On Job Training|Morning|5|
|1|b07ec1b0-f376-43b6-86df-ec03da3b2e16|Outcall|Product Queries|Product Specific Information|NaN|d406b0c7-ce17-4654-b9de-f08d421254bd|NaN|01/08/2023 12:52|01/08/2023 12:54|01-Aug-23|NaN|NaN|NaN|NaN|Vicki Collins|Dylan Kim|Michael Lee|\>90|Morning|5|
|2|200814dd-27c7-4149-ba2b-bd3af3092880|Inbound|Order Related|Installation/demo|NaN|c273368d-b961-44cb-beaf-62d6fd6c00d5|NaN|01/08/2023 20:16|01/08/2023 20:38|01-Aug-23|NaN|NaN|NaN|NaN|Duane Norman|Jackson Park|William Kim|On Job Training|Evening|5|
|3|eb0d3e53-c1ca-42d3-8486-e42c8d622135|Inbound|Returns|Reverse Pickup Enquiry|NaN|5aed0059-55a4-4ec6-bb54-97942092020a|NaN|01/08/2023 20:56|01/08/2023 21:16|01-Aug-23|NaN|NaN|NaN|NaN|Patrick Flores|Olivia Wang|John Smith|\>90|Evening|5|
|4|ba903143-1e54-406c-b969-46c52f92e5df|Inbound|Cancellation|Other|NaN|e8bed5a9-6933-4aff-9dc6-ccefd7dcde59|NaN|01/08/2023 10:30|01/08/2023 10:32|01-Aug-23|NaN|NaN|NaN|NaN|Christopher Sanchez|Austin Johnson|Michael Lee|0-30|Morning|5|



```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 85907 entries, 0 to 85906
Data columns (total 20 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   Unique id                85907 non-null  object 
 1   channel_name             85907 non-null  object 
 2   category                 85907 non-null  object 
 3   Sub-category             85907 non-null  object 
 4   Customer Remarks         28742 non-null  object 
 5   Order_id                 67675 non-null  object 
 6   order_date_time          17214 non-null  object 
 7   Issue_reported at        85907 non-null  object 
 8   issue_responded          85907 non-null  object 
 9   Survey_response_Date     85907 non-null  object 
 10  Customer_City            17079 non-null  object 
 11  Product_category         17196 non-null  object 
 12  Item_price               17206 non-null  float64
 13  connected_handling_time  242 non-null    float64
 14  Agent_name               85907 non-null  object 
 15  Supervisor               85907 non-null  object 
 16  Manager                  85907 non-null  object 
 17  Tenure Bucket            85907 non-null  object 
 18  Agent Shift              85907 non-null  object 
 19  CSAT Score               85907 non-null  int64  
dtypes: float64(2), int64(1), object(17)
memory usage: 13.1+ MB
```



|index|Unique id|channel\_name|category|Sub-category|Customer Remarks|Order\_id|order\_date\_time|Issue\_reported at|issue\_responded|Survey\_response\_Date|Customer\_City|Product\_category|Item\_price|connected\_handling\_time|Agent\_name|Supervisor|Manager|Tenure Bucket|Agent Shift|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|85907|85907|85907|85907|28742|67675|17214|85907|85907|85907|17079|17196|17206\.0|242\.0|85907|85907|85907|85907|85907|85907\.0|
|unique|85907|3|12|57|18231|67675|13766|30923|30262|31|1782|9|NaN|NaN|1371|40|6|5|5|NaN|
|top|7e9ae164-6a8b-4521-a2d4-58f7c9fff13f|Inbound|Returns|Reverse Pickup Enquiry|Good |c27c9bb4-fa36-4140-9f1f-21009254ffdb|09/08/2023 11:55|15/08/2023 10:59|28/08/2023 00:00|28-Aug-23|HYDERABAD|Electronics|NaN|NaN|Wendy Taylor|Carter Park|John Smith|\>90|Morning|NaN|
|freq|1|68142|44097|22389|1390|1|7|13|3378|3452|722|4706|NaN|NaN|429|4273|25261|30660|41426|NaN|
|mean|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|5660\.7748459839595|462\.400826446281|NaN|NaN|NaN|NaN|NaN|4\.242157216524847|
|std|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|12825\.728411195747|246\.29503712116792|NaN|NaN|NaN|NaN|NaN|1\.3789030546991936|
|min|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|0\.0|0\.0|NaN|NaN|NaN|NaN|NaN|1\.0|
|25%|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|392\.0|293\.0|NaN|NaN|NaN|NaN|NaN|4\.0|
|50%|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|979\.0|427\.0|NaN|NaN|NaN|NaN|NaN|5\.0|
|75%|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|2699\.75|592\.25|NaN|NaN|NaN|NaN|NaN|5\.0|
|max|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|164999\.0|1986\.0|NaN|NaN|NaN|NaN|NaN|5\.0|



|index|PassengerId|Survived|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|1|0|third|Braund, Mr\. Owen Harris|1|22\.0|1|0|A/5 21171|7\.25|NaN|S|
|1|2|1|first|Cumings, Mrs\. John Bradley \(Florence Briggs Thayer\)|0|38\.0|1|0|PC 17599|71\.2833|C85|C|
|2|3|1|third|Heikkinen, Miss\. Laina|0|26\.0|0|0|STON/O2\. 3101282|7\.925|NaN|S|
|3|4|1|first|Futrelle, Mrs\. Jacques Heath \(Lily May Peel\)|0|35\.0|1|0|113803|53\.1|C123|S|
|4|5|0|third|Allen, Mr\. William Henry|1|35\.0|0|0|373450|8\.05|NaN|S|



|index|Name|Ticket|Cabin|Embarked|
|---|---|---|---|---|
|count|891|891|204|889|
|unique|891|681|147|3|
|top|Braund, Mr\. Owen Harris|347082|B96 B98|S|
|freq|1|7|4|644|



#### Визуализация распределения атрибутов и связь с целевой переменной

```py
sns.histplot(data=CS_data, x="channel_name")
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-1.png?raw=true)

```py
CS_data.channel_name.value_counts()
```

```
channel_name
Inbound    68142
Outcall    14742
Email       3023
Name: count, dtype: int64
```

```py
sns.catplot(data=CS_data, x="channel_name", y="CSAT Score", kind="bar")
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-1.png?raw=true)

```
category
Returns               44097
Order Related         23215
Refund Related         4550
Product Queries        3692
Shopzilla Related      2792
Payments related       2327
Feedback               2294
Cancellation           2212
Offers & Cashback       480
Others                   99
App/website              84
Onboarding related       65
Name: count, dtype: int64
```

```py
counts = CS_data.category.value_counts()
sns.barplot(x=counts.index, y=counts.values)
plt.xticks(rotation=45)
plt.show()
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-3.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-4.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-5.png?raw=true)

```
Sub-category
Reverse Pickup Enquiry          22389
Return request                   8523
Delayed                          7388
Order status enquiry             6922
Installation/demo                4116
Fraudulent User                  4108
Product Specific Information     3589
Refund Enquiry                   2665
Wrong                            2597
Missing                          2556
Name: count, dtype: int64
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-6.png?raw=true)

```
Customer Remarks
Good             1390
Good             1158
Very good         569
Nice              316
Thanks            276
Ok                259
No                258
Thank you         244
Nice              239
Very good         236
Excellent         171
Thanks            159
Good ??           148
Good service      133
Very nice         122
Thank you          97
??                 95
Nothing            88
5                  76
Good job           71
Name: count, dtype: int64
```

```
Customer_City
HYDERABAD      722
NEW DELHI      688
PUNE           435
MUMBAI         406
BANGALORE      352
CHENNAI        271
KOLKATA        270
LUCKNOW        254
AHMEDABAD      253
JAIPUR         243
GURGAON        215
PATNA          199
SURAT          175
ALLAHABAD      161
KANPUR         138
VARANASI       137
THANE          129
GHAZIABAD      120
BHUBANESWAR    117
VADODARA       105
Name: count, dtype: int64
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-7.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-8.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-9.png?raw=true)

```
Agent_name
Wendy Taylor           429
Timothy Huff           265
David Smith            264
Jamie Smith            253
Kayla Wilson           216
Julie Williams         200
Mrs. Jennifer Stone    200
Sharon Bullock         195
Matthew White PhD      192
Anthony Booth          177
Tina Harrington        177
Kristin Campbell       176
Brianna Wolf           176
Rebecca Walker         176
Jennifer Hernandez     174
Rebecca Graham         173
William Carey DVM      169
Ryan Thompson          167
Brandon Frost          161
Brian Young            160
Name: count, dtype: int64
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-10.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-11.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-12.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-13.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-14.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-15.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-16.png?raw=true)

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.3%20categorical%20features/img/ml53-17.png?raw=true)

#### Укрупнение категорий

```py
CS_data["Customer Remarks New"] = (CS_data["Customer Remarks"].str.len() > 3).astype(int)
```

```
0        0
1        0
2        0
3        0
4        0
        ..
85902    0
85903    1
85904    1
85905    0
85906    0
Name: Customer Remarks New, Length: 85907, dtype: int64
```

```py
CS_data["Is_order"] = (CS_data["Order_id"].isna()).astype(int)
```

```
0        0
1        0
2        0
3        0
4        0
        ..
85902    0
85903    0
85904    0
85905    0
85906    0
Name: Is_order, Length: 85907, dtype: int64
```

```py
CS_data.loc[~CS_data["Sub-category"].isin([
    "Reverse Pickup Enquiry", "Return request", "Delayed", "Order status enquiry", 
    "Installation/demo", "Fraudulent User", "Product Specific Information"
    ]), "Sub-category"] = "Other"
```

```
Sub-category
Other                           28872
Reverse Pickup Enquiry          22389
Return request                   8523
Delayed                          7388
Order status enquiry             6922
Installation/demo                4116
Fraudulent User                  4108
Product Specific Information     3589
Name: count, dtype: int64
```

#### Добавление агрегированной информации

```py
CS_data.groupby(["Agent_name"]).agg({'Agent_name': 'count'})
```



```py
CS_data['Agent_count'] = CS_data.groupby(["Agent_name"])["Agent_name"].transform('count')
```

|index|Unique id|channel\_name|category|Sub-category|Customer Remarks|Order\_id|order\_date\_time|Issue\_reported at|issue\_responded|Survey\_response\_Date|Customer\_City|Product\_category|Item\_price|connected\_handling\_time|Agent\_name|Supervisor|Manager|Tenure Bucket|Agent Shift|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|7e9ae164-6a8b-4521-a2d4-58f7c9fff13f|Outcall|Product Queries|Other|NaN|c27c9bb4-fa36-4140-9f1f-21009254ffdb|NaN|01/08/2023 11:13|01/08/2023 11:47|01-Aug-23|NaN|NaN|NaN|NaN|Richard Buchanan|Mason Gupta|Jennifer Nguyen|On Job Training|Morning|5|
|1|b07ec1b0-f376-43b6-86df-ec03da3b2e16|Outcall|Product Queries|Product Specific Information|NaN|d406b0c7-ce17-4654-b9de-f08d421254bd|NaN|01/08/2023 12:52|01/08/2023 12:54|01-Aug-23|NaN|NaN|NaN|NaN|Vicki Collins|Dylan Kim|Michael Lee|\>90|Morning|5|
|2|200814dd-27c7-4149-ba2b-bd3af3092880|Inbound|Order Related|Installation/demo|NaN|c273368d-b961-44cb-beaf-62d6fd6c00d5|NaN|01/08/2023 20:16|01/08/2023 20:38|01-Aug-23|NaN|NaN|NaN|NaN|Duane Norman|Jackson Park|William Kim|On Job Training|Evening|5|
|3|eb0d3e53-c1ca-42d3-8486-e42c8d622135|Inbound|Returns|Reverse Pickup Enquiry|NaN|5aed0059-55a4-4ec6-bb54-97942092020a|NaN|01/08/2023 20:56|01/08/2023 21:16|01-Aug-23|NaN|NaN|NaN|NaN|Patrick Flores|Olivia Wang|John Smith|\>90|Evening|5|
|4|ba903143-1e54-406c-b969-46c52f92e5df|Inbound|Cancellation|Other|NaN|e8bed5a9-6933-4aff-9dc6-ccefd7dcde59|NaN|01/08/2023 10:30|01/08/2023 10:32|01-Aug-23|NaN|NaN|NaN|NaN|Christopher Sanchez|Austin Johnson|Michael Lee|0-30|Morning|5|

```py
CS_data.groupby(["Supervisor"]).agg({'Agent_name': 'nunique'})
```

|Supervisor|Agent\_name|
|---|---|
|Abigail Suzuki|38|
|Aiden Patel|41|
|Alexander Tanaka|15|
|Amelia Tanaka|19|
|Austin Johnson|29|
|Ava Wong|70|
|Brayden Wong|45|
|Carter Park|64|
|Charlotte Suzuki|22|
|Dylan Kim|17|
|Elijah Yamaguchi|59|
|Emily Yamashita|42|
|Emma Park|55|
|Ethan Nakamura|25|
|Ethan Tan|31|
|Evelyn Kimura|39|
|Harper Wong|22|
|Isabella Wong|17|
|Jackson Park|46|
|Jacob Sato|29|
|Landon Tanaka|28|
|Layla Taniguchi|14|
|Lily Chen|25|
|Logan Lee|39|
|Lucas Singh|20|
|Madison Kim|37|
|Mason Gupta|41|
|Mia Patel|62|
|Mia Yamamoto|8|
|Nathan Patel|50|
|Noah Patel|50|
|Oliver Nguyen|6|
|Olivia Suzuki|44|
|Olivia Wang|28|
|Scarlett Chen|31|
|Sophia Chen|5|
|Sophia Sato|22|
|William Park|36|
|Wyatt Kim|32|
|Zoe Yamamoto|68|

```py
CS_data['Sups_no_agents'] = CS_data.groupby(["Supervisor"])["Agent_name"].transform('nunique')
```

#### Заполнение отсутствующих значений

```py
CS_data['Product_category'] = CS_data['Product_category'].fillna('unknown')
```

```py
CS_data['connected_handling_time'] = CS_data['connected_handling_time'].fillna('0')
```

#### Преобразование бинарных атрибутов

```py
from sklearn.preprocessing import LabelEncoder
LE_sex = LabelEncoder()
T_data.Sex = LE_sex.fit_transform(T_data.Sex)
```

|index|PassengerId|Survived|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|1|0|3|Braund, Mr\. Owen Harris|1|22\.0|1|0|A/5 21171|7\.25|NaN|S|
|1|2|1|1|Cumings, Mrs\. John Bradley \(Florence Briggs Thayer\)|0|38\.0|1|0|PC 17599|71\.2833|C85|C|
|2|3|1|3|Heikkinen, Miss\. Laina|0|26\.0|0|0|STON/O2\. 3101282|7\.925|NaN|S|
|3|4|1|1|Futrelle, Mrs\. Jacques Heath \(Lily May Peel\)|0|35\.0|1|0|113803|53\.1|C123|S|
|4|5|0|3|Allen, Mr\. William Henry|1|35\.0|0|0|373450|8\.05|NaN|S|

#### Преобразование порядковых атрибутов

```py
T_data.Pclass.replace({
    'first': 1, 'second': 2, 'third': 3
}, inplace=True)
```

|index|PassengerId|Survived|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|1|0|3|Braund, Mr\. Owen Harris|1|22\.0|1|0|A/5 21171|7\.25|NaN|S|
|1|2|1|1|Cumings, Mrs\. John Bradley \(Florence Briggs Thayer\)|0|38\.0|1|0|PC 17599|71\.2833|C85|C|
|2|3|1|3|Heikkinen, Miss\. Laina|0|26\.0|0|0|STON/O2\. 3101282|7\.925|NaN|S|
|3|4|1|1|Futrelle, Mrs\. Jacques Heath \(Lily May Peel\)|0|35\.0|1|0|113803|53\.1|C123|S|
|4|5|0|3|Allen, Mr\. William Henry|1|35\.0|0|0|373450|8\.05|NaN|S|

#### Преобразование номинальных атрибутов

```py
from sklearn.preprocessing import OneHotEncoder
OH_embarked = OneHotEncoder(sparse_output=False)
OH_embarked.fit_transform(T_data[['Embarked']])
```

```py
OH_embarked.get_feature_names_out(['Embarked'])
```

```py
dummies = pd.DataFrame(OH_embarked.fit_transform(T_data[['Embarked']]),
                       columns=OH_embarked.get_feature_names_out(['Embarked']), 
                       index = T_data.index)
```

|index|Embarked\_C|Embarked\_Q|Embarked\_S|Embarked\_nan|
|---|---|---|---|---|
|0|0\.0|0\.0|1\.0|0\.0|
|1|1\.0|0\.0|0\.0|0\.0|
|2|0\.0|0\.0|1\.0|0\.0|
|3|0\.0|0\.0|1\.0|0\.0|
|4|0\.0|0\.0|1\.0|0\.0|

```py
T_dummies = pd.concat([T_data, dummies]).drop(["Embarked"], axis=1)
```

|index|PassengerId|Survived|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked\_C|Embarked\_Q|Embarked\_S|Embarked\_nan|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|1\.0|0\.0|3\.0|Braund, Mr\. Owen Harris|1\.0|22\.0|1\.0|0\.0|A/5 21171|7\.25|NaN|NaN|NaN|NaN|NaN|
|1|2\.0|1\.0|1\.0|Cumings, Mrs\. John Bradley \(Florence Briggs Thayer\)|0\.0|38\.0|1\.0|0\.0|PC 17599|71\.2833|C85|NaN|NaN|NaN|NaN|
|2|3\.0|1\.0|3\.0|Heikkinen, Miss\. Laina|0\.0|26\.0|0\.0|0\.0|STON/O2\. 3101282|7\.925|NaN|NaN|NaN|NaN|NaN|
|3|4\.0|1\.0|1\.0|Futrelle, Mrs\. Jacques Heath \(Lily May Peel\)|0\.0|35\.0|1\.0|0\.0|113803|53\.1|C123|NaN|NaN|NaN|NaN|
|4|5\.0|0\.0|3\.0|Allen, Mr\. William Henry|1\.0|35\.0|0\.0|0\.0|373450|8\.05|NaN|NaN|NaN|NaN|NaN|

#### Удаление лишних столбцов и массовое преобразование

```py
CS_dropped = CS_data.drop([
    "Unique id",
    "Sub-category",
    "Customer Remarks",
    "Customer_City", 
    "Agent_name", 
    "Supervisor",
    "Order_id",
    "order_date_time",
    "Issue_reported at",
    "issue_responded",
    "Survey_response_Date",
    "Item_price",

], axis=1)
```

|index|channel\_name|category|Product\_category|connected\_handling\_time|Manager|Tenure Bucket|Agent Shift|CSAT Score|Customer Remarks New|Is\_order|Agent\_count|Sups\_no\_agents|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|Outcall|Product Queries|unknown|0|Jennifer Nguyen|On Job Training|Morning|5|0|0|42|41|
|1|Outcall|Product Queries|unknown|0|Michael Lee|\>90|Morning|5|0|0|32|17|
|2|Inbound|Order Related|unknown|0|William Kim|On Job Training|Evening|5|0|0|35|46|
|3|Inbound|Returns|unknown|0|John Smith|\>90|Evening|5|0|0|48|28|
|4|Inbound|Cancellation|unknown|0|Michael Lee|0-30|Morning|5|0|0|124|29|

|index|channel\_name|category|Product\_category|connected\_handling\_time|Manager|Tenure Bucket|Agent Shift|CSAT Score|Customer Remarks New|Is\_order|Agent\_count|Sups\_no\_agents|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|85907|85907|85907|85907|85907|85907|85907|85907\.0|85907\.0|85907\.0|85907\.0|85907\.0|
|unique|3|12|10|212|6|5|5|NaN|NaN|NaN|NaN|NaN|
|top|Inbound|Returns|unknown|0|John Smith|\>90|Morning|NaN|NaN|NaN|NaN|NaN|
|freq|68142|44097|68711|85665|25261|30660|41426|NaN|NaN|NaN|NaN|NaN|
|mean|NaN|NaN|NaN|NaN|NaN|NaN|NaN|4\.242157216524847|0\.3190077642101342|0\.21222950399851|82\.73343266555693|41\.35869021150779|
|std|NaN|NaN|NaN|NaN|NaN|NaN|NaN|1\.3789030546991936|0\.4660947751428054|0\.408888845294696|47\.49474514198176|15\.86294747805643|
|min|NaN|NaN|NaN|NaN|NaN|NaN|NaN|1\.0|0\.0|0\.0|20\.0|5\.0|
|25%|NaN|NaN|NaN|NaN|NaN|NaN|NaN|4\.0|0\.0|0\.0|53\.0|29\.0|
|50%|NaN|NaN|NaN|NaN|NaN|NaN|NaN|5\.0|0\.0|0\.0|75\.0|41\.0|
|75%|NaN|NaN|NaN|NaN|NaN|NaN|NaN|5\.0|1\.0|0\.0|102\.0|55\.0|
|max|NaN|NaN|NaN|NaN|NaN|NaN|NaN|5\.0|1\.0|1\.0|429\.0|70\.0|

```
RangeIndex: 85907 entries, 0 to 85906
Data columns (total 12 columns):
 #   Column                   Non-Null Count  Dtype 
---  ------                   --------------  ----- 
 0   channel_name             85907 non-null  object
 1   category                 85907 non-null  object
 2   Product_category         85907 non-null  object
 3   connected_handling_time  85907 non-null  object
 4   Manager                  85907 non-null  object
 5   Tenure Bucket            85907 non-null  object
 6   Agent Shift              85907 non-null  object
 7   CSAT Score               85907 non-null  int64 
 8   Customer Remarks New     85907 non-null  int64 
 9   Is_order                 85907 non-null  int64 
 10  Agent_count              85907 non-null  int64 
 11  Sups_no_agents           85907 non-null  int64 
dtypes: int64(5), object(7)
memory usage: 7.9+ MB
```

```py
CS_dummies = pd.get_dummies(CS_dropped)
```

```
(85907, 258)
```


#### Задания для самостоятельного выполнения

1. Постройте на получившимся датасете Customer support модель дерева решений и проанализируйте важность признаков. Сделайте вывод об адекватности наших предположений.
1. Разбейте датасет на тестовую и обучающую выборки и преобразуйте обе подвыборки. Тестовую нужно преобразовывать точно также, как и обучающую (с теми же параметрами). 
1. Проведите полный анализ на датасете Титаник, включая все необходимые визуализации и выводы.
1. Проверьте целесообразность каждого необязательного преобразования данных путем проверки, увеличивает ли данное преобразование точность модели. Проверьте на простом виде модели (линейная регрессия, дерево решений или случайный лес). Поэкспериментируйте с различными вариантами преобразований.
1. Создайте воспроизводимый код обработки данного датасета. 

#### Контрольные вопросы

1. Какие основные типы графиков используются для визуализации эмпирического распределения категориальных атрибутов?
1. Какие графики целесообразно использовать для визуализации совместного распределения категориального атрибута и целевой переменной?
1. Какие способы заполнения отсутствующих значений работают с категориальными признаками?
1. Какие виды категориальных признаков существуют? Чем они определяются?
1. Как следует преобразовывать категориальные признаки в численные?

#### Дополнительные задания

1. Проведите полный анализ и преобразование категориальных атрибутов на датасете, который использовался в предыдущей работе. 
1. (*) Создайте код, реализующий алгоритм очистки данных автоматически (для любого датасета).

