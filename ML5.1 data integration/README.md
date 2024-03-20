### Интеграция данных

#### Цель работы

На практическом примере освоить основные этапы и приемы интеграции разных таблиц данных в единый датасет средствами Python и pandas для моследущей подготовки к машинному обучению.

#### Содержание работы

1. Прочитайте три однотипных файла с данными по обращениям клиентов в службу технической поддержки. Познакомьтесь со структурой этих трех таблиц и их особенностями. 
2. Объедините три эти таблицы в один общий датасет с максимальным соблюдением внутренней согласованности данных.
3. Прочитайте датасет с данными заказов клиентов, присоедините максимальное количество данных из него к датасету, получившемуся на предыдущем шаге.
4. Прочитайте датасет с данными о менеджерах службы клиентской поддержки и также присоедините данные из него к общему даасету.

#### Методические указания

В первую очередь прочитаем первый из трех однотипных табличных файлов, прилагающихся к данной работе:

```py
data1 = pd.read_csv('issues1.csv', index_col=0)
data1.head()
```

Мы видим таблицу, состоящую из  более чем 28 тысяч строк и 11 колонок:

```
|index|Id|Channel|category|Sub-category|Customer Remarks|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|---|---|
|0|e806cdca-d914-43bf-8c9c-ca86c2b38be0|Inbound|Order Related|Order status enquiry|NaN|NaN|03/08/2023 18:23|03/08/2023 20:35|03-Aug-23|C\. Lang|3\.0|
|1|a3c94fd3-b620-4b33-9422-25d3719ae3ff|Inbound|Returns|Return request|NaN|NaN|04/08/2023 21:39|04/08/2023 21:44|04-Aug-23|J\. Arnold|5\.0|
|2|42fc5059-f34b-49b5-b34a-a91a503f0a4e|Inbound|Returns|Reverse Pickup Enquiry|NaN|19dfb969-b591-4262-acab-df6654c0e395|01/08/2023 08:18|01/08/2023 08:22|01-Aug-23|F\. Newman|3\.0|
|3|480f7e83-7598-4774-8223-fe767d62021b|Outcall|Order Related|Invoice request|NaN|9d74deb4-ed83-4ed0-840b-e246aa5bfa78|02/08/2023 12:30|02/08/2023 12:32|02-Aug-23|J\. Rasmussen|5\.0|
|4|7f62e80e-42c1-4e42-90ac-6c7e6a9fef2e|Inbound|Product Queries|Product Specific Information|Gud ??|d62dbc71-9760-44d8-b865-32bb20edd6fc|01/08/2023 17:52|01/08/2023 18:01|01-Aug-23|J\. Foster|5\.0|
```


```py
data2 = pd.read_csv('issues2.csv', index_col=0)
data2
```

```
|index|Id|channel\_name|category|Sub-category|Order\_id|Reported at|Responded at|Survey responded at|Agent\_name|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|---|
|0|abb64ea1-4512-488c-81f5-51dbb130eec1|Inbound|Returns|Fraudulent User|c9ec6182-a825-4d0f-af28-51be601588e7|08 08 2023, 10:22:00 PM|08 08 2023, 11:42:00 PM|08 08 2023|J\. Mcgee|5|
|1|a1771e0f-9ce1-4ca7-b72d-2ba4e812a901|Inbound|Returns|Return request|55ddd823-2d08-4116-a863-1e6d77a27b23|17 08 2023, 08:15:00 PM|17 08 2023, 09:05:00 PM|17 08 2023|M\. Mckee|4|
|2|e556d65f-885c-42db-b8d2-8d726301910e|Inbound|Order Related|Order status enquiry|76b5dac1-946e-4ef4-aefa-ca9ef12abeed|08 08 2023, 07:54:00 PM|08 08 2023, 07:56:00 PM|08 08 2023|H\. Glover|5|
|3|f920ef14-be08-4c76-b446-7acacf6321d7|Outcall|Returns|Return request|591ca014-2903-4ac2-a075-8992ecc2abb8|17 08 2023, 02:39:00 PM|17 08 2023, 03:46:00 PM|17 08 2023|S\. Nelson|5|
|4|c6d3d6a3-3ff3-451d-898c-4bdea1c54932|Inbound|Returns|Return request|5967afff-aa5b-445b-8b9b-d625b93a886c|25 08 2023, 11:56:00 AM|25 08 2023, 12:10:00 PM|25 08 2023|N\. Nguyen|5|
```

```py
data3 = pd.read_csv('issues4.csv', index_col=0)
data3.head()
```

```
|index|Unique id|channel\_name|category|Sub-category|Issue\_reported at|issue\_responded|Survey\_response\_Date|Agent|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|
|0|9273f542-b0e1-411c-bebe-69d25a1cfb81|Inbound|Payments related|Online Payment Issues|August 29, 2023, 05:07:00 PM|August 29, 2023, 06:10:00 PM|August 29, 2023, 12:00:00 AM|John Green|5|
|1|247b304e-fcf5-4e44-bc72-15437056299a|Inbound|Order Related|Order status enquiry|August 16, 2023, 03:34:00 PM|August 16, 2023, 03:42:00 PM|August 16, 2023, 12:00:00 AM|Mark Lucas|5|
|2|d8ac7d94-b1ae-4316-81ef-4199d936f971|Outcall|Order Related|Installation/demo|August 14, 2023, 12:22:00 PM|August 14, 2023, 01:10:00 PM|August 14, 2023, 12:00:00 AM|William Morris|4|
|3|40e0781c-c81d-4646-9f1c-0a3cbe569596|Outcall|Returns|Return request|January 08, 2023, 08:24:00 PM|January 08, 2023, 08:31:00 PM|August 01, 2023, 12:00:00 AM|Joanna Wright|5|
|4|5bb8cfb5-9813-4710-adef-1042a0a16d0b|Inbound|Returns|Missing|August 16, 2023, 08:05:00 AM|August 16, 2023, 12:23:00 PM|August 16, 2023, 12:00:00 AM|Cynthia Sheppard|5|
```

```py
data1.shape, data2.shape, data3.shape
```

```
((28425, 11), (44339, 10), (13143, 9))
```

##### Вертикальная интеграция данных

```py
pd.concat([data1, data2]).head()
```

```
|index|Id|Channel|category|Sub-category|Customer Remarks|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name|CSAT Score|channel\_name|Reported at|Responded at|Survey responded at|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|03f3aaf6-85c5-4eb9-af30-7ab7cc7b627d|Inbound|Cancellation|Not Needed|Food|b0a0dece-5fda-4c0f-a69c-e6bcde3de8d4|2023-08-29 08:02:00|2023-08-29 08:04:00|2023-08-29 00:00:00|J\. Pruitt|5|NaN|NaN|NaN|NaN|
|1|f0442a16-b6d7-4fce-87fa-cbff2591176c|Outcall|Returns|Return request|V nicely |NaN|2023-03-08 18:38:00|2023-03-08 19:27:00|2023-08-03 00:00:00|T\. Bailey|5|NaN|NaN|NaN|NaN|
|2|99b7f6fa-a7b1-41a6-86ad-711ca8658fca|Inbound|Returns|Reverse Pickup Enquiry|Good|96e2ed8b-88b8-4768-a7e4-9468275c8085|2023-08-28 11:00:00|2023-08-28 00:00:00|2023-08-28 00:00:00|J\. Park|4|NaN|NaN|NaN|NaN|
|3|95dd50e0-bdbd-4e99-9278-e139a76a94d3|Inbound|Order Related|Delayed|Thank you for help |4340c8ed-d80f-4712-921a-4fea0d60dcc6|2023-05-08 23:42:00|2023-06-08 02:14:00|2023-08-06 00:00:00|R\. Elliott|5|NaN|NaN|NaN|NaN|
|4|2af35e04-d13b-43e9-9e69-358568f6174a|Inbound|Offers & Cashback|Affiliate Offers|Good |1549b443-c2c1-47b2-8608-cd4d5576d288|2023-01-08 11:13:00|2023-01-08 11:22:00|2023-08-01 00:00:00|M\. Brady|5|NaN|NaN|NaN|NaN|
```

```py
data1.info()
```

```
Int64Index: 28425 entries, 0 to 28424
Data columns (total 11 columns):
 #   Column                Non-Null Count  Dtype 
---  ------                --------------  ----- 
 0   Id                    28425 non-null  object
 1   Channel               28425 non-null  object
 2   category              28425 non-null  object
 3   Sub-category          28425 non-null  object
 4   Customer Remarks      28425 non-null  object
 5   Order_id              22387 non-null  object
 6   Issue_reported_Date   28425 non-null  object
 7   Issue_responded_Date  28425 non-null  object
 8   Survey_response_Date  28425 non-null  object
 9   Agent_name            28425 non-null  object
 10  CSAT Score            28425 non-null  int64 
dtypes: int64(1), object(10)
memory usage: 3.6+ MB
```

```py
data2.info()
```

```
Int64Index: 44339 entries, 0 to 44338
Data columns (total 10 columns):
 #   Column               Non-Null Count  Dtype 
---  ------               --------------  ----- 
 0   Id                   44339 non-null  object
 1   channel_name         44339 non-null  object
 2   category             44339 non-null  object
 3   Sub-category         44339 non-null  object
 4   Order_id             44271 non-null  object
 5   Reported at          44339 non-null  object
 6   Responded at         44339 non-null  object
 7   Survey responded at  44339 non-null  object
 8   Agent_name           44339 non-null  object
 9   CSAT Score           44339 non-null  int64 
dtypes: int64(1), object(9)
memory usage: 4.7+ MB
```

```py
data2 = data2.rename(columns={
    'channel_name': 'Channel',
    'Reported at': 'Issue_reported_Date',
    'Responded at': 'Issue_responded_Date',
    'Survey responded at': 'Survey_response_Date',
})
```

```py
data2['Issue_reported_Date'] = pd.to_datetime(data2['Issue_reported_Date'])
data2['Issue_responded_Date'] = pd.to_datetime(data2['Issue_responded_Date'])
data2['Survey_response_Date'] = pd.to_datetime(data2['Survey_response_Date'], format='%d %m %Y')
```

```py
data2.info()
```

```
Int64Index: 44339 entries, 0 to 44338
Data columns (total 10 columns):
 #   Column                Non-Null Count  Dtype         
---  ------                --------------  -----         
 0   Id                    44339 non-null  object        
 1   Channel               44339 non-null  object        
 2   category              44339 non-null  object        
 3   Sub-category          44339 non-null  object        
 4   Order_id              44271 non-null  object        
 5   Issue_reported_Date   44339 non-null  datetime64[ns]
 6   Issue_responded_Date  44339 non-null  datetime64[ns]
 7   Survey_response_Date  44339 non-null  datetime64[ns]
 8   Agent_name            44339 non-null  object        
 9   CSAT Score            44339 non-null  int64         
dtypes: datetime64[ns](3), int64(1), object(6)
memory usage: 4.7+ MB
```

```py
data2.head()
```

```
|index|Id|Channel|category|Sub-category|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|---|
|0|abb64ea1-4512-488c-81f5-51dbb130eec1|Inbound|Returns|Fraudulent User|c9ec6182-a825-4d0f-af28-51be601588e7|2023-08-08 22:22:00|2023-08-08 23:42:00|2023-08-08 00:00:00|J\. Mcgee|5|
|1|a1771e0f-9ce1-4ca7-b72d-2ba4e812a901|Inbound|Returns|Return request|55ddd823-2d08-4116-a863-1e6d77a27b23|2023-08-17 20:15:00|2023-08-17 21:05:00|2023-08-17 00:00:00|M\. Mckee|4|
|2|e556d65f-885c-42db-b8d2-8d726301910e|Inbound|Order Related|Order status enquiry|76b5dac1-946e-4ef4-aefa-ca9ef12abeed|2023-08-08 19:54:00|2023-08-08 19:56:00|2023-08-08 00:00:00|H\. Glover|5|
|3|f920ef14-be08-4c76-b446-7acacf6321d7|Outcall|Returns|Return request|591ca014-2903-4ac2-a075-8992ecc2abb8|2023-08-17 14:39:00|2023-08-17 15:46:00|2023-08-17 00:00:00|S\. Nelson|5|
|4|c6d3d6a3-3ff3-451d-898c-4bdea1c54932|Inbound|Returns|Return request|5967afff-aa5b-445b-8b9b-d625b93a886c|2023-08-25 11:56:00|2023-08-25 12:10:00|2023-08-25 00:00:00|N\. Nguyen|5|
```


```py
data1['Issue_reported_Date'] = pd.to_datetime(data1['Issue_reported_Date'])
data1['Issue_responded_Date'] = pd.to_datetime(data1['Issue_responded_Date'])
data1['Survey_response_Date'] = pd.to_datetime(data1['Survey_response_Date'], format='%d-%b-%y')
```

```py
data1.info()
```

```
Int64Index: 28425 entries, 0 to 28424
Data columns (total 11 columns):
 #   Column                Non-Null Count  Dtype         
---  ------                --------------  -----         
 0   Id                    28425 non-null  object        
 1   Channel               28425 non-null  object        
 2   category              28425 non-null  object        
 3   Sub-category          28425 non-null  object        
 4   Customer Remarks      28425 non-null  object        
 5   Order_id              22387 non-null  object        
 6   Issue_reported_Date   28425 non-null  datetime64[ns]
 7   Issue_responded_Date  28425 non-null  datetime64[ns]
 8   Survey_response_Date  28425 non-null  datetime64[ns]
 9   Agent_name            28425 non-null  object        
 10  CSAT Score            28425 non-null  int64         
dtypes: datetime64[ns](3), int64(1), object(7)
memory usage: 3.6+ MB
```

```py
data_12 = pd.concat([data1, data2])
data_12.head()
```

```
|index|Id|Channel|category|Sub-category|Customer Remarks|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|---|---|
|0|03f3aaf6-85c5-4eb9-af30-7ab7cc7b627d|Inbound|Cancellation|Not Needed|Food|b0a0dece-5fda-4c0f-a69c-e6bcde3de8d4|2023-08-29 08:02:00|2023-08-29 08:04:00|2023-08-29 00:00:00|J\. Pruitt|5|
|1|f0442a16-b6d7-4fce-87fa-cbff2591176c|Outcall|Returns|Return request|V nicely |NaN|2023-03-08 18:38:00|2023-03-08 19:27:00|2023-08-03 00:00:00|T\. Bailey|5|
|2|99b7f6fa-a7b1-41a6-86ad-711ca8658fca|Inbound|Returns|Reverse Pickup Enquiry|Good|96e2ed8b-88b8-4768-a7e4-9468275c8085|2023-08-28 11:00:00|2023-08-28 00:00:00|2023-08-28 00:00:00|J\. Park|4|
|3|95dd50e0-bdbd-4e99-9278-e139a76a94d3|Inbound|Order Related|Delayed|Thank you for help |4340c8ed-d80f-4712-921a-4fea0d60dcc6|2023-05-08 23:42:00|2023-06-08 02:14:00|2023-08-06 00:00:00|R\. Elliott|5|
|4|2af35e04-d13b-43e9-9e69-358568f6174a|Inbound|Offers & Cashback|Affiliate Offers|Good |1549b443-c2c1-47b2-8608-cd4d5576d288|2023-01-08 11:13:00|2023-01-08 11:22:00|2023-08-01 00:00:00|M\. Brady|5|
```

```py
data_12.info()
```

```
Int64Index: 72764 entries, 0 to 44338
Data columns (total 11 columns):
 #   Column                Non-Null Count  Dtype         
---  ------                --------------  -----         
 0   Id                    72764 non-null  object        
 1   Channel               72764 non-null  object        
 2   category              72764 non-null  object        
 3   Sub-category          72764 non-null  object        
 4   Customer Remarks      28425 non-null  object        
 5   Order_id              66658 non-null  object        
 6   Issue_reported_Date   72764 non-null  datetime64[ns]
 7   Issue_responded_Date  72764 non-null  datetime64[ns]
 8   Survey_response_Date  72764 non-null  datetime64[ns]
 9   Agent_name            72764 non-null  object        
 10  CSAT Score            72764 non-null  int64         
dtypes: datetime64[ns](3), int64(1), object(7)
memory usage: 6.7+ MB
```

```py
data3.info()
```

```
Int64Index: 13143 entries, 0 to 13142
Data columns (total 9 columns):
 #   Column                Non-Null Count  Dtype 
---  ------                --------------  ----- 
 0   Unique id             13143 non-null  object
 1   channel_name          13143 non-null  object
 2   category              13143 non-null  object
 3   Sub-category          13143 non-null  object
 4   Issue_reported at     13143 non-null  object
 5   issue_responded       13143 non-null  object
 6   Survey_response_Date  13143 non-null  object
 7   Agent                 13143 non-null  object
 8   CSAT Score            13143 non-null  int64 
dtypes: int64(1), object(8)
memory usage: 1.5+ MB
```

```py
data3 = data3.rename(columns={
    'Unique id': 'Id',
    'channel_name': 'Channel',
    'Issue_reported at': 'Issue_reported_Date',
    'issue_responded': 'Issue_responded_Date',
    'Survey responded at': 'Survey_response_Date',
    'Agent': 'Agent_name'
})
```

```py
data3['Issue_reported_Date'] = pd.to_datetime(data3['Issue_reported_Date'])
data3['Issue_responded_Date'] = pd.to_datetime(data3['Issue_responded_Date'])
data3['Survey_response_Date'] = pd.to_datetime(data3['Survey_response_Date'])
```

```py
data3.head()
```

```
|index|Id|Channel|category|Sub-category|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|
|0|9273f542-b0e1-411c-bebe-69d25a1cfb81|Inbound|Payments related|Online Payment Issues|2023-08-29 17:07:00|2023-08-29 18:10:00|2023-08-29 00:00:00|J\. Green|5|
|1|247b304e-fcf5-4e44-bc72-15437056299a|Inbound|Order Related|Order status enquiry|2023-08-16 15:34:00|2023-08-16 15:42:00|2023-08-16 00:00:00|M\. Lucas|5|
|2|d8ac7d94-b1ae-4316-81ef-4199d936f971|Outcall|Order Related|Installation/demo|2023-08-14 12:22:00|2023-08-14 13:10:00|2023-08-14 00:00:00|W\. Morris|4|
|3|40e0781c-c81d-4646-9f1c-0a3cbe569596|Outcall|Returns|Return request|2023-01-08 20:24:00|2023-01-08 20:31:00|2023-08-01 00:00:00|J\. Wright|5|
|4|5bb8cfb5-9813-4710-adef-1042a0a16d0b|Inbound|Returns|Missing|2023-08-16 08:05:00|2023-08-16 12:23:00|2023-08-16 00:00:00|C\. Sheppard|5|
```

```py
data3.info()
```

```
Int64Index: 13143 entries, 0 to 13142
Data columns (total 9 columns):
 #   Column                Non-Null Count  Dtype         
---  ------                --------------  -----         
 0   Id                    13143 non-null  object        
 1   Channel               13143 non-null  object        
 2   category              13143 non-null  object        
 3   Sub-category          13143 non-null  object        
 4   Issue_reported_Date   13143 non-null  datetime64[ns]
 5   Issue_responded_Date  13143 non-null  datetime64[ns]
 6   Survey_response_Date  13143 non-null  datetime64[ns]
 7   Agent_name            13143 non-null  object        
 8   CSAT Score            13143 non-null  int64         
dtypes: datetime64[ns](3), int64(1), object(5)
memory usage: 1.5+ MB
```

```py
data_123.info()
```

```py
Int64Index: 85907 entries, 0 to 13142
Data columns (total 11 columns):
 #   Column                Non-Null Count  Dtype         
---  ------                --------------  -----         
 0   Id                    85907 non-null  object        
 1   Channel               85907 non-null  object        
 2   category              85907 non-null  object        
 3   Sub-category          85907 non-null  object        
 4   Customer Remarks      28425 non-null  object        
 5   Order_id              66658 non-null  object        
 6   Issue_reported_Date   85907 non-null  datetime64[ns]
 7   Issue_responded_Date  85907 non-null  datetime64[ns]
 8   Survey_response_Date  85907 non-null  datetime64[ns]
 9   Agent_name            85907 non-null  object        
 10  CSAT Score            85907 non-null  int64         
dtypes: datetime64[ns](3), int64(1), object(7)
memory usage: 7.9+ MB
```

```py
data_123 = pd.concat([data_12, data3])
data_123.head()
```

```
|index|Id|Channel|category|Sub-category|Customer Remarks|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|---|---|
|0|03f3aaf6-85c5-4eb9-af30-7ab7cc7b627d|Inbound|Cancellation|Not Needed|Food|b0a0dece-5fda-4c0f-a69c-e6bcde3de8d4|2023-08-29 08:02:00|2023-08-29 08:04:00|2023-08-29 00:00:00|J\. Pruitt|5|
|1|f0442a16-b6d7-4fce-87fa-cbff2591176c|Outcall|Returns|Return request|V nicely |NaN|2023-03-08 18:38:00|2023-03-08 19:27:00|2023-08-03 00:00:00|T\. Bailey|5|
|2|99b7f6fa-a7b1-41a6-86ad-711ca8658fca|Inbound|Returns|Reverse Pickup Enquiry|Good|96e2ed8b-88b8-4768-a7e4-9468275c8085|2023-08-28 11:00:00|2023-08-28 00:00:00|2023-08-28 00:00:00|J\. Park|4|
|3|95dd50e0-bdbd-4e99-9278-e139a76a94d3|Inbound|Order Related|Delayed|Thank you for help |4340c8ed-d80f-4712-921a-4fea0d60dcc6|2023-05-08 23:42:00|2023-06-08 02:14:00|2023-08-06 00:00:00|R\. Elliott|5|
|4|2af35e04-d13b-43e9-9e69-358568f6174a|Inbound|Offers & Cashback|Affiliate Offers|Good |1549b443-c2c1-47b2-8608-cd4d5576d288|2023-01-08 11:13:00|2023-01-08 11:22:00|2023-08-01 00:00:00|M\. Brady|5|
```

```py
data3['Agent_name'] = data3['Agent_name'].apply(
    lambda x: x.split()[0][0] + ". " + x.split()[1] if isinstance(x, str) else np.NAN
)
```

```py
data_123 = pd.concat([data_12, data3])
```

##### Добавление данных заказа

```py
orders_data = pd.read_csv('orders.csv', index_col=0)
orders_data.head()
```

```
|index|Id|order\_date\_time|Customer\_City|Product\_category|Item\_price|
|---|---|---|---|---|---|
|0|5779ad15-5147-438c-a7eb-94b44a79b610|14/07/2023 16:07|AMBEGAON|LifeStyle|145\.0|
|1|5db8efa3-4cdc-4eff-9b97-7ff8ca20dcbd|09/08/2023 21:15|BULANDSHAHR|Electronics|1599\.0|
|2|5f678412-588d-4a8a-96a8-ef5998c5f34f|05/07/2023 12:53|SAHARSA|Electronics|41990\.0|
|3|786444c1-d3a7-473e-94a3-461fa635339b|15/07/2023 13:49|BULDHANA|Electronics|2699\.0|
|4|58e5dae7-2b28-4f1e-9347-384a172b3469|06/08/2023 15:52|RAJKOT|LifeStyle|183\.0|
```

```py
orders_data = orders_data.rename(columns={
    'Id': 'Order_id',
})
```

```py
orders_data.info()
```

```
Int64Index: 86070 entries, 0 to 86069
Data columns (total 5 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Order_id          86070 non-null  object 
 1   order_date_time   86070 non-null  object 
 2   Customer_City     85359 non-null  object 
 3   Product_category  85963 non-null  object 
 4   Item_price        86017 non-null  float64
dtypes: float64(1), object(4)
memory usage: 6.0+ MB
```

```py
data_with_orders = data_123.merge(orders_data, on='Order_id', how='left')
data_with_orders.head()
```

```
|index|Id|Channel|category|Sub-category|Customer Remarks|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name|CSAT Score|order\_date\_time|Customer\_City|Product\_category|Item\_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|03f3aaf6-85c5-4eb9-af30-7ab7cc7b627d|Inbound|Cancellation|Not Needed|Food|b0a0dece-5fda-4c0f-a69c-e6bcde3de8d4|2023-08-29 08:02:00|2023-08-29 08:04:00|2023-08-29 00:00:00|J\. Pruitt|5|NaN|NaN|NaN|NaN|
|1|f0442a16-b6d7-4fce-87fa-cbff2591176c|Outcall|Returns|Return request|V nicely |NaN|2023-03-08 18:38:00|2023-03-08 19:27:00|2023-08-03 00:00:00|T\. Bailey|5|NaN|NaN|NaN|NaN|
|2|99b7f6fa-a7b1-41a6-86ad-711ca8658fca|Inbound|Returns|Reverse Pickup Enquiry|Good|96e2ed8b-88b8-4768-a7e4-9468275c8085|2023-08-28 11:00:00|2023-08-28 00:00:00|2023-08-28 00:00:00|J\. Park|4|NaN|NaN|NaN|NaN|
|3|95dd50e0-bdbd-4e99-9278-e139a76a94d3|Inbound|Order Related|Delayed|Thank you for help |4340c8ed-d80f-4712-921a-4fea0d60dcc6|2023-05-08 23:42:00|2023-06-08 02:14:00|2023-08-06 00:00:00|R\. Elliott|5|31/07/2023 12:11|JALGAON|Home|599\.0|
|4|2af35e04-d13b-43e9-9e69-358568f6174a|Inbound|Offers & Cashback|Affiliate Offers|Good |1549b443-c2c1-47b2-8608-cd4d5576d288|2023-01-08 11:13:00|2023-01-08 11:22:00|2023-08-01 00:00:00|M\. Brady|5|NaN|NaN|NaN|NaN|
```

```py
orders_data[orders_data.Order_id == 'a5464619-b2d8-49de-9346-5e5db0972756']
```

```
|index|Order\_id|order\_date\_time|Customer\_City|Product\_category|Item\_price|
|---|---|---|---|---|---|

```

```py
sns.heatmap(data_with_orders.isnull(), yticklabels=False, cbar=False)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.1%20data%20integration/ml51-1.png?raw=true)

##### Добавление данных менеджеров

```py
agent_data = pd.read_csv("agents.csv", index_col=0)
agent_data.head()
```

```
|index|Agent|Supervisor|Manager|Tenure Bucket|Shift|
|---|---|---|---|---|---|
|0|Aaron Edwards|Mia Patel|Emily Chen|61-90|Evening|
|1|Aaron Romero|Mason Gupta|Jennifer Nguyen|On Job Training|Morning|
|2|Abigail Gonzalez|Jacob Sato|Jennifer Nguyen|On Job Training|Morning|
|3|Adam Barnett|Abigail Suzuki|Jennifer Nguyen|On Job Training|Morning|
|4|Adam Hammond|Olivia Suzuki|John Smith|31-60|Morning|
```

```py
agent_data.info()
```

```py
Int64Index: 1371 entries, 0 to 1370
Data columns (total 5 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   Agent          1371 non-null   object
 1   Supervisor     1371 non-null   object
 2   Manager        1371 non-null   object
 3   Tenure Bucket  1371 non-null   object
 4   Shift          1371 non-null   object
dtypes: object(5)
memory usage: 64.3+ KB
```

```py
agent_data.groupby(['Agent']).agg('count').head(20)
```

```
|Agent|Supervisor|Manager|Tenure Bucket|Shift|
|---|---|---|---|---|
|Aaron Edwards|1|1|1|1|
|Aaron Romero|1|1|1|1|
|Abigail Gonzalez|1|1|1|1|
|Adam Barnett|1|1|1|1|
|Adam Hammond|1|1|1|1|
|Adam Henderson|1|1|1|1|
|Adam Hernandez|1|1|1|1|
|Adam Schwartz|1|1|1|1|
|Adam Torres|1|1|1|1|
|Adrian Branch|1|1|1|1|
|Adrian Johnson|1|1|1|1|
|Adrian Richards|1|1|1|1|
|Adriana Adams|1|1|1|1|
|Alan Cruz|1|1|1|1|
|Alan Davies|1|1|1|1|
|Alan Sweeney|1|1|1|1|
|Albert Thornton|1|1|1|1|
|Alexa Mcpherson|1|1|1|1|
|Alexander Marshall|1|1|1|1|
|Alexander Mata|1|1|1|1|
```

```py
agent_data['Agent_name'] = agent_data['Agent'].apply(
    lambda x: x.split()[0][0] + ". " + x.split()[1] if isinstance(x, str) else np.NAN
)
```

```py
agent_data.groupby(['Agent_name']).agg('first').head(20)
```

```
|Agent\_name|Agent|Supervisor|Manager|Tenure Bucket|Shift|
|---|---|---|---|---|---|
|A\. Adams|Adriana Adams|Sophia Sato|John Smith|0-30|Morning|
|A\. Aguilar|Andrew Aguilar|Nathan Patel|Emily Chen|On Job Training|Evening|
|A\. Barnett|Adam Barnett|Abigail Suzuki|Jennifer Nguyen|On Job Training|Morning|
|A\. Barrett|Andrew Barrett|Elijah Yamaguchi|John Smith|\>90|Evening|
|A\. Barron|Alexandra Barron|Austin Johnson|Jennifer Nguyen|On Job Training|Evening|
|A\. Beck|Andrea Beck|Evelyn Kimura|Jennifer Nguyen|On Job Training|Evening|
|A\. Bell|Amy Bell|Wyatt Kim|Jennifer Nguyen|On Job Training|Evening|
|A\. Benjamin|Amanda Benjamin|Amelia Tanaka|Emily Chen|\>90|Morning|
|A\. Berry|Amy Berry|Elijah Yamaguchi|John Smith|\>90|Evening|
|A\. Booker|Angela Booker MD|Carter Park|Jennifer Nguyen|On Job Training|Evening|
|A\. Booth|Anthony Booth|William Park|John Smith|31-60|Morning|
|A\. Bowen|Anthony Bowen|Olivia Suzuki|Jennifer Nguyen|On Job Training|Morning|
|A\. Bowers|Allen Bowers|Olivia Wang|John Smith|\>90|Evening|
|A\. Branch|Adrian Branch|Carter Park|Jennifer Nguyen|On Job Training|Morning|
|A\. Brown|Amanda Brown|Emily Yamashita|John Smith|\>90|Morning|
|A\. Burke|Amy Burke|Sophia Chen|William Kim|On Job Training|Morning|
|A\. Burton|Alicia Burton|Mia Patel|Emily Chen|0-30|Evening|
|A\. Calderon|Amy Calderon|Elijah Yamaguchi|Michael Lee|31-60|Afternoon|
|A\. Campbell|Alexis Campbell|Elijah Yamaguchi|Michael Lee|\>90|Afternoon|
|A\. Campos|Alisha Campos|Jacob Sato|Emily Chen|0-30|Evening|
```

```py
data_123.groupby(['Agent_name']).agg('count').head(20)
```

```
|Agent\_name|Id|Channel|category|Sub-category|Customer Remarks|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|CSAT Score|
|---|---|---|---|---|---|---|---|---|---|---|
|A\. Adams|215|215|215|215|70|175|215|215|215|215|
|A\. Aguilar|31|31|31|31|11|19|31|31|31|31|
|A\. Barnett|56|56|56|56|21|41|56|56|56|56|
|A\. Barrett|50|50|50|50|16|35|50|50|50|50|
|A\. Barron|25|25|25|25|9|18|25|25|25|25|
|A\. Beck|70|70|70|70|17|51|70|70|70|70|
|A\. Bell|20|20|20|20|10|18|20|20|20|20|
|A\. Benjamin|64|64|64|64|22|48|64|64|64|64|
|A\. Berry|135|135|135|135|62|89|135|135|135|135|
|A\. Booker|21|21|21|21|3|19|21|21|21|21|
|A\. Booth|177|177|177|177|60|133|177|177|177|177|
|A\. Bowen|26|26|26|26|5|24|26|26|26|26|
|A\. Bowers|126|126|126|126|35|101|126|126|126|126|
|A\. Branch|43|43|43|43|19|40|43|43|43|43|
|A\. Brown|197|197|197|197|55|155|197|197|197|197|
|A\. Burke|80|80|80|80|28|69|80|80|80|80|
|A\. Burton|29|29|29|29|9|24|29|29|29|29|
|A\. Calderon|36|36|36|36|13|29|36|36|36|36|
|A\. Campbell|117|117|117|117|39|91|117|117|117|117|
|A\. Campos|95|95|95|95|31|75|95|95|95|95|
```

```py
data_with_agents = data_with_orders.join(agent_data, rsuffix='_1', how='left')
data_with_agents.head()
```

```
|index|Id|Channel|category|Sub-category|Customer Remarks|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name|CSAT Score|order\_date\_time|Customer\_City|Product\_category|Item\_price|Agent|Supervisor|Manager|Tenure Bucket|Shift|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|03f3aaf6-85c5-4eb9-af30-7ab7cc7b627d|Inbound|Cancellation|Not Needed|Food|b0a0dece-5fda-4c0f-a69c-e6bcde3de8d4|2023-08-29 08:02:00|2023-08-29 08:04:00|2023-08-29 00:00:00|J\. Pruitt|5|NaN|NaN|NaN|NaN|Aaron Edwards|Mia Patel|Emily Chen|61-90|Evening|
|1|f0442a16-b6d7-4fce-87fa-cbff2591176c|Outcall|Returns|Return request|V nicely |NaN|2023-03-08 18:38:00|2023-03-08 19:27:00|2023-08-03 00:00:00|T\. Bailey|5|NaN|NaN|NaN|NaN|Aaron Romero|Mason Gupta|Jennifer Nguyen|On Job Training|Morning|
|2|99b7f6fa-a7b1-41a6-86ad-711ca8658fca|Inbound|Returns|Reverse Pickup Enquiry|Good|96e2ed8b-88b8-4768-a7e4-9468275c8085|2023-08-28 11:00:00|2023-08-28 00:00:00|2023-08-28 00:00:00|J\. Park|4|NaN|NaN|NaN|NaN|Abigail Gonzalez|Jacob Sato|Jennifer Nguyen|On Job Training|Morning|
|3|95dd50e0-bdbd-4e99-9278-e139a76a94d3|Inbound|Order Related|Delayed|Thank you for help |4340c8ed-d80f-4712-921a-4fea0d60dcc6|2023-05-08 23:42:00|2023-06-08 02:14:00|2023-08-06 00:00:00|R\. Elliott|5|31/07/2023 12:11|JALGAON|Home|599\.0|Adam Barnett|Abigail Suzuki|Jennifer Nguyen|On Job Training|Morning|
|4|2af35e04-d13b-43e9-9e69-358568f6174a|Inbound|Offers & Cashback|Affiliate Offers|Good |1549b443-c2c1-47b2-8608-cd4d5576d288|2023-01-08 11:13:00|2023-01-08 11:22:00|2023-08-01 00:00:00|M\. Brady|5|NaN|NaN|NaN|NaN|Adam Hammond|Olivia Suzuki|John Smith|31-60|Morning|
```

```py
agent_data[agent_data.Agent_name == 'C. Norman']
```

```
|index|Agent|Supervisor|Manager|Tenure Bucket|Shift|Agent\_name|
|---|---|---|---|---|---|---|
|191|Carolyn Norman|Abigail Suzuki|Jennifer Nguyen|On Job Training|Evening|C\. Norman|
```

```py
sns.heatmap(data_with_agents.isnull(), yticklabels=False, cbar=False)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.1%20data%20integration/ml51-2.png?raw=true)

```py
data_with_agents = data_with_orders.merge(agent_data, how='left', on='Agent_name', copy=False)
data_with_agents.head()
```

```
|index|Id|Channel|category|Sub-category|Customer Remarks|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name|CSAT Score|order\_date\_time|Customer\_City|Product\_category|Item\_price|Agent|Supervisor|Manager|Tenure Bucket|Shift|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|18f329f8-90af-4277-af78-acc1f41e8af2|Inbound|Returns|Reverse Pickup Enquiry|I AM satisfied for inconvenience |4cec934f-48c1-41da-9a54-7bea0248864d|2023-07-08 11:48:00|2023-07-08 11:57:00|2023-08-07 00:00:00|B\. Suarez|5|NaN|NaN|NaN|NaN|Brenda Suarez|Ethan Nakamura|Jennifer Nguyen|On Job Training|Morning|
|1|21972b69-2e02-4d39-8643-d0c6153e6c04|Inbound|Returns|Missing|Good |NaN|2023-08-16 09:16:00|2023-08-16 09:18:00|2023-08-16 00:00:00|M\. Robinson|4|NaN|NaN|NaN|NaN|Melissa Robinson|Elijah Yamaguchi|Emily Chen|31-60|Evening|
|2|97626f96-bd35-42ef-bf9e-07dfdde0400c|Inbound|Returns|Return request|Thank you sir for solving my return problem|c00bff95-5657-4acc-a8f2-2e242277b7d2|2023-09-08 19:19:00|2023-09-08 20:58:00|2023-08-09 00:00:00|S\. Warner|5|26/07/2023 23:10|AURANGABAD|Electronics|1409\.0|Samantha Warner|Emily Yamashita|John Smith|\>90|Morning|
|3|515adae8-3483-4e6a-b857-968dafdd04c5|Inbound|Order Related|Order status enquiry|Impossible to reach customer support\.|21f4a92f-4a8e-4113-b9da-3af750f04f7c|2023-08-30 13:51:00|2023-08-30 15:41:00|2023-08-30 00:00:00|M\. Whitehead|1|NaN|NaN|NaN|NaN|Michelle Whitehead|Carter Park|Jennifer Nguyen|On Job Training|Evening|
|4|f5a03787-5528-4bec-9fe1-9ed44bfcc2ec|Inbound|Feedback|UnProfessional Behaviour|  from Shopzilla is amazing \. THANKS A TON TO HIM FOR HELPING ME A PERSON WITH POLITE NATURE & BETTER ETIQUETTES LOTS OF HUGS & APPRECIATION TO HIM |9c8aa1c3-eb4d-4c65-8434-0af9fa8c75ff|2023-08-23 17:52:00|2023-08-23 18:16:00|2023-08-23 00:00:00|B\. Key|5|NaN|NaN|NaN|NaN|Brittney Key|Mia Yamamoto|Jennifer Nguyen|On Job Training|Morning|
```

```py
sns.heatmap(data_with_agents.isnull(), yticklabels=False, cbar=False)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.1%20data%20integration/ml51-3.png?raw=true)

```py
agent_data.Agent_name.value_counts()
```

```
J. Moore        6
A. Brown        4
M. Smith        4
D. Smith        4
K. Martin       4
               ..
J. Meadows      1
J. Edwards      1
J. Schroeder    1
J. Robertson    1
Z. Simpson      1
Name: Agent_name, Length: 1221, dtype: int64
```

```py
agent_data[agent_data.Agent_name == 'J. Moore']
```

```
|index|Agent|Supervisor|Manager|Tenure Bucket|Shift|Agent\_name|
|---|---|---|---|---|---|---|
|468|Jacob Moore|Ava Wong|Jennifer Nguyen|On Job Training|Morning|J\. Moore|
|529|Jenna Moore|Aiden Patel|John Smith|\>90|Morning|J\. Moore|
|542|Jennifer Moore|Nathan Patel|Jennifer Nguyen|On Job Training|Evening|J\. Moore|
|612|Jon Moore|Ava Wong|William Kim|On Job Training|Evening|J\. Moore|
|617|Jonathan Moore|Zoe Yamamoto|John Smith|\>90|Morning|J\. Moore|
|654|Juan Moore|Mason Gupta|Jennifer Nguyen|On Job Training|Morning|J\. Moore|
```

```py
data_with_agent_numbers = data_with_orders.merge(agent_data.Agent_name.value_counts(), how='left', left_on='Agent_name', right_index=True)
data_with_agent_numbers.info()
```

```
Int64Index: 85907 entries, 0 to 85906
Data columns (total 16 columns):
 #   Column                Non-Null Count  Dtype         
---  ------                --------------  -----         
 0   Id                    85907 non-null  object        
 1   Channel               85907 non-null  object        
 2   category              85907 non-null  object        
 3   Sub-category          85907 non-null  object        
 4   Customer Remarks      28425 non-null  object        
 5   Order_id              66658 non-null  object        
 6   Issue_reported_Date   85907 non-null  datetime64[ns]
 7   Issue_responded_Date  85907 non-null  datetime64[ns]
 8   Survey_response_Date  85907 non-null  datetime64[ns]
 9   Agent_name_x          85907 non-null  object        
 10  CSAT Score            85907 non-null  int64         
 11  order_date_time       16950 non-null  object        
 12  Customer_City         16816 non-null  object        
 13  Product_category      16932 non-null  object        
 14  Item_price            16942 non-null  float64       
 15  Agent_name_y          85907 non-null  int64         
dtypes: datetime64[ns](3), float64(1), int64(2), object(10)
memory usage: 13.2+ MB
```

```py
data_with_agent_numbers.Agent_name_y.value_counts()
```

```py
1    68749
2    11326
3     3695
4     1843
6      294
Name: Agent_name_y, dtype: int64
```

```py
data_with_agent_numbers['Agent_name_x'] = data_with_agent_numbers.Agent_name_x.where(
    data_with_agent_numbers['Agent_name_y'] == 1.0, np.NAN
)
data_with_agent_numbers.head(20)
```

```
|index|Id|Channel|category|Sub-category|Customer Remarks|Order\_id|Issue\_reported\_Date|Issue\_responded\_Date|Survey\_response\_Date|Agent\_name\_x|CSAT Score|order\_date\_time|Customer\_City|Product\_category|Item\_price|Agent\_name\_y|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|18f329f8-90af-4277-af78-acc1f41e8af2|Inbound|Returns|Reverse Pickup Enquiry|I AM satisfied for inconvenience |4cec934f-48c1-41da-9a54-7bea0248864d|2023-07-08 11:48:00|2023-07-08 11:57:00|2023-08-07 00:00:00|B\. Suarez|5|NaN|NaN|NaN|NaN|1|
|1|21972b69-2e02-4d39-8643-d0c6153e6c04|Inbound|Returns|Missing|Good |NaN|2023-08-16 09:16:00|2023-08-16 09:18:00|2023-08-16 00:00:00|M\. Robinson|4|NaN|NaN|NaN|NaN|1|
|2|97626f96-bd35-42ef-bf9e-07dfdde0400c|Inbound|Returns|Return request|Thank you sir for solving my return problem|c00bff95-5657-4acc-a8f2-2e242277b7d2|2023-09-08 19:19:00|2023-09-08 20:58:00|2023-08-09 00:00:00|S\. Warner|5|26/07/2023 23:10|AURANGABAD|Electronics|1409\.0|1|
|3|515adae8-3483-4e6a-b857-968dafdd04c5|Inbound|Order Related|Order status enquiry|Impossible to reach customer support\.|21f4a92f-4a8e-4113-b9da-3af750f04f7c|2023-08-30 13:51:00|2023-08-30 15:41:00|2023-08-30 00:00:00|M\. Whitehead|1|NaN|NaN|NaN|NaN|1|
|4|f5a03787-5528-4bec-9fe1-9ed44bfcc2ec|Inbound|Feedback|UnProfessional Behaviour|  from Shopzilla is amazing \. THANKS A TON TO HIM FOR HELPING ME A PERSON WITH POLITE NATURE & BETTER ETIQUETTES LOTS OF HUGS & APPRECIATION TO HIM |9c8aa1c3-eb4d-4c65-8434-0af9fa8c75ff|2023-08-23 17:52:00|2023-08-23 18:16:00|2023-08-23 00:00:00|B\. Key|5|NaN|NaN|NaN|NaN|1|
|5|d9ed182b-d7e1-4e80-bbbd-b227805e6aa6|Inbound|Returns|Reverse Pickup Enquiry|Check product after delivery\. |48c7535a-255b-4e9e-82d9-35f0cac408c1|2023-08-08 10:43:00|2023-08-08 10:45:00|2023-08-08 00:00:00|L\. Colon|5|NaN|NaN|NaN|NaN|1|
|6|3413c9fe-6e8a-4e57-b870-43995bfeaa56|Inbound|Refund Related|Refund Related Issues|No response from team|74a95ec1-0d44-4c76-90d5-17f0e442c097|2023-08-24 10:19:00|2023-08-24 10:19:00|2023-08-24 00:00:00|J\. Pruitt|1|NaN|NaN|NaN|NaN|1|
|7|d5bf2d68-4f8a-4684-873d-3766c92b0b7f|Inbound|Returns|Reverse Pickup Enquiry|I talked to someone yesterday also regarding my issue but she said it will be resolved & I will get the update but I didn't receive anything\.Today also I talked to another one she was quite good to listen & understand my issue & promised to help\. I am waiting for this resolution\. \. |283d3120-17e8-423e-885d-99c8d3a3ee78|2023-06-08 07:53:00|2023-06-08 08:04:00|2023-08-06 00:00:00|G\. Jones|5|NaN|NaN|NaN|NaN|1|
|8|1df994e3-64ab-41b9-b008-fbe3f684b6aa|Inbound|Refund Related|Refund Enquiry|Very good |97009fbc-d7c2-4c6a-880f-21e0e4846dba|2023-08-23 09:58:00|2023-08-23 10:07:00|2023-08-23 00:00:00|A\. Richards|5|NaN|NaN|NaN|NaN|1|
|9|4fa42b3c-016f-42a2-a879-c1d7cc08dce8|Inbound|Returns|Missing|My problem solve for tq mam|2fe68e47-5163-44bc-b64c-6ed3204d99c8|2023-08-24 10:59:00|2023-08-24 11:35:00|2023-08-24 00:00:00|V\. Howell|5|NaN|NaN|NaN|NaN|1|
|10|9055c1b6-9ccd-42d7-9b1c-28a1c909e61f|Inbound|Offers & Cashback|Other Cashback|Very good people he is solved my all problems\.\.\. I am so happy\. Thanks   or thanks this people who helped me\.\.\. ??|dad1ead7-16c4-4319-81b8-4e3b24cb8272|2023-08-23 23:05:00|2023-08-23 23:16:00|2023-08-23 00:00:00|A\. Romero|5|NaN|NaN|NaN|NaN|1|
|11|14f0e42a-75aa-4d09-b374-0e1accaf6912|Inbound|Returns|Reverse Pickup Enquiry|Excellent thanks sir|48d27b9a-0381-4938-b096-65d3647780e9|2023-09-08 16:15:00|2023-09-08 16:16:00|2023-08-09 00:00:00|E\. Wilson|5|NaN|NaN|NaN|NaN|1|
|12|caa6fd1e-bf59-4583-964a-98f4a213064c|Outcall|Order Related|Delayed|Excellent |76b742cd-bf0c-4f9b-92bc-0b37a2b4f12f|2023-08-21 16:21:00|2023-08-21 16:27:00|2023-08-21 00:00:00|S\. Walker|5|NaN|NaN|NaN|NaN|1|
|13|444709ac-e21f-44f7-b925-b38d0feb77d1|Outcall|Cancellation|Not Needed|Nothing |6e76129d-b8cc-4fc7-8057-1a961a169765|2023-08-22 21:47:00|2023-08-22 21:50:00|2023-08-22 00:00:00|C\. Chavez|5|NaN|NaN|NaN|NaN|1|
|14|10571558-a847-46b0-9d0d-3249d3e84669|Inbound|Returns|Reverse Pickup Enquiry|Very nice |7e7de0df-20a8-4ca6-88e1-ec6a5adebcc2|2023-08-29 16:00:00|2023-08-29 16:02:00|2023-08-29 00:00:00|C\. Chavez|5|NaN|NaN|NaN|NaN|1|
|15|6dc6225c-8fc3-48ed-9469-f3aa077bb612|Inbound|Feedback|UnProfessional Behaviour|Good |c01495db-e05a-4ce1-9c7c-41a97e8bf907|2023-06-08 18:40:00|2023-06-08 18:42:00|2023-08-06 00:00:00|W\. James|4|NaN|NaN|NaN|NaN|1|
|16|7845f2df-41b3-4223-a654-623d46879b3b|Inbound|Returns|Exchange / Replacement|I like your explain |b5e98dd8-2c1f-4ff2-a2e8-17d12f6bb87d|2023-08-31 10:54:00|2023-08-31 11:09:00|2023-08-31 00:00:00|NaN|4|NaN|NaN|NaN|NaN|2|
|17|c097bdea-972a-4d02-b695-53378877d4aa|Outcall|Returns|Return request|Nice to talk with her\.Thank you|d6d4696b-f4f2-4351-b9b4-8f5b7d038101|2023-08-20 10:28:00|2023-08-20 10:34:00|2023-08-20 00:00:00|R\. Cook|5|NaN|NaN|NaN|NaN|1|
|18|369deaa2-f2ef-4f5b-855e-cb718c355ea6|Inbound|Returns|Fraudulent User|Good experience overall |NaN|2023-12-08 14:34:00|2023-08-13 00:30:00|2023-08-13 00:00:00|O\. Nguyen|4|NaN|NaN|NaN|NaN|1|
|19|b32d8148-4d7a-4e90-9e36-26e8ba9e4238|Inbound|Refund Related|Refund Related Issues|Good service |5761dc36-92f3-4344-a375-6e49eebdc3f1|2023-08-24 13:37:00|2023-08-24 13:39:00|2023-08-24 00:00:00|M\. Hickman|5|17/08/2023 13:59|FEROZEPUR|LifeStyle|276\.0|1|
```

```py
data_with_agents = data_with_agent_numbers.merge(agent_data, how='left', left_on='Agent_name_x', right_on="Agent_name")
data_with_agents.info()
```

```
Int64Index: 85907 entries, 0 to 85906
Data columns (total 22 columns):
 #   Column                Non-Null Count  Dtype         
---  ------                --------------  -----         
 0   Id                    85907 non-null  object        
 1   Channel               85907 non-null  object        
 2   category              85907 non-null  object        
 3   Sub-category          85907 non-null  object        
 4   Customer Remarks      28425 non-null  object        
 5   Order_id              66658 non-null  object        
 6   Issue_reported_Date   85907 non-null  datetime64[ns]
 7   Issue_responded_Date  85907 non-null  datetime64[ns]
 8   Survey_response_Date  85907 non-null  datetime64[ns]
 9   Agent_name_x          68749 non-null  object        
 10  CSAT Score            85907 non-null  int64         
 11  order_date_time       16950 non-null  object        
 12  Customer_City         16816 non-null  object        
 13  Product_category      16932 non-null  object        
 14  Item_price            16942 non-null  float64       
 15  Agent_name_y          85907 non-null  int64         
 16  Agent                 68749 non-null  object        
 17  Supervisor            68749 non-null  object        
 18  Manager               68749 non-null  object        
 19  Tenure Bucket         68749 non-null  object        
 20  Shift                 68749 non-null  object        
 21  Agent_name            68749 non-null  object        
dtypes: datetime64[ns](3), float64(1), int64(2), object(16)
memory usage: 15.1+ MB
```

```py
data_with_agents = data_with_agents.drop(['Agent_name_x', 'Agent_name_y', 'Agent_name'], axis=1)
```

```py
sns.heatmap(data_with_agents.isnull(), yticklabels=False, cbar=False)
```

![](https://github.com/koroteevmv/ML_course/blob/main/ML5.1%20data%20integration/ml51-3.png?raw=true)

#### Задания для самостоятельного выполнения

1. При горизонтальной интеграции первой и второй таблицы выведите по каждому столбцу гистограмму распределения или плотность распределения, чтобы убедиться, что признаки выражаются по соотносимым шкалам.
1. При выполнении третьего задания убедитесь более явно, что обильное количество пропущенных значений не является ошибкой объединения. Проведите больше, чем одну точечную проверку.
1. Изучите документацию sklearn в части описания методов merge и join.

#### Контрольные вопросы

1. Какие ошибки могут произойти при горизонтальной интеграции данных?
1. Какими средствами можно проверить совпадение шкал измерения атрибута в разных датасетах?
1. Какие ошибки могут произойти при вертикальной интеграции данных?

#### Дополнительные задания

1. 
