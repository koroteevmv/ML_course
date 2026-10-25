# credit-g

**Author**: Dr. Hans Hofmann  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) - 1994    
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)

**German Credit dataset**  
This dataset classifies people described by a set of attributes as good or bad credit risks.

This dataset comes with a cost matrix: 
``` 
Good  Bad (predicted)  
Good   0    1   (actual)  
Bad    5    0  
```

It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).  

### Attribute description  

1. Status of existing checking account, in Deutsche Mark.  
2. Duration in months  
3. Credit history (credits taken, paid back duly, delays, critical accounts)  
4. Purpose of the credit (car, television,...)  
5. Credit amount  
6. Status of savings account/bonds, in Deutsche Mark.  
7. Present employment, in number of years.  
8. Installment rate in percentage of disposable income  
9. Personal status (married, single,...) and sex  
10. Other debtors / guarantors  
11. Present residence since X years  
12. Property (e.g. real estate)  
13. Age in years  
14. Other installment plans (banks, stores)  
15. Housing (rent, own,...)  
16. Number of existing credits at this bank  
17. Job  
18. Number of people being liable to provide maintenance for  
19. Telephone (yes,no)  
20. Foreign worker (yes,no)

Downloaded from openml.org.