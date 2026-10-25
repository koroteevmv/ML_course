# prnn_crabs

Краткое описание: 2 класса, маленький датасет.
Целевая переменная: `target`.
Размер: 200 строк, 7 признаков.

Полное описание (фрагмент):

**Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Datasets for `Pattern Recognition and Neural Networks' by B.D. Ripley
=====================================================================

Cambridge University Press (1996)  ISBN  0-521-46086-7

The background to the datasets is described in section 1.4; this file
relates the computer-readable files to that description.



Cushing's syndrome
------------------

Data from Aitchison & Dunsmore (1975, Tables 11.1-3).

Data file Cushings.dat has four columns,

Label of the patient
Tetrhydrocortisone  (mg/24hr)
Pregnanetriol  (mg/24hr)
Type

The type of the last six patients (u1 to u6) should be
regarded as unknown.  (The code `o' indicates `other').



synthetic two-class problem
---------------------------

Data from Ripley (1994a).

This has two real-valued co-ordinates (xs and ys) and a class (xc)
which is 0 or 1.

Data file  synth.tr   has 250 rows of the training set
synth.te   has 1000 rows of the test set  (not used here)



viruses
-------

This is a dataset on 61 viruses with rod-shaped particles affecting
various crops (tobacco, tomato, cucumber and others) described by
{Fauquet et al. (1988) and analysed by Eslava-G\'omez (1989).  There
are 18 measurements on each virus,  the number of amino acid residues
per molecule of coat protein.

Data file  viruses.dat  has 61 rows of 18 counts
virus3.dat   has 38 rows corresponding to the distinct
Tobamoviruses.

The whole dataset is in order Hordeviruses
