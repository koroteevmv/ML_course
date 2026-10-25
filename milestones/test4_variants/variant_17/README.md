# phoneme

**Author**: Dominique Van Cappel, THOMSON-SINTRA  
**Source**: [KEEL](http://sci2s.ugr.es/keel/dataset.php?cod=105#sub2), [ELENA](https://www.elen.ucl.ac.be/neural-nets/Research/Projects/ELENA/databases/REAL/phoneme/) - 1993  
**Please cite**: None  

The aim of this dataset is to distinguish between nasal (class 0) and oral sounds (class 1). Five different attributes were chosen to characterize each vowel: they are the amplitudes of the five first harmonics AHi, normalised by the total energy Ene (integrated on all the frequencies): AHi/Ene. The phonemes are transcribed as follows: sh as in she, dcl as in dark, iy as the vowel in she, aa as the vowel in dark, and ao as the first vowel in water.  

### Source

The current dataset was formatted by the KEEL repository, but originally hosted by the [ELENA Project](https://www.elen.ucl.ac.be/neural-nets/Research/Projects/ELENA/elena.htm#stuff). The dataset originates from the European ESPRIT 5516 project: ROARS. The aim of this project was the development and the implementation of a real time analytical system for French and Spanish speech recognition.  

### Relevant information

Most of the already existing speech recognition systems are global systems (typically Hidden Markov Models and Time Delay Neural Networks) which recognizes signals and do not really use the speech
specificities.  On the contrary, analytical systems take into account the articulatory process leading to the different phonemes of a given language, the idea being to deduce the presence of each of the
phonetic features from the acoustic observation.

The main difficulty of analytical systems is to obtain acoustical parameters sufficiantly reliable. These acoustical measurements must :

   - contain all the information relative to the concerned phonetic feature.
   - being speaker independent.
   - being context independent.
   - being more or less robust to noise.

The primary acoustical observation is always voluminous (spectrum x N different obse