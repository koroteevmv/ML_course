# sonar

**Author**:   
**Source**: Unknown -   
**Please cite**:   

NAME: Sonar, Mines vs. Rocks
 
 SUMMARY: This is the data set used by Gorman and Sejnowski in their study
 of the classification of sonar signals using a neural network [1].  The
 task is to train a network to discriminate between sonar signals bounced
 off a metal cylinder and those bounced off a roughly cylindrical rock.
 
 SOURCE: The data set was contributed to the benchmark collection by Terry
 Sejnowski, now at the Salk Institute and the University of California at
 San Deigo.  The data set was developed in collaboration with R. Paul
 Gorman of Allied-Signal Aerospace Technology Center.
 
 MAINTAINER: Scott E. Fahlman
 
 PROBLEM DESCRIPTION:
 
 The file "sonar.mines" contains 111 patterns obtained by bouncing sonar
 signals off a metal cylinder at various angles and under various
 conditions.  The file "sonar.rocks" contains 97 patterns obtained from
 rocks under similar conditions.  The transmitted sonar signal is a
 frequency-modulated chirp, rising in frequency.  The data set contains
 signals obtained from a variety of different aspect angles, spanning 90
 degrees for the cylinder and 180 degrees for the rock.
 
 Each pattern is a set of 60 numbers in the range 0.0 to 1.0.  Each number
 represents the energy within a particular frequency band, integrated over
 a certain period of time.  The integration aperture for higher frequencies
 occur later in time, since these frequencies are transmitted later during
 the chirp.
 
 The label associated with each record contains the letter "R" if the object
 is a rock and "M" if it is a mine (metal cylinder).  The numbers in the
 labels are in increasing order of aspect angle, but they do not encode the
 angle directly.
 
 METHODOLOGY: 
 
 This data set can be used in a number of different ways to test learning
 speed, quality of ultimate learning, ability to generalize, or combinations
 of these factors.
 
 In [1], Gorman and Sejnowski report two series of