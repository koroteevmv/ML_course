# vowel

**Author**:   
**Source**: Unknown -   
**Please cite**:   

*** Please use version 2 of this dataset. This version has a train/test feature that should be ignored in OpenML.

Introduction
                 ============
 
 In my work on context-sensitive learning, I used the "Deterding Vowel
 Recognition Data", but I found it necessary to reformulate the data.
 Implicit in the original data is contextual information on the
 speaker's gender and identity. For my work, it was necessary to make
 this information explicit. The file "vowel-context.data" adds the
 speaker's sex and identity as new features. The format of the data file
 is described below.
 
 
 Peter Turney
 peter@ai.iit.nrc.ca
 
 
 
                 References
                 ==========
 
 P. Turney. "Robust Classification With Context-Sensitive Features."
 Proceedings of the Sixth International Conference on Industrial
 and Engineering Applications of Artificial Intelligence and Expert
 Systems (IEA/AIE-93): 268-276. 1993.
 
 URL: ftp://ai.iit.nrc.ca/pub/ksl-papers/NRC-35074.ps.Z
 
 
 P. Turney. "Exploiting Context When Learning to Classify."
 Proceedings of the European Conference on Machine Learning
 (ECML-93): 402-407. 1993.
 
 URL: ftp://ai.iit.nrc.ca/pub/ksl-papers/NRC-35058.ps.Z
 
 
 
                 File Structure
                 ==============
 
 
         Column          Description
         -------------------------------
         0               Train or Test
         1               Speaker Number
         2               Sex
         3               Feature 0
         4               Feature 1
         5               Feature 2
         6               Feature 3
         7               Feature 4
         8               Feature 5
         9               Feature 6
         10              Feature 7
         11              Feature 8
         12              Feature 9
         13              Class
 
 
 
 
                 Numerical Codes
                 ===============
 
 
         Speak