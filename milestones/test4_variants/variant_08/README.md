# ionosphere

**Author**: Space Physics Group, Applied Physics Laboratory, Johns Hopkins University. Donated by Vince Sigillito.  
**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ionosphere)  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html) 

**Johns Hopkins University Ionosphere database**  
This radar data was collected by a system in Goose Bay, Labrador.  This system consists of a phased array of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts.  See the paper for more details.  

### Attribute information
Received signals were processed using an autocorrelation function whose arguments are the time of a pulse and the pulse number.  There were 17 pulse numbers for the Goose Bay system.  Instances in this database are described by 2 attributes per pulse number, corresponding to the complex values returned by the function resulting from the complex electromagnetic signal.

The targets were free electrons in the ionosphere.  "Good" (g) radar returns are those showing evidence of some type of structure in the ionosphere.  "Bad" (b) returns are those that do not; their signals pass through the ionosphere.  

### Relevant papers  
Sigillito, V. G., Wing, S. P., Hutton, L. V., & Baker, K. B. (1989). Classification of radar returns from the ionosphere using neural networks. Johns Hopkins APL Technical Digest, 10, 262-266.

Downloaded from openml.org.