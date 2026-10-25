# kr-vs-kp

Author: Alen Shapiro
Source: [UCI](https://archive.ics.uci.edu/ml/datasets/Chess+(King-Rook+vs.+King-Pawn))
Please cite: [UCI citation policy](https://archive.ics.uci.edu/ml/citation_policy.html)

1. Title: Chess End-Game -- King+Rook versus King+Pawn on a7
(usually abbreviated KRKPA7). The pawn on a7 means it is one square
away from queening. It is the King+Rook's side (white) to move.

2. Sources:
(a) Database originally generated and described by Alen Shapiro.
(b) Donor/Coder: Rob Holte (holte@uottawa.bitnet). The database
was supplied to Holte by Peter Clark of the Turing Institute
in Glasgow (pete@turing.ac.uk).
(c) Date: 1 August 1989

3. Past Usage:
- Alen D. Shapiro (1983,1987), "Structured Induction in Expert Systems",
Addison-Wesley. This book is based on Shapiro's Ph.D. thesis (1983)
at the University of Edinburgh entitled "The Role of Structured
Induction in Expert Systems".
- Stephen Muggleton (1987), "Structuring Knowledge by Asking Questions",
pp.218-229 in "Progress in Machine Learning", edited by I. Bratko
and Nada Lavrac, Sigma Press, Wilmslow, England SK9 5BB.
- Robert C. Holte, Liane Acker, and Bruce W. Porter (1989),
"Concept Learning and the Problem of Small Disjuncts",
Proceedings of IJCAI. Also available as technical report AI89-106,
Computer Sciences Department, University of Texas at Austin,
Austin, Texas 78712.

4. Relevant Information:
The dataset format is described below. Note: the format of this
database was modified on 2/26/90 to conform with the format of all
the other databases in the UCI repository of machine learning databases.

5. Number of Instances: 3196 total

6. Number of Attributes: 36

7. Attribute Summaries:
Classes (2): -- White-can-win ("won") and White-cannot-win ("nowin").
I believe that White is deemed to be unable to win if the Black pawn
can safely advance.
Attributes: see Shapiro's book.

8. Missing Attributes: -- none

9. Class Distribution:
In 1669 of the positions (52%), White can win.
In 1527 of the positions (