# Decision-Tree-Classifier

# Overview
In this assignment, you will implement a decision tree algorithm and use it to determine one of the indoor
locations based on WIFI signal strengths collected from a mobile phone. See Figure 1 for an illustration of
the experimental scenario. The results of your experiments should be discussed in the report. You should also
deliver the code you have written.

# Setup
Please see the guidelines provided on Scientia to set up your machines for the coursework. It is your own
responsibility to ensure that your code runs on the DoC lab machines. We reserve the right to reduce your
marks by 30% for any bits of your code that cannot be run.

# Coursework
## Step 1: Loading data
You can load the datasets from the files ”WIFI db/clean dataset.txt” and
”WIFI db/noisy dataset.txt”. They contain a 2000x8 array. This array represents a dataset of 2000 samples.
Each sample is composed of 7 WIFI signal strength while the last column indicates the room number in which
the user is standing (i.e., the label of the sample). All the features in the dataset are continuous except
the room number. You can load the text file with the ”loadtxt” function from Numpy. Given the nature
of the dataset you will have to build decision trees capable of dealing with continuous attributes and multiple
labels.

## For the report:
There is nothing to add in the report for this section.

## Step 2: Creating Decision Trees
To create the decision tree, you will write a recursive function called decision tree learning(), that takes as
arguments a matrix containing the dataset and a depth variable (which is used to compute the maximal depth
of the tree, for plotting purposes for instance). The label of the training dataset is assumed to be the last
column of the matrix. The pseudo-code of this function is described below. This pseudo-code is taken from
Artificial Intelligence: A Modern Approach by Stuart Russell and Peter Norvig (Figure 18.5), but modified to
take into account that the considered dataset contains continuous attributes (see section 18.3.6 of the book).
The function FIND SPLIT chooses the attribute and the value that results in the highest information gain.
Because the dataset has continuous attributes, the decision-tree learning algorithms search for the split point
(defined by an attribute and a value) that gives the highest information gain. For instance, if you have two
attributes (A0 and A1) with values that both range from 0 to 10, the algorithm might determine that splitting
the dataset according to ”A1>4” gives the most information. An efficient method for finding good split points
is to sort the values of the attribute, and then consider only split points that are between two examples in sorted
order, while keeping track of the running totals of examples of each class for each side of the split point.
To evaluate the information gain, suppose that the training dataset Sall has K different labels. We can define
two subsets (Slef t and Sright) of the dataset depending on the splitting rule (for instance ”A1>4”) and for each
dataset and subset, we can compute the distribution (or probability) of each label. For instance, {p
1p
2
. . . pK}
(p
k
is the number of samples with the label k divided by the total number of samples from the initial dataset).
The information gain is defined by using the general definition of the entropy as follow:

![image](https://user-images.githubusercontent.com/83886065/229232520-c73f0e70-74a6-484f-a69c-1048c318def4.png)

![image](https://user-images.githubusercontent.com/83886065/229232968-7a21f9c3-23d6-4e4b-b13d-ab36bd6968f1.png)

![image](https://user-images.githubusercontent.com/83886065/229233011-8dbf9c54-2256-466f-b223-8a644c92beda.png)

Figure 2: example of Decision tree visualization (not all the tree is displayed, and this tree has been trained on
a different dataset.)

![image](https://user-images.githubusercontent.com/83886065/229233073-2d3ba029-a8ea-4381-8ac7-2954ea2eca6b.png)



