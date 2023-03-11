# Implement Classifiers and Compare Performance Achieved
## Description
Course: Data Mining and Knowledge Discovery (Fall 2021) <br />
Task: Implement Decision Tree, KNN and Random Forest and compare the performance achieved by different classifiers

> Given a decision tree, classifying a test instance is straightforward. Starting from the root node, we apply its attribute test condition and follow the appropriate branch based on the outcome of the test. This will lead us either to another internal node, for which a new attribute test condition is applied, or to a leaf node. Once a leaf node is reached, we assign the class label associated with the node to the test instance.
>
> Tan, Pang-Ning, et al. Introduction to Data Mining EBook: Global Edition, Pearson Education, Limited, 2019.


> A nearest neighbor classiﬁer represents each example as a data point in a d -dimensional space, where d is the number of attributes. Given a test instance, we compute its proximity to the training instances according to one of the proximity measures described in Section 2.4 on page 91. The k -nearest neighbors of a given test instance z refer to the k training examples that are closest to z .
>
> Tan, Pang-Ning, et al. Introduction to Data Mining EBook: Global Edition, Pearson Education, Limited, 2019.


> Random forests attempt to improve the generalization performance by constructing an ensemble of decorrelated decision trees. Random forests build on the idea of bagging to use a diﬀerent bootstrap sample of the training data for learning decision trees. However, a key distinguishing feature of random forests from bagging is that at every internal node of a tree, the best splitting criterion is chosen among a small set of randomly selected attributes. In this way, random forests construct ensembles of decision trees by not only manipulating training instances (by using bootstrap samples similar to bagging), but also the input attributes (by using diﬀerent subsets of attributes at every internal node).
>
> Tan, Pang-Ning, et al. Introduction to Data Mining EBook: Global Edition, Pearson Education, Limited, 2019.



