# DecisionTreeClassifier
Dataset:
ID: 1504 - The Steel Plates Faults Data Set is a dataset of steel plates' faults, classified into 7 different
types. The goal of the dataset is to train machine learning models for automatic pattern recognition of
these faults. The dataset consists of 27 features describing each fault's location, size, and other
characteristics, and 7 binary features indicating the type of fault. The target variable is a binary feature
that classifies each fault as either a 'common' or 'other' fault. The dataset includes 1941 instances, with
34 features and 2 distinct classes. There are no missing values in the dataset. This dataset can be used
for classification tasks and serves as a benchmark for evaluating the performance of different
classification algorithms in recognizing different types of faults in steel plates.
ID: 971 - The dataset is a binarized version of an original dataset, where the multi-class target feature
is converted into a two-class nominal target feature by labeling the majority class as positive ('P') and
all others as negative ('N'). There are 77 features, with att1 to att6 being numeric and the target feature
'binaryClass' being nominal. The task and the original source of the dataset are unknown. The dataset
has 2000 instances with no missing values.

Subtask 1:
In this subtask, we have used the DecisionTreeClassifier algorithm with varying values of minimum
sample leaf to determine the optimal value for the dataset. We have used ROC AUC score as the
evaluation metric and used 10-fold cross-validation. The results are plotted in the form of a graph with
minimum sample leaf on the x-axis and ROC AUC score on the y-axis. The graph shows two lines, one
for the training score and the other for the testing score. We have also marked the points where the
model is overfitting and underfitting.
The graph shows that the model performs the best when the minimum sample leaf is 94. When the value
of the minimum sample leaf is less than 94, the model overfits the data, which means it performs well
on the training data but poorly on the testing data. When the value of the minimum sample leaf is greater
than 94, the model underfits the data, which means it performs poorly on both training and testing data.

Subtask 2:
In this subtask, we have used the GridSearchCV function to tune the hyperparameters of the
DecisionTreeClassifier algorithm. We have used the same evaluation metric and 10-fold crossvalidation
as in subtask 1. The results are plotted in the form of a graph with minimum sample leaf on
the x-axis and ROC AUC score on the y-axis. The graph shows two lines, one for the training score and
the other for the testing score. We have also marked the best parameter value on the graph.
The graph shows that the model performs the best when the minimum sample leaf is 93. The
performance of the model is slightly better than the model trained in subtask 1 with minimum sample
leaf value of 94. The results suggest that tuning hyperparameters can improve the performance of the
model.

Conclusion:
From the results of subtask 1 and subtask 2, we can conclude that the DecisionTreeClassifier algorithm
can perform well on the dataset. However, the performance of the model depends on the value of
hyperparameters such as minimum sample leaf. Tuning hyperparameters can improve the performance
of the model.th att1 to att6 being numeric and the target feature 'binaryClass' being nominal. The task
and the original source of the dataset are unknown. The dataset has 2000 instances with no missing
values.
