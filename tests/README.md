# final_project

This project first analyzed data from a file about blood donation statistics collected from a blood donation van in Taiwan.
Then, we used this data to create models using machine learning to predict outcomes of future cases in the dataset.

In the analysis file, we did our exploratory data analysis by plotting graphs of the various variables and seeing how they correlate.
kNN and kNN-log files used the k-Nearest Neighbors approach, with kNN-log first taking the logarithm of all of the values in the feature columns.
nb and nb-log files used the Naive Bayes approach, with nb-log first taking the logarithm of all of the values in the feature columns.
dt used the decision tree approach.
There is no dt-log file because taking the logarithm did not affect the outcomes of the decision tree model.
