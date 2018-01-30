# Luigi_ML_Pipeline
A process for managing parallelised Grid/Random search parameter optimisation in Luigi for binary classification tasks.

#TODO

1) Implement Other ML models, currently RandomForest and LogisticRegression are implemented. Suggestions: XGBoost, SVM, AdaBoost, CatBoost

2) Produce a framework for visualising the results across all classifiers and parameter sets.

3) Simplify the data processing step, make it easy to specify dependencies by filename.

#Directory Structure

#Run from command line

luigid
time PYTHONPATH='.' LUIGI_CONFIG_PATH=Batch.cfg luigi RunAllBatches --module NewTrain --workers 4
