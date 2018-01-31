# Luigi_ML_Pipeline
A process for managing parallelised Grid/Random search parameter optimisation in Luigi for binary classification tasks.

## TODO

1) Implement Other ML models, currently RandomForest and LogisticRegression are implemented. Suggestions: XGBoost, SVM, AdaBoost, CatBoost

2) Produce a framework for visualising the results across all classifiers and parameter sets.

3) Simplify the data processing step, make it easy to specify dependencies by filename.

## Framework


### LUIGI TASK NAME - Train_Input_Exists

This is a luigi.ExternalTask, which checks for the existence of the input files.

### LUIGI TASK NAME - Generate_Merged_Models 

Files in the **/raw_data/** directory consist currently of three files - topics, epf and match. 

epf is a file with '\t' delimiters, and no Header. Column 0 is the key, in the format 'EFH=1024071=20170918'

topics is a file with ' ' delimiters, and no Header. Column 200 is the key in the format 'EFH=1001674=20170801:3'

Both epf and topics contain one feature per column.

match is a file with '\t' delimiters and a Header. This file contains defs for y vector in training. It also contains some personal identifying information and the 'cuid' column contaning the keys.

In the merge process, cuid in match corresponds to '1024071' in 'EFH=1024071=20170918', and '1001674' in 'EFH=1001674=20170801:3'.
As a result, these strings are stripped from the epf and topics files, and then a merge is performed on these new strings, with the cuid column in match.

Three datasets are created using merge. 1) defs + topics (merge topics with match), 2) defs + epf (merge epf with match), 3) defs + epf + topics (merge epf with topics with match).

**Much of the above code is quite idiosyncratic, and dependent on the particular format of the data provided. If you wish to skip this and provide readily formatted data for training you must do the following:**

1) your readymade data must be in a csv file in the following format:

```
feat_1,feat_2,defs
0,1,1
1,2,0
0,0,1
```

Where there can be an unspecified number of **feat columns**, but the **defs** column must be present, and must contain the target vector.

This csv file should be saved as **/data_matricies/Xy_exampledata.csv** in module directory. It will be called in the config file by the string **exampledata**

2)

The **Luigi_Train_Evaluation.py** file must edited. The class **Generate_merged_models()** should be edited from:

```
class Generate_merged_models(luigi.Task):

    def requires(self):
        return train_input_exists()

    def output(self):
        return [luigi.LocalTarget('data_matricies/Xy_epf.csv'),
                luigi.LocalTarget('data_matricies/Xy_topics.csv'),
                luigi.LocalTarget('data_matricies/Xy_both.csv')]
```

to:

```
class Generate_merged_models(luigi.Task):

    def requires(self):
        return train_input_exists()

    def output(self):
        return [luigi.LocalTarget('data_matricies/Xy_exampledata1.csv'),
                luigi.LocalTarget('data_matricies/Xy_exampledata2.csv')]
```

Where the output return is a list of all of the necessary example files to run subsequent model training.

### LUIGI TASK NAME - New_Classifier



## Directory Structure

Insert here

## Config File Tutorial

Insert here

## Run from command line

To initialise luigi server

```
luigid
```

To run all batches specified in the config file

```
time PYTHONPATH='.' LUIGI_CONFIG_PATH=Batch_Config.cfg luigi RunAllBatches --module Luigi_Train_Evaluation --workers 4
```