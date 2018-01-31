# Luigi_ML_Pipeline

A process for managing parallelised Grid/Random search parameter optimisation in Luigi for binary classification tasks.

From raw training data and targets, this script generates a set of models with different parameters in order to evaluate the cross validated accuracy of the models. Parameter ranges are defined in the config file. Upon having run all models, logs of model performance are stored, and an html report detailing the best parameters for each model is produced. 

### TODO

1) Implement Other ML models, currently RandomForest and LogisticRegression are implemented. Suggestions: XGBoost, SVM, AdaBoost, CatBoost

2) Produce a framework for visualising the results across all classifiers and parameter sets.

3) Simplify the data processing step, make it easy to specify dependencies by filename.

# Tasks Framework

### LUIGI TASK NAME - Train_Input_Exists

This is a luigi.ExternalTask, which checks for the existence of the input files.

### LUIGI TASK NAME - Generate_Merged_Models 

Files in the **/raw_data/** directory consist currently of three files - topics, epf and match. 

epf is a file with '\t' delimiters, and no Header. Column 0 is the key, in the format 'EFH=1024071=20170918'

topics is a file with ' ' delimiters, and no Header. Column 200 is the key in the format 'EFH=1001674=20170801:3'

Both epf and topics contain one feature per column.

match is a file with '\t' delimiters and a Header. This file contains **defs** for y vector in training. It also contains some personal identifying information and the 'cuid' column contaning the keys.

In the merge process, cuid in match corresponds to '1024071' in 'EFH=1024071=20170918', and '1001674' in 'EFH=1001674=20170801:3'.
As a result, these strings are stripped from the epf and topics files, and then a merge is performed on these new strings, with the cuid column in match.

Three datasets are created using merge. 1) defs + topics (merge topics with match), 2) defs + epf (merge epf with match), 3) defs + epf + topics (merge epf with topics with match).

**Much of the above code is quite idiosyncratic, and dependent on the particular format of the data provided. If you wish to skip this and provide readily formatted data for training you must do the following:**

1) your ready-made data must be in a csv file in the following format:

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

    def output(self):
        return [luigi.LocalTarget('data_matricies/Xy_epf.csv'),
                luigi.LocalTarget('data_matricies/Xy_topics.csv'),
                luigi.LocalTarget('data_matricies/Xy_both.csv')]
```

to:

```
class Generate_merged_models(luigi.Task):

    def output(self):
        return [luigi.LocalTarget('data_matricies/Xy_exampledata1.csv'),
                luigi.LocalTarget('data_matricies/Xy_exampledata2.csv')]
```

Where the output return is a list of all of the necessary example files to run subsequent model training, in this case: **Xy_exampledata1.csv** and **Xy_exampledata2.csv**

### LUIGI TASK NAME - NewClassifier

This task is the main work engine of the script. Its dependencies are the files specified in **Generate_Merged_Models**. It takes three parameters. 

1) **model_param**, a string indicating which file the model is to be run on. In the case above, we would pass the string **'exampledata1'**

2) **model_type**, a string indicating the model function to be evaluated later. One example would be **'LogisticRegression()'**

3) **classifier_params**, a dictionary containing the names and values of parameters for this classifier. an example would be: **{"C": 0.01, "penalty": "l2"}**

This class trains the model, computes a ROC_AUC value, and saves the model and its score in a logs file. It outputs a png with model metrics. **This outputted png contains the model parameters in its name, and serves as proof for dependent tasks that NewClassifier has been run on a particular dataset, with a particular set of parameters.** 

### LUIGI TASK NAME - NewBatch

NewBatch processes parameter spaces to be distributed to individual NewClassifier tasks. It takes four parameters.

1) **model_type**, a string indicating the model function to be evaluated later. One example would be **'LogisticRegression()'**

2) **data_set**, a string indicating which file the model is to be run on. In the case above, we would pass the string **'exampledata1'**

3) **parameter_spaces**, a dictionary of parameters as keys, and further dictionaries defining their spaces as values. An example would be the following:

```
"parameter_spaces":
	{
	"C": {"param_type": "logspace", "min_val": -5, "max_val": 3, "steps": 10},
	"max_iter": {"param_type": "linspace_int", "min_val": 20, "max_val": 100, "steps": 5},
	"penalty": {"param_type": "defined_list", "vals": ["l1", "l2"]}
	}
```

This specifies that the parameters 'C', 'max_iter', and 'penalty' are to be passed. C is a logspace parameter, and is defined in a logarithmic space between a max and min power, over a number of steps. The parameter 'max_iter' is linearspace_int, indicating a range of integers between two values, with a given step. The parameter 'penalty' is a defined_list, indicating that it can be selected from a list of possible values.

4) **randomsearch_fraction**, a float between 0 and 1, indicating the proportion of the total grid to trial. The proportion is randomly selected from all parameter combinations.

### LUIGI TASK NAME - RunAllBatches

This task parses the config file to determine the range of models to run and the parameters to be parsed in NewBatch jobs. When implemented, this task will also be responsible for report visualisation.

## Directory Structure

/data_matricies/			- Folder for storing models ready for classification

/models/					- Folder for output of logs and model pngs

/raw_data/					- Folder for pre-processed data

Batch_Config.cfg 			- Configuration file

Luigi_Train_Evaluation.py	- Main Luigi Script


## Config File Example

Below is an example config file. It calls the task RunAllBatches, running a set of LogisticRegression() and RandomForestClassifier() models with given hyperparameter spaces. Each batch dictionary is a new item in the list.

```
[RunAllBatches]
batch_list:
	[

	{
	"model_type": "LogisticRegression()",
	"data_set":	"epf",
	"randomsearch_fraction" : 0.1,
	"parameter_spaces":
	{
	"C": {"param_type": "logspace", "min_val": -5, "max_val": 3, "steps": 10},
	"max_iter": {"param_type": "linspace_int", "min_val": 20, "max_val": 100, "steps": 5},
	"penalty": {"param_type": "defined_list", "vals": ["l1", "l2"]}
	}
	},

	{
	"model_type": "RandomForestClassifier()",
	"data_set":	"epf",
	"randomsearch_fraction" : 0.1,
	"parameter_spaces":
	{
	"max_depth": {"param_type": "linspace_int", "min_val": 1, "max_val": 10, "steps": 50},
	"n_estimators": {"param_type": "linspace_int", "min_val": 10, "max_val": 600, "steps": 20}
	}
	}

	]
```


## Run from command line

To initialise luigi server

```
luigid
```

To run all batches specified in the config file

```
time PYTHONPATH='.' LUIGI_CONFIG_PATH=Batch_Config.cfg luigi RunAllBatches --module Luigi_Train_Evaluation --workers 4
```