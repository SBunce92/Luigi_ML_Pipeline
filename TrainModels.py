
import pandas as pd
import luigi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from operator import itemgetter
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from transliterate import translit, get_available_language_codes
from sklearn.model_selection import train_test_split
import eli5
from sklearn.metrics import roc_auc_score
import pickle
import csv
import os
import itertools


def draw_cv_graph(y_true, pred, fn='ROC.png'):
    fpr, tpr, t = roc_curve(y_true, pred, pos_label=1)
    prec, rec, t = precision_recall_curve(y_true, pred, pos_label=1)
    pairs = np.float64(sorted(zip(y_true, pred), key=itemgetter(1)))
    perc = np.percentile((1-pairs[:,1]), range(0, 110, 10))
    badrate = [np.mean(pairs[(tup[0] < (1-pairs[:,1])) & ((1-pairs[:,1]) < tup[1])][:,0]) for tup in zip(perc, perc[1:])]
    fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, pred, n_bins=10, normalize=False)
 
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    ax = ax.flatten()

    ax[0].plot([0, 0.5, 1], [0, 0.5, 1], color='green', ls='dashed')
    ax[0].plot(fpr, tpr, color='red')
    ax[0].plot(prec, rec, color='blue')
    ax[0].set_xlabel('RECALL/PRECISION VS. FALSE-ALARM, ROC_AUC = ' + str(roc_auc_score(y_true, pred))[:5],
                     fontsize=14)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position('right')
    ax[1].bar(np.linspace(0.03, 0.93, 10), badrate, 0.04, color='red')
    ax[1].set_xlabel('BADRATE-PER-DECILE', fontsize=14)
    ax[2].yaxis.tick_right()
    ax[2].yaxis.set_label_position('right')
    ax[2].hist(pairs[:,1], bins=20, color='blue', histtype='bar', normed=True)
    ax[2].set_xlabel('HISTOGRAM(CLASS=1)', fontsize=14)
    ax[2].yaxis.set_label_position('left')
    ax[3].plot(mean_predicted_value, fraction_of_positives,
               's-')
    ax[3].plot([0, 1], [0, 1], '--k', lw=0.5, alpha=0.5)
    ax[3].set_xlabel('CALIBRATION CURVE', fontsize=14)
 
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
    plt.suptitle('test')

    file = fn.open('w')
    plt.savefig(file, dpi=300)
    file.close()

class train_input_exists(luigi.ExternalTask):

    def output(self):
        return [luigi.LocalTarget('raw_data/train1-epf.tsv.gz'),
                luigi.LocalTarget('raw_data/train1-to-match.tsv.gz'),
                luigi.LocalTarget('raw_data/train1-topics.tsv.gz')]

class Generate_merged_models(luigi.Task):

    def requires(self):
        return train_input_exists()

    def output(self):
        return [luigi.LocalTarget('data_matricies/Xy_epf.csv'),
                luigi.LocalTarget('data_matricies/Xy_topics.csv'),
                luigi.LocalTarget('data_matricies/Xy_both.csv')]


    def run(self):

        # Load in EPF and matched defs data
        train_epf = pd.read_csv('raw_data/train1-epf.tsv.gz', delimiter= '\t', header= None)
        train_topics = pd.read_csv('raw_data/train1-topics.tsv.gz', delimiter= ' ', header= None)
        train_match = pd.read_csv('raw_data/train1-to-match.tsv.gz', delimiter= '\t')

        # Generate cuid columns to match to match file
        train_epf['cuid'] = train_epf[0].apply(lambda x: int(x.split('=')[1]))
        train_topics['cuid'] = train_topics[200].apply(lambda x: int(x.split('=')[1]))

        # Clean the raw cuid columns (with matching probability data and date)
        epf_clean = train_epf.set_index('cuid').drop(0, axis = 1)
        topics_clean = train_topics.set_index('cuid').drop(200, axis = 1)

        # Rename columns to prevent clash in both datasets
        epf_clean.columns = ['epffeat_' + str(col) for col in epf_clean.columns]
        topics_clean.columns = ['topicsfeat_' + str(col) for col in topics_clean.columns]

        # Extract only defs column from match dataframe
        defs_clean = train_match.set_index('cuid')[['defs']]

        #Merge datasets (epf+defs, topics+defs and both+defs)
        epf_with_defs = epf_clean.merge(defs_clean, left_index=True, right_index= True)
        topics_with_defs = topics_clean.merge(defs_clean, left_index=True, right_index= True)
        both_with_defs = epf_clean.merge(topics_clean, 
                                        left_index=True,
                                        right_index= True).merge(
                                                                defs_clean,
                                                                left_index=True,
                                                                right_index= True)

        # output data
        f_epf = self.output()[0].open('w')
        epf_with_defs.to_csv(f_epf, index=False)
        f_epf.close()

        f_topics = self.output()[1].open('w')
        topics_with_defs.to_csv(f_topics, index=False)
        f_topics.close()

        f_both = self.output()[2].open('w')
        both_with_defs.to_csv(f_both, index=False)
        f_both.close()


class LogReg(luigi.Task):
    
    C_param = luigi.FloatParameter(default=1.0)
    model_param = luigi.Parameter(default = 'both')

    def requires(self):
        return Generate_merged_models()

    def output(self):

        filename = 'models/LR-param-{}_C-param-%.2f.png'.format(self.model_param) % self.C_param
        return luigi.LocalTarget(filename)

    def run(self):
        
        # Reading in dataset, popping y values and forming X, y for training.
        input_Xy = pd.read_csv('data_matricies/Xy_{}.csv'.format(self.model_param))
        y = input_Xy.pop('defs')
        X = input_Xy.fillna(0)

        # Define model and produce cross_val predictions on data
        classifier_model = LogisticRegression(C = self.C_param, penalty= 'l2')
        y_pred = cross_val_predict(classifier_model, X, y, method= 'predict_proba')

        # Extract model parameters, score and dataset used into dict
        model_log = classifier_model.get_params()
        model_log['dataset'] = self.model_param
        model_log['ROC_AUC'] = roc_auc_score(y, y_pred[:,1])
        model_log['model_type'] = 'LogisticRegression'

        # Convert log to df and define Logpath
        model_df = pd.DataFrame([model_log.values()], columns= model_log.keys())
        log_path = 'models/logs.csv'

        # Writing log if log already exists
        if not os.path.isfile(log_path):
            model_df.to_csv(log_path, index = False)
        else:
            pd.read_csv(log_path).append(model_df).to_csv(log_path, index = False)

        # Finally save model performance graphs
        draw_cv_graph(y, y_pred[:,1], fn= self.output())

class RFClassifier(luigi.Task):
    
    model_param = luigi.Parameter(default = 'both')
    max_feats_param = luigi.FloatParameter(default = 1.0)

    def requires(self):
        return Generate_merged_models()

    def output(self):

        filename = 'models/RF-param-{}_maxfeats-param-%.2f.png'.format(self.model_param) % self.max_feats_param
        return luigi.LocalTarget(filename)

    def run(self):
        
        # Reading in dataset, popping y values and forming X, y for training.
        input_Xy = pd.read_csv('data_matricies/Xy_{}.csv'.format(self.model_param))
        y = input_Xy.pop('defs')
        X = input_Xy.fillna(0)

        # Define model and produce cross_val predictions on data
        classifier_model = RandomForestClassifier(max_features = self.max_feats_param)
        y_pred = cross_val_predict(classifier_model, X, y, method= 'predict_proba')

        # Extract model parameters, score and dataset used into dict
        model_log = classifier_model.get_params()
        model_log['dataset'] = self.model_param
        model_log['ROC_AUC'] = roc_auc_score(y, y_pred[:,1])
        model_log['model_type'] = 'RandomForestClassifier'

        # Convert log to df and define Logpath
        model_df = pd.DataFrame([model_log.values()], columns= model_log.keys())
        log_path = 'models/logs.csv'

        # Writing log if log already exists
        if not os.path.isfile(log_path):
            model_df.to_csv(log_path, index = False)
        else:
            pd.read_csv(log_path).append(model_df).to_csv(log_path, index = False)

        # Finally save model performance graphs
        draw_cv_graph(y, y_pred[:,1], fn= self.output())

class LogRegBatch(luigi.Task):

    C_vals = luigi.ListParameter(default = [])
    model_vals = luigi.ListParameter(default = [])
    check_path = 'models/Check_LR.txt'

    def requires(self):

        return [[LogReg(C_param = float(C), model_param = model) 
        for C in self.C_vals] 
        for model in self.model_vals] 

    def run(self):
        # Generate text file  batch complete
        f = self.output().open('w')
        f.write('LR Models complete')  # python will convert \n to os.linesep
        f.close()  # you can omit in most cases as the destructor will call it

    def output(self):
        return luigi.LocalTarget(self.check_path)

class RFBatch(luigi.Task):

    max_feats_vals = luigi.ListParameter(default = [])
    model_vals = luigi.ListParameter(default = [])
    check_path = 'models/Check_RF.txt'

    def requires(self):

        return [[RFClassifier(max_feats_param = float(F), model_param = model) 
        for F in self.max_feats_vals] 
        for model in self.model_vals] 

    def run(self):
        # Generate text file  batch complete
        f = self.output().open('w')
        f.write('RF models complete')  # python will convert \n to os.linesep
        f.close()  # you can omit in most cases as the destructor will call it

    def output(self):
        return luigi.LocalTarget(self.check_path)

class ModelBatch(luigi.Task):

    model_types = luigi.ListParameter(default = [])

    def requires(self):
        models_to_run = []

        if 'LR' in self.model_types:
            models_to_run.append(LogRegBatch())
        if 'RF' in self.model_types:
            models_to_run.append(RFBatch())

        return [models_to_run]


    def run(self):
        return

    def output(self):
        return 


class NewClassifier(luigi.Task):
    
    model_param = luigi.Parameter(default = 'both') #epf both or topics
    classifier_params = luigi.DictParameter(default={})
    model_type = luigi.Parameter() #LR or RF

    def requires(self):
        return Generate_merged_models()

    def output(self):

        filename = 'models/' + str(dict(self.classifier_params)).strip("{}").replace(' ', '').\
        replace(',', '__').replace("u'", '').replace("'", '').replace(":", '-') + '__' +\
        str(self.model_param) + '__' + str(self.model_type) + '.png'

        return luigi.LocalTarget(filename)

    def run(self):

        # Reading in dataset, popping y values and forming X, y for training.
        input_Xy = pd.read_csv('data_matricies/Xy_{}.csv'.format(self.model_param))
        y = input_Xy.pop('defs')
        X = input_Xy.fillna(0)

        # Define model type to use
        if self.model_type == 'LR':
            classifier_model = LogisticRegression()

        if self.model_type == 'RF':
            classifier_model = RandomForestClassifier()


        #Set model params from luigi Parameter
        classifier_model.set_params(**self.classifier_params)

        #Cross_val_predict Train model
        y_pred = cross_val_predict(classifier_model, X, y, method= 'predict_proba')

        # Add model output to model.log
        model_log = classifier_model.get_params()
        model_log['dataset'] = self.model_param
        model_log['model_type'] = self.model_type
        model_log['ROC_AUC'] = roc_auc_score(y, y_pred[:,1])
        
        # Convert log to df and define Logpath
        model_df = pd.DataFrame([model_log.values()], columns= model_log.keys())
        log_path = 'models/logs.csv'

        # Writing log if log already exists
        if not os.path.isfile(log_path):
            model_df.to_csv(log_path, index = False)
        else:
            pd.read_csv(log_path).append(model_df).to_csv(log_path, index = False)

        # Finally save model performance graphs
        draw_cv_graph(y, y_pred[:,1], fn= self.output())



class NewBatch(luigi.Task):

    model_type = luigi.Parameter()
    parameter_spaces = luigi.DictParameter()

    def requires(self):

        return [NewClassifier(classifier_params = params, 
            model_param = 'epf',
            model_type = self.model_type) 
        for params in self.run()]


    def output(self):
        yield

    def run(self):

        def param_interpreter(param, func, min_val=0, max_val=0, steps=0, vals=None):           
            if func == 'defined_list':
                return {param: vals}         
            if func == 'linspace':
                return {param: (list(np.linspace(min_val, max_val, steps)))}            
            if func == 'logspace':
                return {param: (list(np.logspace(min_val, max_val, steps)))}

        def merge_dicts(*dict_args):
            """
            Given any number of dicts, shallow copy and merge into a new dict,
            precedence goes to key value pairs in latter dicts.
            """
            result = {}
            for dictionary in dict_args:
                result.update(dictionary)
            return result

        p_space = dict(self.parameter_spaces)

        keys_list = [key for key in p_space.keys()]
        values_list = [p_space[key] for key in p_space.keys()]

        param_spaces_list = []

        for number, key in enumerate(keys_list):

            if 'vals' in values_list[number]:
                param_spaces_list.append(param_interpreter(param = keys_list[number], 
                    func = values_list[number]['param_type'],
                    vals = values_list[number]['vals']))

            if 'vals' not in values_list[number]:
                param_spaces_list.append(param_interpreter(param = keys_list[number], 
                    func = values_list[number]['param_type'],
                    min_val = values_list[number]['min_val'],
                    max_val = values_list[number]['max_val'],
                    steps = values_list[number]['steps']))

        #Creates a list of lists, with each inner list representing space over one parameter
        list_of_param_spaces = [[{param_space.keys()[0] : value} for
                 value in param_space.values()[0]] for 
                 param_space in param_spaces_list]

        # Joins the lists of parameter spaces together, 
        # to make each list item a point on the parameter grid space
        joined_param_space_list = list(itertools.product(*list_of_param_spaces))

        #Combines the parameter mixes into a list of individual dictionaries
        final_parameter_dict_list = [merge_dicts(*parameter_list) for
        parameter_list in joined_param_space_list]

        return final_parameter_dict_list


    

if __name__ == '__main__':
    luigi.run()

