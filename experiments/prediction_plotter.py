import csv
import os
import numpy as np
import pandas as pd
from IPython.core.display import display
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

bucketing = ["single", "state", "cluster"]
cls_method = ["xgboost", "rf", "logit"]
dataset = ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
# bucketing = ["single"]
# cls_method = ["xgboost"]
# dataset = ["sepsis_cases_2"]
results_folder = "results"
postprocessing_folder = 'results_postprocessing'
metrics = ['score, score2, score3, score4, score5']


def get_overall_metric_scores(filepath):
    """
    Helper function:
    Gets the overall metrics (AUC, Accuracy, F-score, prec and recall)
    from a CSV file in the results

    :return a Dataframe containing all the overall metrics
    """

    df = pd.read_csv(filepath, sep=";")

    df_overall = df[(df['nr_events'] == -1) & (df['metric'] == 'auc')]

    drop_cols = ['nr_events', 'n_iter']

    df_metrics = df_overall.drop(drop_cols, axis='columns')

    return df_metrics


def calc_diff(df_before, df_after):
    """
    Helper function:
    calculate the difference between the metric results of the Dataframe before Rgttm
    and the Dataframe after Rgttm.
    Metrics are (AUC, Accuracy, F-score, precision and recall)

    :return a Dataframe containing the difference of the overall metrics after Rgttm is applied (before - after)
    """

    df_before = df_before.drop(['index'], axis='columns')

    dataset = (df_before['dataset']).iloc[0]
    method = (df_before['method']).iloc[0]
    cls = (df_before['cls']).iloc[0]
    auc = (df_before['score'] - df_after['score']).iloc[0]
    acc = (df_before['score2'] - df_after['score2']).iloc[0]
    prec = (df_before['score3'] - df_after['score3']).iloc[0]
    recall = (df_before['score4'] - df_after['score4']).iloc[0]
    fscore = (df_before['score5'] - df_after['score5']).iloc[0]

    cols_diff = ['dataset', 'method', 'cls', 'auc_diff', "acc_diff", "prec_diff", "recall_diff", "fscore_diff"]
    df_diff = pd.DataFrame([[dataset, method, cls, auc, acc, prec, recall, fscore]], columns=[cols_diff])
    return df_diff


def get_all_metric_scores(filepath):
    """
    Helper function:
    Gets the individual metrics (AUC, Accuracy, F-score, prec and recall)
    from a CSV file in the results

    :return a Dataframe containing all the overall metrics
    """
    df = pd.read_csv(filepath, sep=";")

    df_metric_rows = df[(df['nr_events'] != -1) & (df['metric'] == 'auc')].reset_index()

    drop_cols = ['index', 'n_iter']

    df_metrics = df_metric_rows.drop(drop_cols, axis='columns')

    return df_metrics


def calc_all_diffs(df_before, df_after):
    cols = ['dataset', 'method', 'cls', 'auc_diff', "acc_diff", "prec_diff", "recall_diff", "fscore_diff"]
    df_diffs = df_before[['dataset', 'method', 'cls', 'nr_events']]
    df_diffs['auc_diff'] = df_after.score - df_before.score
    df_diffs['acc_diff'] = df_after.score2 - df_before.score2
    df_diffs['prec_diff'] = df_after.score3 - df_before.score3
    df_diffs['recall_diff'] = df_after.score4 - df_before.score4
    df_diffs['fscore_diff'] = df_after.score5 - df_before.score5
    display(df_diffs)
    return df_diffs


def magic(df_before, df_after):
    """
       Takes the first Dataframe and essentially just returns it without changing anything.
       For some reason this was necessary (probably some kind of type error), hence it's called magic

       :return a Dataframe containing the overall metrics before Rgttm is applied
       """
    df_before = df_before.drop(['index'], axis='columns')

    dataset = (df_before['dataset']).iloc[0]
    method = (df_before['method']).iloc[0]
    cls = (df_before['cls']).iloc[0]
    auc = (df_before['score']).iloc[0]
    acc = (df_before['score2']).iloc[0]
    prec = (df_before['score3']).iloc[0]
    recall = (df_before['score4']).iloc[0]
    fscore = (df_before['score5']).iloc[0]

    cols_diff = ['dataset', 'method', 'cls', 'auc', "acc", "prec", "recall", "fscore"]
    df_diff = pd.DataFrame([[dataset, method, cls, auc, acc, prec, recall, fscore]], columns=[cols_diff])
    return df_diff


def generate_differences_overall_metrics():
    cols = ['dataset', 'method', 'cls', 'auc_diff', "acc_diff", "prec_diff", "recall_diff", "fscore_diff"]
    master_table = pd.DataFrame(columns=[cols])
    for log in dataset:
        for cls in cls_method:
            for bucket in bucketing:
                print("Bucket: " + bucket)
                print("Classifier: " + cls)
                path_before_rgttm = os.path.join(results_folder,
                                                 'performance_results_' + cls + '_' + log + '_' + bucket + '_laststate.csv')
                metrics_before_rgttm = get_overall_metric_scores(path_before_rgttm).reset_index()
                display(metrics_before_rgttm)

                path_after_rgttm = os.path.join(results_folder,
                                                'performance_results_' + cls + '_' + log + '_' + bucket + '_laststate_gap1_rgttm.csv')
                metrics_after_rgttm = get_overall_metric_scores(path_after_rgttm).reset_index()
                display(metrics_after_rgttm)

                df_differences = calc_diff(metrics_before_rgttm, metrics_after_rgttm)
                display(df_differences)
                display(master_table)

                master_table = pd.concat([master_table, df_differences])

    master_table.to_csv(os.path.join(postprocessing_folder, 'differences_overall_metrics.csv'), sep=';')


def generate_overall_metrics():
    cols = ['dataset', 'method', 'cls', 'auc', "acc", "prec", "recall", "fscore"]
    master_table_metrics = pd.DataFrame(columns=[cols])
    for log in dataset:
        for cls in cls_method:
            for bucket in bucketing:
                print("Bucket: " + bucket)
                print("Classifier: " + cls)
                path_before_rgttm = os.path.join(results_folder,
                                                 'performance_results_' + cls + '_' + log + '_' + bucket + '_laststate.csv')
                metrics_before_rgttm = get_overall_metric_scores(path_before_rgttm).reset_index()
                display(metrics_before_rgttm)

                path_after_rgttm = os.path.join(results_folder,
                                                'performance_results_' + cls + '_' + log + '_' + bucket + '_laststate_gap1_rgttm.csv')
                metrics_after_rgttm = get_overall_metric_scores(path_after_rgttm).reset_index()
                display(metrics_after_rgttm)

                df_differences = magic(metrics_before_rgttm, metrics_after_rgttm)
                display(df_differences)
                display(master_table_metrics)

                master_table_metrics = pd.concat([master_table_metrics, df_differences])

    master_table_metrics.to_csv(os.path.join(postprocessing_folder, 'overall_metrics.csv'), sep=';')


########### Graph the difference of the metric score of each prefix length  ##############
#######
def generate_differences_all_metrics():
    for log in dataset:
        for cls in cls_method:
            for bucket in bucketing:
                path_before_rgttm = os.path.join(results_folder,
                                                 'performance_results_' + cls + '_' + log + '_' + bucket + '_laststate.csv')
                metrics_before_rgttm = get_all_metric_scores(path_before_rgttm)
                display(metrics_before_rgttm)

                path_after_rgttm = os.path.join(results_folder,
                                                'performance_results_' + cls + '_' + log + '_' + bucket + '_laststate_gap1_rgttm.csv')
                metrics_after_rgttm = get_all_metric_scores(path_after_rgttm)
                display(metrics_after_rgttm)

                df_differences = calc_all_diffs(metrics_before_rgttm, metrics_after_rgttm)

                df_differences.to_csv(os.path.join(postprocessing_folder,
                                                   'differences_all_metrics_' + log + '_' + cls + '_' + bucket + '.csv'))


#
# ## create/plot graphs
def generate_graphs():
    for cls in cls_method:
        for log in dataset:
            df_graph = pd.DataFrame()
            for bucket in bucketing:
                path_all_metrics = os.path.join(postprocessing_folder,
                                                'differences_all_metrics_' + log + '_' + cls + '_' + bucket + '.csv')
                df = pd.read_csv(path_all_metrics, sep=",")
                if not ('dataset' in df_graph):
                    df_graph['dataset'] = df.dataset
                    df_graph['prefix_length'] = df.nr_events
                df_graph['' + bucket] = df.auc_diff
            display(df_graph)
            df_graph.to_csv(os.path.join(postprocessing_folder, 'for_graphing_' + log + '_' + cls + '.csv'), sep=';')
            df_graph.plot(x='prefix_length', y=bucketing, kind='line', ylabel='AUC difference', title=log)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(postprocessing_folder, 'graph_' + log + '_' + cls + '.jpg'))


# plot before and after graphs
def generate_before_after_graphs(cls, log, bucket):
    path_before_rgttm = os.path.join(results_folder,
                                     'performance_results_' + cls + '_' + log + '_' + bucket + '_laststate.csv')
    metrics_before_rgttm = get_all_metric_scores(path_before_rgttm)
    display(metrics_before_rgttm)

    path_after_rgttm = os.path.join(results_folder,
                                    'performance_results_' + cls + '_' + log + '_' + bucket + '_laststate_gap1_rgttm.csv')
    metrics_after_rgttm = get_all_metric_scores(path_after_rgttm)
    display(metrics_after_rgttm)

    df_combined = pd.DataFrame()
    df_combined['prefix_length'] = metrics_before_rgttm.nr_events
    df_combined['auc_before'] = metrics_before_rgttm.score.round(2)
    df_combined['auc_after'] = metrics_after_rgttm.score.round(2)

    display(df_combined)

    df_combined.to_csv(
        os.path.join(postprocessing_folder, 'combined_for_graphing_' + log + '_' + cls + '_' + bucket + '.csv'),
        sep=';')
    df_combined.plot(x='prefix_length', y=['auc_before', 'auc_after'], kind='line', ylabel='AUC', title=cls + '_' + bucket, grid=True)
    plt.tight_layout()
    plt.savefig(os.path.join(postprocessing_folder, 'graph_before_after_' + log + '_' + cls + '_' + bucket + '.jpg'))


#
for log in dataset:
    for cls in cls_method:
        for bucket in bucketing:
            generate_before_after_graphs(cls, log, bucket)
# generate_overall_metrics()
# generate_differences_all_metrics()
# generate_graphs()
# generate_before_after_graphs('xgboost', 'sepsis_cases_2', 'single')
