import os
import pickle
import time
from sys import argv

import joblib
import numpy
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, top_k_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import BucketFactory
import EncoderFactory
from DatasetManager import DatasetManager

dataset_ref = argv[1]
params_dir = argv[2]
results_dir = argv[3]
bucket_method = argv[4]
cls_encoding = argv[5]
cls_method = argv[6]
gap = int(argv[7])
n_iter = int(argv[8])

if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

method_name = "%s_%s" % (bucket_method, cls_encoding)

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s" % formula for formula in range(5, 6)],
    "bpic2015": ["bpic2015_%s_f2" % (municipality) for municipality in range(1, 6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    # "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
    "sepsis_cases": ["sepsis_cases_2"]

    # "synth_log": ["synthetic_log_1", "synthetic_log_2", "synthetic_log_3"],
    # "synth_log_2": ["synthetic_log_4", "synthetic_log_5", "synthetic_log_6"],
    # "synth_log_3": ["synthetic_log_7", "synthetic_log_8", "synthetic_log_9"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]

train_ratio = 0.8
random_state = 22

# create results directory
# if not os.path.exists(os.path.join(params_dir)):
#     os.makedirs(os.path.join(params_dir))
# think the above was a typo
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))


# the mean of each group of n events
def adjust_with_regression_to_the_mean(preds, mean, confidence):
    return confidence * preds + mean * (1 - confidence)


for dataset_name in datasets:

    # load optimal params
    optimal_params_filename = os.path.join(params_dir,
                                           "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
        continue

    with open(optimal_params_filename, "rb") as fin:
        args = pickle.load(fin)

    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols,
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                        'fillna': True}

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    # split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

    if gap > 1:
        outfile = os.path.join(results_dir,
                               "median_performance_results_%s_%s_%s_gap%s.csv" % (cls_method, dataset_name, method_name, gap))
    else:
        outfile = os.path.join(results_dir,
                               "median_performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))

    start_test_prefix_generation = time.time()
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
    test_prefix_generation_time = time.time() - start_test_prefix_generation

    offline_total_times = []
    online_event_times = []
    train_prefix_generation_times = []
    for ii in range(n_iter):
        # create prefix logs
        start_train_prefix_generation = time.time()
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, gap)
        train_prefix_generation_time = time.time() - start_train_prefix_generation
        train_prefix_generation_times.append(train_prefix_generation_time)

        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method': bucket_encoding,
                         'case_id_col': dataset_manager.case_id_col,
                         'cat_cols': [dataset_manager.activity_col],
                         'num_cols': [],
                         'random_state': random_state}
        if bucket_method == "cluster":
            bucketer_args["n_clusters"] = int(args["n_clusters"])
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)

        start_offline_time_bucket = time.time()
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
        offline_time_bucket = time.time() - start_offline_time_bucket

        bucket_assignments_test = bucketer.predict(dt_test_prefixes)

        preds_all = []
        test_y_all = []
        nr_events_all = []
        offline_time_fit = 0
        current_online_event_times = []
        for bucket in set(bucket_assignments_test):
            if bucket_method == "prefix":
                current_args = args[bucket]
            else:
                current_args = args
            relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[
                bucket_assignments_train == bucket]
            relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[
                bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)

            # necessary due to bug that would cause test_y_all var to be smaller than preds_all and nr_events all
            if not (len(relevant_train_cases_bucket) == 0):
                nr_events_all.extend(list(dataset_manager.get_prefix_lengths(dt_test_bucket)))

            if len(relevant_train_cases_bucket) == 0:
                preds = [dataset_manager.get_class_ratio(train)] * len(relevant_test_cases_bucket)
                current_online_event_times.extend([0] * len(preds))
            else:
                dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                               relevant_train_cases_bucket)  # one row per event
                train_y = dataset_manager.get_label_numeric(dt_train_bucket)

                if len(set(train_y)) < 2:
                    # doesnt go into this
                    preds = [train_y[0]] * len(relevant_test_cases_bucket)
                    current_online_event_times.extend([0] * len(preds))
                    test_y_all.extend(dataset_manager.get_label_numeric(dt_test_bucket))
                else:
                    start_offline_time_fit = time.time()
                    feature_combiner = FeatureUnion(
                        [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

                    if cls_method == "rf":
                        cls = RandomForestClassifier(n_estimators=500,
                                                     max_features=current_args['max_features'],
                                                     random_state=random_state)

                    elif cls_method == "xgboost":
                        cls = xgb.XGBClassifier(objective='binary:logistic',
                                                n_estimators=500,
                                                learning_rate=current_args['learning_rate'],
                                                subsample=current_args['subsample'],
                                                max_depth=int(current_args['max_depth']),
                                                colsample_bytree=current_args['colsample_bytree'],
                                                min_child_weight=int(current_args['min_child_weight']),
                                                seed=random_state)

                    elif cls_method == "logit":
                        cls = LogisticRegression(C=2 ** current_args['C'],
                                                 random_state=random_state)

                    elif cls_method == "svm":
                        cls = SVC(C=2 ** current_args['C'],
                                  gamma=2 ** current_args['gamma'],
                                  random_state=random_state)

                    if cls_method == "svm" or cls_method == "logit":
                        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                    else:
                        pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

                    # fitting the training data onto the model/pipeline
                    pipeline.fit(dt_train_bucket, train_y)
                    print("pipeline fitted")
                    offline_time_fit += time.time() - start_offline_time_fit

                    pipeline_dump_filename = os.path.join(params_dir, 'pipeline_%s.pkl' % (dataset_name))
                    joblib.dump(pipeline, pipeline_dump_filename)

                    # predict separately for each prefix case
                    preds = []
                    test_all_grouped = dt_test_bucket.groupby(dataset_manager.case_id_col)
                    for _, group in test_all_grouped:

                        test_y_all.extend(dataset_manager.get_label_numeric(group))

                        start = time.time()
                        _ = bucketer.predict(group)
                        if cls_method == "svm":
                            pred = pipeline.decision_function(group)
                        else:
                            preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                            pred = pipeline.predict_proba(group)[:, preds_pos_label_idx]

                        pipeline_pred_time = time.time() - start
                        current_online_event_times.append(pipeline_pred_time / len(group))
                        preds.extend(pred)

            # necessary due to bug that would cause test_y_all var to be smaller than preds_all and nr_events all
            if not (len(relevant_train_cases_bucket) == 0):
                preds_all.extend(preds)


        offline_total_time = offline_time_bucket + offline_time_fit + train_prefix_generation_time
        offline_total_times.append(offline_total_time)
        online_event_times.append(current_online_event_times)
        print("Flag: Iterations num " + str(ii))

    print("**************Flag: Writing Regression to the mean results for " + str(dataset_name) + "******************")
    outfile_rgttm = os.path.join(results_dir,
                                 "median_performance_results_%s_%s_%s_gap%s_rgttm.csv" % (
                                     cls_method, dataset_name, method_name, gap))

    with open(outfile_rgttm, 'w') as fout2:
        # write headers for performance_results
        fout2.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "nr_events", "n_iter", "metric", "score", "metric2", "score2", "metric3", "score3", "metric4", "score4", "metric5", "score5"))

        confidence = map(lambda x: 1 - x * 2, (map(abs, map(lambda x: (round(x) - x), preds_all))))
        confidenceList = list(confidence)

        dt_results_rgttm = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all, "confidence": confidenceList, "adjusted_preds": ""})

        list_auc_collection_nr_events = []
        list_auc_collection_actual = []
        list_auc_collection_preds = []
        list_auc_collection_adjusted_preds = []
        list_auc_collection_mean = []
        list_auc_collection_confidence = []
        for nr_events, group in dt_results_rgttm.groupby("nr_events"):
            df_group_rgttm = pd.DataFrame(group).reset_index()
            mean = pd.DataFrame.median(df_group_rgttm.actual)
            print("df_group_rggtm")
            print(df_group_rgttm)
            print("df_group_rggtm.actual")
            print(df_group_rgttm.actual)
            print("Mean: ")
            print(mean)
            print("Group length: ")
            print(len(df_group_rgttm.index))
            print("Confidences: ")
            print(df_group_rgttm.confidence)
            index = 0

            while index < len(df_group_rgttm.index):
                df_group_rgttm.adjusted_preds[index] = adjust_with_regression_to_the_mean(df_group_rgttm.predicted[index], mean, df_group_rgttm.confidence[index])

                # collecting/duplicating these to calculate the overall AUC score. Due to grouping by nr_events it's difficult to
                # pair the adjusted values with their corresponding values in the original dt_results_rgttm, hence the index
                # print("nr_events: " + str(nr_events))
                list_auc_collection_nr_events.append(nr_events)
                # print("Actual results: " + str(df_group_rgttm.actual[index]))
                list_auc_collection_actual.append(df_group_rgttm.actual[index])
                # print("Predicted results: " + str(df_group_rgttm.predicted[index]))
                list_auc_collection_preds.append(df_group_rgttm.predicted[index])
                # print("Adjusted predictions: " + str(df_group_rgttm.adjusted_preds[index]))
                list_auc_collection_adjusted_preds.append(df_group_rgttm.adjusted_preds[index])
                # print("mean: " + str(mean))
                list_auc_collection_mean.append(mean)
                # print("Confidence: " + str(df_group_rgttm.confidence[index]))
                list_auc_collection_confidence.append(df_group_rgttm.confidence[index])
                index = index + 1
                # print(df_group_rgttm)

            # writing roc/auc score AFTER RGTTM based on actual vs predicted results (sorted by number of events)
            print("Writing AUC for nr_events: " + str(nr_events))
            rounded_values_group = df_group_rgttm['adjusted_preds'].astype(float).round(0)

            if len(set(group.actual)) < 2:
                fout2.write(
                    "%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc", np.nan, "accuracy", np.nan, "precision", np.nan, "recall", np.nan, "fscore", np.nan))
            else:
                # print("ROUNDED VALUES GROUP:")
                # print(rounded_values_group)
                fout2.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1,
                                                              "auc", roc_auc_score(df_group_rgttm.actual, df_group_rgttm.adjusted_preds),
                                                              "accuracy", accuracy_score(df_group_rgttm.actual, rounded_values_group),
                                                              "precision", precision_score(df_group_rgttm.actual, rounded_values_group),
                                                              "recall", recall_score(df_group_rgttm.actual, rounded_values_group),
                                                              "fscore", f1_score(df_group_rgttm.actual, rounded_values_group)))

        dt_auc_collection = pd.DataFrame({'nr_events': list_auc_collection_nr_events, 'actual':  list_auc_collection_actual, 'predicted': list_auc_collection_preds, 'adjusted_preds': list_auc_collection_adjusted_preds,
                                          "mean": list_auc_collection_mean, "confidence": list_auc_collection_confidence})
        comparison_path = os.path.join(results_dir, "median_comparison_after_rgttm_%s_%s_%s_gap%s.csv" % (cls_method, dataset_name, method_name, gap))
        dt_auc_collection.to_csv(comparison_path)

        # writing roc/auc score AFTER RGTTM based on actual vs predicted results (THE ENTIRE SET)
        rounded_values = dt_auc_collection['adjusted_preds'].astype(float).round(0)
        fout2.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1,
                "auc", roc_auc_score(dt_auc_collection.actual, dt_auc_collection.adjusted_preds),
                "accuracy", accuracy_score(dt_auc_collection.actual, rounded_values),
                "precision", precision_score(dt_auc_collection.actual, rounded_values),
                "recall", recall_score(dt_auc_collection.actual, rounded_values),
                "fscore", f1_score(dt_auc_collection.actual, rounded_values)))

    print("**************Flag: Writing standard results for " + str(dataset_name) + "******************")
    with open(outfile, 'w') as fout:
        # write headers for performance_results
        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (
        "dataset", "method", "cls", "nr_events", "n_iter", "metric", "score", "metric2", "score2", "metric3", "score3",
        "metric4", "score4", "metric5", "score5"))

        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "test_prefix_generation_time", test_prefix_generation_time))

        for ii in range(len(offline_total_times)):
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, ii, "train_prefix_generation_time",
                train_prefix_generation_times[ii]))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, ii, "offline_time_total", offline_total_times[ii]))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, ii, "online_time_avg", np.mean(online_event_times[ii])))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, ii, "online_time_std", np.std(online_event_times[ii])))

        dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all})

        for nr_events, group in dt_results.groupby("nr_events"):
            # if the bucketing group is too small, give a NaN as result
            if len(set(group.actual)) < 2:
                fout.write(
                    "%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc", np.nan, "accuracy", np.nan, "precision", np.nan, "recall", np.nan, "fscore", np.nan))
            # writing roc/auc score based on actual vs predicted results (sorted by number of events)
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1,
                                                        "auc", roc_auc_score(group.actual, group.predicted),
                                                        "accuracy", accuracy_score(group.actual, numpy.round(group.predicted)),
                                                        "precision", precision_score(group.actual, numpy.round(group.predicted)),
                                                        "recall", recall_score(group.actual, numpy.round(group.predicted)),
                                                        "fscore", f1_score(group.actual, numpy.round(group.predicted)),
                                                        ))

        rounded_values = dt_results['predicted'].astype(float).round(0)
        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1,
            "auc", roc_auc_score(dt_results.actual, dt_results.predicted),
            "accuracy", accuracy_score(dt_results.actual, rounded_values),
            "precision", precision_score(dt_results.actual, rounded_values),
            "recall", recall_score(dt_results.actual, rounded_values),
            "fscore", f1_score(dt_results.actual, rounded_values)))

        online_event_times_flat = [t for iter_online_event_times in online_event_times for t in iter_online_event_times]
        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "online_time_avg", np.mean(online_event_times_flat)))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "online_time_std", np.std(online_event_times_flat)))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "offline_time_total_avg", np.mean(offline_total_times)))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "offline_time_total_std", np.std(offline_total_times)))
        print("Flag: End of predictions for " + str(dataset_name))
