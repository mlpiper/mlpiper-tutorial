import numpy as np
import pandas as pd
import argparse
import pickle
import multiprocessing
import xgboost as xgb

from parallelm.mlops import StatCategory as st
from parallelm.mlops import mlops as mlops
from parallelm.components import ConnectableComponent
from parallelm.ml_engine.python_engine import PythonEngine
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.table import Table
from parallelm.mlops.stats.graph import MultiGraph

from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn_pandas import DataFrameMapper



class XGBoostTrain(ConnectableComponent):

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        model_file = self._prep_and_train(parent_data_objs[0])
        self._logger.info("Model file saved: {}".format(model_file))
        return [model_file]

    def _prep_and_train(self, df_dataset):
        self.min_auc_requirement = self._params["auc_threshold"]
        self.max_ks_requirement = self._params["ks_threshold"]
        self.min_psi_requirement = self._params["psi_threshold"]
        train_on_col = self._params["train_on_column"]

        #mlops Init
        mlops.init()

        mlops.set_data_distribution_stat(df_dataset)
        y = df_dataset[train_on_col]
        self._logger.info("train_on_col= {}".format(train_on_col))
        self._logger.info("df_dataset {}".format(df_dataset.shape[1]))
        X = df_dataset.drop(train_on_col, axis=1)
        self._logger.info("df_dataset {}".format(X.shape[1]))

        # Splitting the data to train and test sets:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=self._params["validation_split"],
                                                            random_state=42)
        All_columns = X_train.columns.tolist()
        categorical_columns = self._params["categorical_cols"]
        mapper_list =[]
        for d in All_columns:
            if d in categorical_columns:
                mapper_list.append(([d], OneHotEncoder(handle_unknown='ignore')))
            else:
                mapper_list.append(([d], MinMaxScaler()))

        mapper = DataFrameMapper(mapper_list)

        ## Training
        # XGBoost Training:
        n_cpu = multiprocessing.cpu_count()

        xgboost_model = xgb.XGBClassifier(max_depth=int(self._params["max_depth"]),
                                          min_child_weight=int(self._params["min_child_weight"]),
                                          learning_rate=float(self._params["learning_rate"]),
                                          n_estimators=int(self._params["n_estimators"]),
                                          silent=True,
                                          objective=self._params["objective"],
                                          gamma=float(self._params["gamma"]),
                                          max_delta_step=int(self._params["max_delta_step"]),
                                          subsample=float(self._params["subsample"]),
                                          colsample_bytree=1,
                                          colsample_bylevel=1,
                                          reg_alpha=float(self._params["reg_alpha"]),
                                          reg_lambda=float(self._params["reg_lambda"]),
                                          scale_pos_weight=float(self._params["scale_pos_weight"]),
                                          seed=1,
                                          n_jobs=n_cpu,
                                          missing=None)

        final_model = Pipeline([("mapper", mapper), ("xgboost", xgboost_model)])
        final_model.fit(X_train, y_train)

        # Prediction and prediction distribution
        pred_labels = final_model.predict(X_test)
        pred_probs = final_model.predict_proba(X_test)

        # Accuracy calculation
        # Accuracy for the xgboost model
        accuracy = accuracy_score(y_test, pred_labels)
        self._logger.info("XGBoost Accuracy value: {0}".format(accuracy))
        #     Output accuracy of the chosen model using MCenter
        mlops.set_stat("XGBoost Accuracy", accuracy, st.TIME_SERIES)

        # Label distribution:
        # Label distribution in training
        value, counts = np.unique(y_test, return_counts=True)
        label_distribution = np.asarray((value, counts)).T
        self._logger.info("Validation Actual Label distributions: \n {0}".format(label_distribution))
        # Output Label distribution as a BarGraph using MCenter
        export_bar_table(label_distribution[:,0], label_distribution[:,1], "Validation - Actual Label Distribution")

        # Prediction distribution and prediction confidence distribution
        # Pred Label distribution in training
        pred_value, pred_counts = np.unique(pred_labels, return_counts=True)
        pred_label_distribution = np.asarray((pred_value, pred_counts)).T
        self._logger.info("XGBoost Validation Prediction Label Distributions: \n {0}".format(pred_label_distribution))
        # Output Pred label distribution as a BarGraph using MCenter
        export_bar_table(pred_label_distribution[:,0], pred_label_distribution[:,1], "Validation - XGBoost Prediction Distribution")

        # Pred confidence per label
        label_number = len(pred_counts)
        average_confidence = np.zeros(label_number)
        max_pred_probs = pred_probs.max(axis=1)
        for i in range(0, label_number):
            index_class = np.where(pred_labels == i)[0]
            if pred_counts[i] > 0:
                average_confidence[i] = np.sum(max_pred_probs[index_class])/(float(pred_counts[i]))
            else:
                average_confidence[i] = 0
        self._logger.info("XGBoost Validation Average Prediction confidence per label: \n {0}".format(average_confidence))

        # Output Pred label distribution as a BarGraph using MCenter
        export_bar_table(pred_value, average_confidence, "Validation - XGBoost Average confidence per class")

        # Confusion Matrix
        # XGBoost Confusion Matrix
        confmat = confusion_matrix(y_true=y_test, y_pred=pred_labels)
        self._logger.info("Confusion Matrix for XGBoost: \n {0}".format(confmat))
        # Output Confusion Matrix as a Table using MCenter
        export_confusion_table(confmat, "XGBoost")

        # Classification Report
        # XGBoost Classification Report
        class_rep = classification_report(y_true=y_test, y_pred=pred_labels, output_dict=True)
        self._logger.info("XGBoost Classification Report: \n {0}".format(class_rep))

        # AUC and ROC Curves
        # ROC for XGBoost model
        roc_auc = roc_auc_score(y_test, pred_probs[:, 1])
        self._logger.info("XGBoost ROC AUC value: {}".format(roc_auc))

        # Output ROC of the chosen model using MCenter
        mlops.set_stat("XGBoost ROC AUC", roc_auc, st.TIME_SERIES)

        if roc_auc <= self.min_auc_requirement:
            mlops.health_alert("[Training] AUC Violation From Training Node",
                               "AUC Went Below {}. Current AUC Is {}".format(self.min_auc_requirement, roc_auc))

        # ROC curve
        fpr, tpr, thr = roc_curve(y_test, pred_probs[:, 1])

        cg = MultiGraph().name("Receiver Operating Characteristic ").set_continuous()
        cg.add_series(label='Random curve ''', x=fpr.tolist(), y=fpr.tolist())
        cg.add_series(label='XGBoost ROC curve (area = {0:0.2f})'''.format(roc_auc), x=fpr.tolist(), y=tpr.tolist())
        cg.x_title('False Positive Rate')
        cg.y_title('True Positive Rate')
        mlops.set_stat(cg)

        # Feature importance comparison
        # XGBoost Feature importance
        export_feature_importance(final_model, list(X_train.columns), 5, "XGBoost")

        # KS Analysis
        max_pred_probs = pred_probs.max(axis=1)
        y_test0=np.where(y_test == 0)[0]
        y_test1=np.where(y_test == 1)[0]

        # KS for the XGBoost model
        ks = ks_2samp(max_pred_probs[y_test0], max_pred_probs[y_test1])
        ks_stat = ks.statistic
        ks_pvalue = ks.pvalue
        self._logger.info("KS values for XGBoost: \n Statistics: {} \n pValue: {}\n".format(ks_stat, ks_pvalue))

        # Output KS Stat of the chosen model using MCenter
        mlops.set_stat("KS Stats for CGBoost", ks_stat, st.TIME_SERIES)

        # raising alert if ks-stat goes above required threshold
        if ks_stat >= self.max_ks_requirement:
            mlops.health_alert("[Training] KS Violation From Training Node",
                               "KS Stat Went Above {}. Current KS Stat Is {}".format(self.max_ks_requirement, ks_stat))

        ks_table = Table().name("KS Stats for XGBoost").cols(["Statistic", "pValue"])
        ks_table.add_row([ks_stat, ks_pvalue])
        mlops.set_stat(ks_table)

        # PSI Analysis
        # Calculating PSI
        total_psi, psi_table = get_psi(self, max_pred_probs[y_test0], max_pred_probs[y_test1])
        psi_table_stat = Table().name("PSI Stats for XGBoost").cols(
            ["Base Pop", "Curr Pop", "Lower Bound", "Upper Bound", "Base Percent", "Curr Percent",
             "Segment PSI"])
        row_num = 1
        for each_value in psi_table.values:
            str_values = [str(i) for i in each_value]
            psi_table_stat.add_row(str(row_num), str_values)
            row_num += 1
        mlops.set_stat(psi_table_stat)
        self._logger.info("Total XGBoost PSI values: \n {}".format(total_psi))
        #     Output Total PSI of the chosen model using MCenter
        mlops.set_stat("Total XGBoost PSI ", total_psi, st.TIME_SERIES)

        if total_psi >= self.min_psi_requirement:
            mlops.health_alert("[Training] PSI Violation From Training Node",
                               "PSI Went Below {}. Current PSI Is {}".format(self.min_psi_requirement,
                                                                             total_psi))

        # ## Save the XGBoost Model
        model_file = open(self._params["output-model"], 'wb')
        pickle.dump(final_model, model_file)
        model_file.close()

        # ## Finish the program
        mlops.done()

        return(model_file)


def export_bar_table(bar_names, bar_data, title_name):
    """
    This function provides a bar_graph for a bar type data at MCenter data scientist view
    :param bar_names: Bar graph names
    :param bar_data: Bar graph data.
    :param title_name: Title of the bar Graph
    :return:
    """
    bar_graph_data = BarGraph().name(title_name).cols(
        bar_names.astype(str).tolist()).data(
        bar_data.tolist())
    mlops.set_stat(bar_graph_data)

def export_confusion_table(confmat, algo):
    """
    This function provides the confusion matrix as a table in at MCenter data scientist view
    :param confmat: Confusion matrix
    :param algo: text for the algorithm type
    :return:
    """

    tbl = Table() \
        .name("Confusion Matrix for " + str(algo)) \
        .cols(["Predicted label: " + str(i) for i in range(0, confmat.shape[0])])
    for i in range(confmat.shape[1]):
        tbl.add_row("True Label: " + str(i), [str(confmat[i, j]) for j in range(0, confmat.shape[0])])
    mlops.set_stat(tbl)

def export_feature_importance(final_model, column_names, num_features, title_name):
    """
    This function provides a feature importance at MCenter data scientist view
    :param final_model: Pipeline model (Assume - Feature_Eng + Algo)
    :param column_names: Column names of the input dataframe.
    :param num_features: Number of fefatures to shpw.
    :param title_name: Title of the bar Graph
    :return:
    """
    model_oh = final_model.steps[0][1].features
    trans_feature_names = []
    for mod_el in range(0,len(model_oh)):
        if("OneHotEncoder" in model_oh[mod_el][1].__class__.__name__):
            trans_feature_names += list(model_oh[mod_el][1].get_feature_names([column_names[mod_el]]))
        else:
            trans_feature_names.append(column_names[mod_el])
    trans_feature_names1 = np.asarray(trans_feature_names)
    model_FE_index = np.argsort(final_model.steps[-1][1].feature_importances_)[::-1][:num_features]
    feat_eng = pd.DataFrame({'Name': trans_feature_names1[model_FE_index],
                             'Importance': final_model.steps[-1][1].feature_importances_[model_FE_index]})
    print("Feature Importance for " + str(title_name) + "\n", feat_eng)
    # Output Feature Importance as a BarGraph using MCenter
    export_bar_table(trans_feature_names1[model_FE_index],
                     final_model.steps[-1][1].feature_importances_[model_FE_index],
                     "Feature Importance for " + str(title_name))

def get_psi(caller_ctx, v1, v2, num1=10):
    """
    calculate PSI.

    :param v1: vector 1
    :param v2: vector 2
    :param num1: number of bins
    :return: PSI Value
    """
    rank1 = pd.qcut(v1, num1, labels=False, duplicates="drop") + 1
    num = min(num1, max(rank1))

    basepop1 = pd.DataFrame({'v1': v1, 'rank1': rank1})

    quantiles = basepop1.groupby('rank1').agg({'min', 'max'})
    quantiles.loc[1, 'v1'][0] = 0

    currpop = pd.DataFrame({'v2': v2, 'rank1': [1] * v2.shape[0]})
    for i in range(2, num + 1):
        currpop.loc[currpop['v2'] >= quantiles['v1'].loc[i][0], 'rank1'] = i
        quantiles.loc[i - 1, 'v1'][1] = quantiles.loc[i, 'v1'][0]
    quantiles.loc[num, 'v1'][1] = 1

    basepop2 = basepop1.groupby('rank1').agg({'count'})
    basepop2 = basepop2.rename(columns={'count': 'basenum'})

    currpop2 = currpop.groupby('rank1').agg({'count'})
    currpop2 = currpop2.rename(columns={'count': 'currnum'})

    nbase = basepop1.shape[0]
    ncurr = currpop.shape[0]

    mrged1 = basepop2['v1'].join(currpop2['v2'], how='left')
    if mrged1.shape[0] > 1:
        mrged1.loc[mrged1.currnum.isna(), "currnum"] = 0

    mrged2 = mrged1.join(quantiles['v1'], how='left')

    mrged3 = mrged2
    mrged3['basepct'] = mrged3.basenum / nbase
    mrged3['currpct'] = mrged3.currnum / ncurr

    mrged4 = mrged3
    mrged4['psi'] = (mrged4.currpct - mrged4.basepct) * np.log((mrged4.currpct / mrged4.basepct))

    caller_ctx._logger.info("Merged DF: {}".format(mrged4))

    tot_PSI = sum(mrged4.psi[mrged4.psi != float('inf')])
    final_table = mrged4
    return tot_PSI, final_table