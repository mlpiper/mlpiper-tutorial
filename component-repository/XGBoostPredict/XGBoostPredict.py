from __future__ import print_function

import argparse
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.datasets import make_classification

from parallelm.mlops import StatCategory as st
from parallelm.mlops import mlops as mlops
from parallelm.mlops.predefined_stats import PredefinedStats
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.table import Table
from parallelm.components import ConnectableComponent

class XGBoostPredict(ConnectableComponent):

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        model_file = self._prep_and_infer(parent_data_objs[0])
        self._logger.info("Model file saved: {}".format(model_file))
        return [model_file]

    def _prep_and_infer(self, df_dataset):
        # Get number of features 
        self.num_features = df_dataset.shape[1]
        # Get number of samples
        self.num_samples = df_dataset.shape[0]
        #get input model
        self.input_model = self._params["input-model"]

        self._logger.info("PM: Configuration:")
        self._logger.info("PM: # Sample:                    [{}]".format(self.num_samples))
        self._logger.info("PM: # Features:                  [{}]".format(self.num_features))
        self._logger.info("PM: # Input-Model:               [{}]".format(self.input_model))

        # Initialize MLOps Library
        mlops.init()
        # Load the model
        if self.input_model is not None:
            try:
                filename = self._params["input-model"]
                model_file_obj = open(filename, 'rb')
                mlops.set_stat("# Model Files Used", 1)
            except Exception as e:
                #self._logger.error("Model Not Found")
                self._logger.error("Got Exception: {}".format(e))
                mlops.set_stat("# Model Files Used", 0)
                mlops.done()
                return 0

        final_model = pickle.load(model_file_obj)
        features = df_dataset

        # Output Health Statistics to MCenter
        # MLOps API to report the distribution statistics of each feature in the data
        # and compare it automatically with the ones
        mlops.set_data_distribution_stat(features)

        # Output the number of samples being processed using MCenter
        mlops.set_stat(PredefinedStats.PREDICTIONS_COUNT, len(features), st.TIME_SERIES)

        # Accuracy for the chosen model
        pred_labels = final_model.predict(features)
        pred_probs = final_model.predict_proba(features)

        self._logger.info("Pred Labels: {}".format(pred_labels))  # Remove printout can be huge
        self._logger.info("Pred Probabilities: {}".format(pred_probs))  # Remove printout can be huge

        # Pred Label distribution
        pred_value, pred_counts = np.unique(pred_labels, return_counts=True)
        pred_label_distribution = np.asarray((pred_value, pred_counts)).T
        # pred_column_names = pred_value.astype(str).tolist()
        self._logger.info("Pred Label distributions: \n {}".format(pred_label_distribution))

        # Output Pred label distribution as a BarGraph using MCenter
        pred_bar = BarGraph().name("Pred Label Distribution").cols(
            (pred_label_distribution[:, 0]).astype(str).tolist()).data(
            (pred_label_distribution[:, 1]).tolist())
        mlops.set_stat(pred_bar)

        # Pred Label confidence per label
        label_number = len(pred_counts)
        average_confidence = np.zeros(label_number)
        max_pred_probs = pred_probs.max(axis=1)
        for i in range(0, label_number):
            index_class = np.where(pred_labels == i)[0]
            self._logger.info("np.sum(confidence[index_class]) {}".format(np.sum(max_pred_probs[index_class])))
            self._logger.info("counts_elements[i] {}".format(pred_counts[i]))
            if pred_counts[i] > 0:
                average_confidence[i] = np.sum(max_pred_probs[index_class]) / (float(pred_counts[i]))
            else:
                average_confidence[i] = 0

        # BarGraph showing confidence per class
        pred_values1 = [str(i) for i in pred_value]
        bar = BarGraph().name("Average Confidence Per Class").cols(pred_values1).data(average_confidence.tolist())
        mlops.set_stat(bar)
        # Terminate MLOPs
        mlops.done()

        df_result = pd.concat([df_dataset,
            pd.DataFrame({'predict':pred_labels}),
            pd.DataFrame({'probs-0':pred_probs[:,0], 'probs-1':pred_probs[:,1]})],
            axis=1)

        df_result.insert(0,
                'idx',
                [x  for x in range(1, df_result.shape[0] + 1)],
                allow_duplicates=False)
        
        return df_result

