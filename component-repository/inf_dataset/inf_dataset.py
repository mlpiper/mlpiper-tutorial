from __future__ import print_function

import argparse
import sys
import time
import os
import pandas
import numpy as np

from parallelm.components import ConnectableComponent
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.multi_line_graph import MultiLineGraph
from parallelm.mlops.predefined_stats import PredefinedStats


class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for get_inf_dataset
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _gen_inf_dataset(self, df_data):
        """
        Split dataset: (Feature Engineering)
           a) drop labels column from dataset 
           b) split the datset to user provided size
           return the generated Inference dataset 
        """
        # Drop the Label-column
        label_col = self._params["label_column"]
        df_data = df_data.drop(label_col, axis=1)

        # Splitting the data to train and test sets:
        infer_data = np.split(df_data, [int(self._params["infer_data_split"]) * len(df_data)])
        return infer_data

    def _materialize(self, parent_data_objs, user_data):
        # Initialize MLOps Library
        mlops.init()
        df_infer_set = self._gen_inf_dataset(parent_data_objs[0])

    	#Record the data distribution stats for the DataFrame
        mlops.set_data_distribution_stat(df_infer_set)

        # Terminate MLOPs
        mlops.done()
        return[df_infer_set]

