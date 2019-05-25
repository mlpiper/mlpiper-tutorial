from __future__ import print_function

import argparse
import sys
import time
import os
import pandas
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn import preprocessing
from parallelm.components import ConnectableComponent
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.multi_line_graph import MultiLineGraph
from parallelm.mlops.predefined_stats import PredefinedStats


class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for the do_label_encoding
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        # Initialize MLOps Library
        mlops.init()
        df_data = parent_data_objs[0]
        df_clean = do_nan_removal(df_data)
        get_data_distribution_stat(df_clean)
        # Terminate MLOPs
        mlops.done()
        return[df_clean]


def do_nan_removal(df_data):
    """
    Cleanup: (Feature Engineering)
        a) simple label encoding, convert string to real values
        b) remove NaN's drop rows with NaN
        c) using mlops APIs get Data Distribution stats
    """
    #for column in df_data.columns:
    #    if df_data[column].dtype == type(object):
    #        le = preprocessing.LabelEncoder()
    #        df_data[column] = le.fit_transform(df_data[column])

    df_data = df_data.dropna()
    
   
    return df_data

def get_data_distribution_stat(df_clean):
    """
    Record the data distribution stats for the DataFrame
    """
    mlops.set_data_distribution_stat(df_clean)
    
