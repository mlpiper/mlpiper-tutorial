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


#class MCenterComponentAdapter(ConnectableComponent):
class MCenterStatsComponentAdapter(ConnectableComponent):
    """
    Adapter for get_prediction_stats
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _get_prediction_stats(self, df_data):
        """
        Stats component:
           generate label distribution for the given prediction probabilities
           return: unchanged probability dataset
        """
        
        # Initialize MLOps Library
        mlops.init()
        y = df_data.T.idxmax()

	# Label distribution:
        value, counts = np.unique(y, return_counts=True)
        label_distribution = np.asarray((value, counts)).T
        self._logger.info("Label distributions: Count {}\n values{}".format(counts,
            label_distribution))

        # Output Label distribution as a BarGraph using MCenter
        export_bar_table(label_distribution[:,0],
                label_distribution[:,1],
                "Label Distribution")

        #Record the data distribution stats for the DataFrame
        mlops.set_data_distribution_stat(df_data)

        # Terminate MLOPs
        mlops.done()

        return df_data

    def _materialize(self, parent_data_objs, user_data):
        df_set = self._get_prediction_stats(parent_data_objs[0])
        return[df_set]


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

