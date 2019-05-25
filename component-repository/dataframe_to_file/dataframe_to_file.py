from __future__ import print_function

import argparse
import sys
import time
import os
import pandas

from parallelm.components import ConnectableComponent
from parallelm.mlops import mlops as mlops

class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for df_to_file 
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        results_path = self._params["sinkfilename"]
        df_results = parent_data_objs[0]
        return [df_to_file(df_results, results_path)]


def df_to_file(df_predict_results, filepath):
    """
    Save DataFrame to file
    """
    prog_start_time = time.time()
    mlops.init()
    suffix_time_stamp = str(int(time.time()))
    save_file = str(filepath) + '.' + suffix_time_stamp
    sfile = open(save_file, 'w+')
    pandas.DataFrame(df_predict_results).to_csv(save_file)
    sfile.close()
    mlops.done()
    return save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sinkfilepath", default='/tmp/results', help="Save DataFrame to file")
    options = parser.parse_args()
    return options

