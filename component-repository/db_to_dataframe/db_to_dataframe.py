from __future__ import print_function

import argparse
import sys
import time
import os
import pandas
from pandas.io import sql
from sqlalchemy import create_engine

from parallelm.components import ConnectableComponent
from parallelm.mlops import mlops as mlops

SELECT_STATEMENT = "SELECT * FROM"

class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for df_to_db
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _get_db_connection(self):
        return(create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                       		.format(user = self._params["user"],
                               	pw=self._params["password"],
				host=self._params["host"],
                                db=self._params["db"])))

    def _materialize(self, parent_data_objs, user_data):
        engine = self._get_db_connection()
        db_table = self._params["table"]
        df_data, record_count = db_to_df(engine, db_table)
        return[df_data]


def db_to_df(engine, table):
    """
    Save DataFrame to Database
    """
    mlops.init()
    df_sink = pandas.read_sql("{} {}".format(SELECT_STATEMENT, table), con = engine)
    mlops.set_stat(table, df_sink.shape[0])
    mlops.done()
    return(df_sink, df_sink.shape[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default='localhost', help="MySql hostname")
    parser.add_argument("--user", default='pmuser', help="User nam")
    parser.add_argument("--password", default='password', help="MySql access password")
    parser.add_argument("--db", default='dataset', help="Database to use")
    parser.add_argument("--table", default='dloan', help="Database Table to use")
    options = parser.parse_args()
    return options

