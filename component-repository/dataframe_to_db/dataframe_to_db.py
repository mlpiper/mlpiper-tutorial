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

class MCenterDBInsertAdapter(ConnectableComponent):
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

    def _close_connection(connection):
        connection.close()

    def _materialize(self, parent_data_objs, user_data):
        df_dataset = parent_data_objs[0]
        # Type check of incomming parameters
        if not isinstance(df_dataset, pandas.core.frame.DataFrame):
            self._logger.debug("Datatype mismatch got {}".format(type(df_dataset)))
            raise Exception("Datatype mismatch got {}".format(type(df_dataset)))

        self._logger.info(" df_dataset: {}".format(df_dataset))
        engine = self._get_db_connection()
        db_table = self._params["table"]
        db_name = self._params["db"]
        insert_count = self._df_to_db(engine, df_dataset, db_table, db_name)
        return[insert_count]

    def _df_to_db(self, engine, df_sink, table, database):
        """
        Save DataFrame to Database
        """
        mlops.init()
        df_sink.to_sql(con = engine, name = table, if_exists = 'replace', index=False)
        mlops.set_stat(database.join(table), df_sink.shape[0])
        mlops.done()
        return(df_sink.shape[0])

