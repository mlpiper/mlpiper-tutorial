# mlpiper-tutorial
A Tutorial repo for mlpiper pipeline infrastructure

Tutorial, starter guide to designing MCenter components
=======

* Chapter-1 (run01.sh)
  Simple "hello world" MLApp pipeline, it comprises of 2 source/sink components named
  "String-source" and "String-sink" component

  > ./run01.sh

* Chapter-2 (run02.sh)
  Pipeline to load file from local filesystem to in memory DataFrame using the
  Pandas python package. The pipeline comprises of 2 (source and sink) components
  "file_to_dataframe" to load file from user provided path to dataframe and
  "dataframe_to_file" to save the dataframe to a file on the user provided path.

  In order to run the pipeline, download the preprocessed LendingClub dataset
  to /tmp/loan.csv

  > wget -O /tmp/loan.csv https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/lending-club/loan.csv
  > ./run02.sh

* Chapter-3 (run03.sh)
  The Pipeline component "remove_nan" is added to perform simple Feature Engineering
  or removing NaNs from the dataset "/tmp/loan.csv". The pipeline loads the file in
  memory as a dataframe, the pandas dataframe is used to perform "dropna()" operation
  before the dataframe is persisted to the file-path provided by user.

  > ./run03.sh

* Chapter-4 (run04.sh)
  Pipeline component "dataframe_to_db" is used to save contents to the database(MySql)
  The ML App, loads the file in memory as a dataframe, the pandas dataframe is used to
  write the dataset to the Database using the DB input parameter provided.

  > mysql -u $<USER> -p
  > create database <dataset> ;

  The pipeline creates a table, (dloan in database: dataset) or any other table name
  based on the parameter provided

  > ./run04.sh

* Chapter-5 (run05.sh)
  The Pipeline components "remove_nan", "dataframe_to_db" "dataframe_to_file" are used
  to perform feature engineering on the dataset followed by writing the dataset to the
  Database and the dataset is written to a file. The ML App, loads the file in memory as
  a dataframe, the pandas dataframe is used to perform "dropna()" operation before the
  pandas dataframe is used to write both to the DataBase and a file.

  > mysql -u $<USER> -p
  > create database <dataset> ;

  The pipeline creates a table, (dloan in database: dataset) or any other table name
  based on the parameter provided

  > ./run05.sh

* Chapter-6 (run06.sh)
  Pipeline component "db_to_dataframe" is used to load contents from the database(MySql)
  to the Pandas dataframe, the dataframe is then saved to a file.

  Required: RUN Chapter-4 OR Chapter-5 before running Chapter-6 pipeline

  > ./run06.sh

* Chapter-7 (run07.sh)
  The Pipeline components "db_to_dataframe", and "XGBoostTrain" are used to perform XGBoost
  Training using the dataset read from the Database to the Pandas dataframe. The dataframe
  is used by the "XGBoostTrain" component  to train a XGBoost model. During the course of
  training the required model, various charateristics are recorded (using mlops API) and
  the resulting model is exported/saved as a pickle file.

  Required: RUN Chapter-5 before running Chapter-7 pipeline

  > ./run07.sh


