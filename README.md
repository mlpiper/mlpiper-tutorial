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
  "file\_to\_dataframe" to load file from user provided path to dataframe and
  "dataframe\_to\_file" to save the dataframe to a file on the user provided path.

  In order to run the pipeline, download the preprocessed LendingClub dataset
  to /tmp/loan.csv

  > wget -O /tmp/loan.csv https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/lending-club/loan.csv
  > ./run02.sh

* Chapter-3 (run03.sh)
  The Pipeline component "remove\_nan" is added to perform simple Feature Engineering
  or removing NaNs from the dataset "/tmp/loan.csv". The pipeline loads the file in
  memory as a dataframe, the pandas dataframe is used to perform "dropna()" operation
  before the dataframe is persisted to the file-path provided by user.

  > ./run03.sh

* Chapter-4 (run04.sh)
  Pipeline component "dataframe\_to\_db" is used to save contents to the database(MySql)
  The ML App, loads the file in memory as a dataframe, the pandas dataframe is used to
  write the dataset to the Database using the DB input parameter provided.

  This example requires some additional setup.  Within your MLPiper Python virtual environment, be
  sure to pip install the following packages:
  * cryptography
  * pymysql
  * sqlalchemy
  
  > pip install flask-sqlalchemy pymysql cryptography

  In addtion, it is helpful to have access to a MySQL instance.  A MySQL Docker container can be found at
   https://hub.docker.com/_/mysql

  Launch the MySQL instance with the following command:

  > docker run --name <instance-name> -e MYSQL\_ROOT\_PASSWORD=<my-root-pw> -d mysql:latest

  Access the command line of the container in order to configure the database:

  > docker exec -it <instance-name> bash

  At the command line, access the MySQL command line as the root user:

  > mysql -uroot -p

  Create the user and the database that will be used for this example:

  > CREATE USER 'pmuser'@'%' IDENTIFIED BY 'P3M3Admin!';
  > GRANT ALL PRIVILEGES ON * . * TO 'pmuser'@'%';
  > CREATE DATABASE dataset;

  Exit out of the MySQL command line and the Docker container.

  The pipeline creates a table, (dloan in database: dataset) or any other table name
  based on the parameter provided

  > ./run04.sh

* Chapter-5 (run05.sh)
  The Pipeline components "remove\_nan", "dataframe\_to\_db" "dataframe\_to\_file" are used
  to perform feature engineering on the dataset followed by writing the dataset to the
  Database and the dataset is written to a file. The ML App, loads the file in memory as
  a dataframe, the pandas dataframe is used to perform "dropna()" operation before the
  pandas dataframe is used to write both to the DataBase and a file.

  This example uses the same setup as in Chapter-4.

  The pipeline creates a table, (dloan in database: dataset) or any other table name
  based on the parameter provided

  > ./run05.sh

* Chapter-6 (run06.sh)
  Similar to the pipeline in Chapter-5, reusing the "file_to_dataframe", "remove\_nan",
  and "dataframe\_to\_db" along with "inf_dataset" to generate the required inference 
  dataset.

  > ./run06.sh

* Chapter-7 (run07.sh)
  Pipeline component "db\_to\_dataframe" is used to load contents from the database(MySql)
  to the Pandas dataframe, the dataframe is then saved to a file.

  Required: RUN Chapter-4 OR Chapter-5 before running the Chapter-6 pipeline

  > ./run07.sh

* Chapter-8 (run08.sh)
  The Pipeline components "db_to_dataframe", and "XGBoostTrain" are used to perform XGBoost
  Training using the dataset read from the Database to the Pandas dataframe. The dataframe
  is used by the "XGBoostTrain" component  to train a XGBoost model. During the course of
  training the required model, various charateristics are recorded (using mlops API) and
  the resulting model is exported/saved as a pickle file.
  
  This example requires some additional setup.  Within your MLPiper Python virtual environment, be
  sure to pip install the following packages:
  * xgboost
  * sklearn_pandas
  
  > pip install xgboost sklearn_pandas

  Required: RUN Chapter-5 before running Chapter-8 pipeline

  > ./run08.sh

* Chapter-9 (run09.sh)
  The Pipeline components "db_to_dataframe", and "XGBoostPredict" are used for predictions
  using the models produced by XGBoostTrain pipeline (run08.sh). The prediction is performed
  on the dataset read from the Database to the Pandas dataframe. The model pickle file is
  read and loaded to perform predictions on the dataframe being passed, result of predictions
  along with the class prediction probabilites is returned as dataframe, passed down to 
  component "dataframe_to_db" to persist the prediction to the Database.

  This example requires some additional setup.  Within your MLPiper Python virtual environment,
  be sure to pip install the following packages:
  * sqlalchemy
  * pymysql
  * sklearn_pandas
  * xgboost

  > pip install sqlalchemy xgboost sklearn_pandas pymysql

  Required: RUN Chapter-5 and Chapter-8 before running Chapter-9 pipeline

  > ./run09.sh
