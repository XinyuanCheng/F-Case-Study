# Functions for connecting to and interacting with snowflake
# from common.settings import *
import os
from tempfile import NamedTemporaryFile
from pandas import DataFrame, to_datetime
from snowflake import connector
# import snowflake.connector
# Need to add these as environment variables in your bash or zshrc file

# Todo figure out how to make the dev connection global, trickling down from the orchestration through to the _get_connection method
# def get_connection_config():
#     connection_dict = dict(
#                             SNOWFLAKE_USER=os.getenv("SNOWFLAKE_USER"),
#                             SNOWFLAKE_PASSWORD=os.getenv("SNOWFLAKE_PASSWORD"),
#                             SNOWFLAKE_PROD_DB=os.getenv("SNOWFLAKE_PROD_DB"),  # All pipes use this one; just change to dev or prod for now
#                             SNOWFLAKE_DEVELOP_DB=os.getenv("SNOWFLAKE_DEVELOP_DB"),
#                             SNOWFLAKE_WAREHOUSE=os.getenv("SNOWFLAKE_WAREHOUSE"),
#                             SNOWFLAKE_ACCOUNT=os.getenv("SNOWFLAKE_ACCOUNT"),
#     )
#     return connection_dict

def get_connection_config():
    connection_dict = dict(
                            SNOWFLAKE_USER=os.getenv("XCHENG"),
                            SNOWFLAKE_PASSWORD=os.getenv("19960302Cxy_"),
                            SNOWFLAKE_PROD_DB=os.getenv("CASESTUDY"),  # All pipes use this one; just change to dev or prod for now
                            SNOWFLAKE_DEVELOP_DB=os.getenv("CASESTUDY"),
                            SNOWFLAKE_WAREHOUSE=os.getenv("CASESTUDY_WH"),
                            SNOWFLAKE_ACCOUNT=os.getenv("CASESTUDY_XINYUAN"),
    )
    return connection_dict


class Snowflake(object):
    """
    Connect to Snowflake. Fetch SQL results as data frames, or send external data to snowflake.
    to-do: create_table, create_stage, update to_table to create stage if DNE
    """

    def __init__(self):
        self.connection_dict = get_connection_config()
        self.database = self.connection_dict.get("SNOWFLAKE_PROD_DB")
        self.warehouse = self.connection_dict.get("SNOWFLAKE_WAREHOUSE")
        self.account = self.connection_dict.get("SNOWFLAKE_ACCOUNT")


    def __get_connection(self):
        """
        Connects to Snowflake; use in a with statement to ensure connection closes.
        """

        return connector.connect(user=self.connection_dict.get("SNOWFLAKE_USER"),
                                           password=self.connection_dict.get("SNOWFLAKE_PASSWORD"),
                                           account=self.connection_dict.get("SNOWFLAKE_ACCOUNT"),
                                           database=self.connection_dict.get("SNOWFLAKE_PROD_DB"),
                                           warehouse=self.connection_dict.get("SNOWFLAKE_WAREHOUSE"))

    def fetch_sql_df(self, sql: str) -> DataFrame:
        """
        query snowflake and return the result as a dataframe
        """
        with self.__get_connection() as conn:
            with conn.cursor() as curr:
                curr = curr.execute(sql)
                results = curr.fetchall()
                cols = [c[0].lower() for c in curr.description]
                return DataFrame(results, columns=cols)

    def truncate_table(self, schema: str, table: str):
        """
        Truncate the existing table to clear the data
        :return:
        """
        schema_table = schema + '.' + table
        use_schema = f'USE {self.database}.{schema};'
        truncate_table = f'TRUNCATE TABLE IF EXISTS {schema_table};'


        with self.__get_connection() as conn:
            with conn.cursor() as curr:
                # Multiple SQL statements in a single API call are not supported
                curr.execute(use_schema)
                curr.execute(truncate_table)  # execute truncate

    def clear_stage(self, schema: str, stage: str):
        """
        Clear the files present in the staging area
        :return:
        """

        use_schema = f'USE {self.database}.{schema};'  # Set target schema in connection's database
        remove_staged_file = f'REMOVE @{stage};'  # Command to clear the old staged file
        with self.__get_connection() as conn:
            with conn.cursor() as curr:
                # Multiple SQL statements in a single API call are not supported
                curr.execute(use_schema)

                curr.execute(remove_staged_file)  # execute remove unless incremental

    def __to_staging(self, df: DataFrame, stage_schema: str, stage_table: str, incremental: bool = False,
                     staging_suffix: str = None):
        """
        Send a dataframe to an existing staging bucket.
        Set incremental to True to keep the old staging file.
        """
        schema_stage = stage_schema + '.' + stage_table

        use_schema = f'USE {self.database}.{stage_schema};'  # Set target schema in connection's database

        if not incremental:
            remove_staged_file = f'REMOVE @{schema_stage};'  # Command to clear the old staged file

        if staging_suffix:
            suffix = '_' + staging_suffix
        else:
            now = to_datetime('now')
            suffix = now.strftime('_%Y%m%d_%H%M')

        with NamedTemporaryFile(suffix=suffix, mode='r+') as temp:
            df.to_csv(temp.name, index=False)

            put_staged_file = f'PUT file://{temp.name} @{schema_stage};'

            with self.__get_connection() as conn:
                with conn.cursor() as curr:
                    # Multiple SQL statements in a single API call are not supported
                    curr.execute(use_schema)

                    if not incremental:
                        curr.execute(remove_staged_file)  # execute remove unless incremental

                    curr.execute(put_staged_file)

    def __stage_to_table(self, target_schema: str, target_table: str, stage_schema: str, stage_table: str, incremental: bool = False):
        """
        Copy data from an existing staging bucket into a table.
        Set incremental True to append new data rather than overwrite old data.
        """
        target_schema_table = target_schema + '.' + target_table
        stage_schema_table = stage_schema + '.' + stage_table
        # Set target schema
        use_schema = f'USE {self.database}.{target_schema};'

        # truncate rows from table, insert new ones from staging
        if not incremental:
            truncate_table = f'TRUNCATE TABLE IF EXISTS {target_schema_table};'  # Command to clear the old staged file

        copy_into_table = f'COPY INTO {target_schema_table} FROM @{stage_schema_table} FILE_FORMAT = (TYPE = CSV skip_header = 1 EMPTY_FIELD_AS_NULL = TRUE);'  #ON_ERROR = SKIP_FILE ;'
        with self.__get_connection() as conn:
            with conn.cursor() as curr:
                # Multiple SQL statements in a single API call are not supported
                curr.execute(use_schema)

                if not incremental:
                    curr.execute(truncate_table)  # execute truncate unless incremental

                curr.execute(copy_into_table)

    def to_table(self, df: DataFrame, target_schema: str, target_table: str, stage_schema: str, stage_table: str, staging_suffix: str = None,
                 incremental: bool = False):
        """
        Send a dataframe to a staging bucket and then copy it into a table.
        Set incremental to True to keep files currently in the staging bucket and append new data to the table.
        Set staging_suffix to add a suffix to files loaded into staging, else they will be given a temp name.
        """

        self.__to_staging(df, stage_schema, stage_table, incremental, staging_suffix)
        self.__stage_to_table(target_schema, target_table, stage_schema, stage_table, incremental)



