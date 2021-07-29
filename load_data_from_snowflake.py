import pandas as pd
import snowflake.connector

def load_data():
    conn = snowflake.connector.connect(
                user="XCHENG",
                password="19960302Cxy_",
                account="ke74435",
                warehouse="CASESTUDY_WH",
                database="CASESTUDY",
                schema="CASESTUDY_XINYUAN"
                )

    cur = conn.cursor()

    # Execute a statement that will generate a result set.
    sql = '''
    SELECT "CASESTUDY"."CASESTUDY_XINYUAN"."actual_prices"."startTime", "actualValue",
    "CASESTUDY"."CASESTUDY_XINYUAN"."our_forecast"."p50" AS "ourForecast",
    "CASESTUDY"."CASESTUDY_XINYUAN"."their_forecast"."p50" AS "theirForecast"
    FROM "CASESTUDY"."CASESTUDY_XINYUAN"."actual_prices"
    INNER JOIN "CASESTUDY"."CASESTUDY_XINYUAN"."our_forecast" ON "CASESTUDY"."CASESTUDY_XINYUAN"."actual_prices"."startTime" = "CASESTUDY"."CASESTUDY_XINYUAN"."our_forecast"."startTime"
    INNER JOIN "CASESTUDY"."CASESTUDY_XINYUAN"."their_forecast" ON "CASESTUDY"."CASESTUDY_XINYUAN"."actual_prices"."startTime" = "CASESTUDY"."CASESTUDY_XINYUAN"."their_forecast"."startTime"
    '''
    cur.execute(sql)
    # Fetch the result set from the cursor and deliver it as the Pandas 
    df = cur.fetch_pandas_all()
    cur.close()
    conn.close()
    return df
