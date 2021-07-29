# from load_data_from_snowflake import *
import snowflake.connector
# ## Basic setup and app layout
# st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called
# st.title("Which forecast is better: ours, or the market operators?")

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache)")

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)

# import streamlit as st
# import pandas as pd
# from load_data_from_snowflake import *
# import datetime

# from datetime import datetime

# df = load_data()
# df.sort_values('startTime')
# pd.to_datetime(df['startTime'])
# min_time = df['startTime'].min()
# # max_time = df['startTime'].max()

# start_time = st.slider(
#      "When do you start?",
#      value=min_time)
# st.write("Start time:", start_time)

# # @st.cache

import streamlit as st

# st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called
import datetime as dt
import pandas as pd
import plotly_express as px
# from dateutil.relativedelta import relativedelta # to add days or years
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar



st.title("Which forecast is better: ours, or the market operators?")

@st.cache(allow_output_mutation=True)
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

df = load_data()
df.sort_values('startTime')

st.markdown('This app is used to compared the performance of forecasting of electricity price between our performance and the market forecast performance.')

st.subheader('Part I: By Criteria')


criteria1 = ['Default', 'Weekday', 'Weekend', 'Holiday']
criteria2 = ['R2', 'RMSE', 'MAE']

choose1 = st.selectbox("Pick a criteria", criteria1)
choose2 = st.selectbox("Pick an evaluation metric", criteria2)


# if st.sidebar.button('Get metrics performance'):
if choose1 == 'Default':
    if choose2 == 'R2':
        r2_1 = r2_score(df['actualValue'], df['ourForecast'])
        r2_2 = r2_score(df['actualValue'], df['theirForecast'])
        data = [[r2_1, r2_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [r2_1, r2_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'R2 score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('r2 score of our forecast:', r2_1)
        st.write('r2 score of their forecast:', r2_2)
        if r2_1 > r2_2:
            st.write('We have a better forecast!')
        elif r2_1 < r2_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')
    if choose2 == 'RMSE':
        rmse_1 = mean_squared_error(df['actualValue'], df['ourForecast'], squared=False)
        rmse_2 = mean_squared_error(df['actualValue'], df['theirForecast'], squared=False)
        data = [[rmse_1, rmse_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [rmse_1, rmse_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'RMSE score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('rmse score of our forecast:', rmse_1)
        st.write('rmse score of their forecast:', rmse_2)
        if rmse_1 < rmse_2:
            st.write('We have a better forecast!')
        elif rmse_1 > rmse_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')
    if choose2 == 'MAE':
        mae_1 = mean_absolute_error(df['actualValue'], df['ourForecast'])
        mae_2 = mean_absolute_error(df['actualValue'], df['theirForecast'])
        data = [[mae_1, mae_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [mae_1, mae_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'MAE score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('mae score of our forecast:', mae_1)
        st.write('mae score of their forecast:', mae_2)
        if mae_1 < mae_2:
            st.write('We have a better forecast!')
        elif mae_1 > mae_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')

if choose1 == 'Weekday':
    if choose2 == 'R2':
        r2_1 = r2_score(df[df['startTime'].dt.dayofweek<5]['actualValue'], df[df['startTime'].dt.dayofweek<5]['ourForecast'])
        r2_2 = r2_score(df[df['startTime'].dt.dayofweek<5]['actualValue'], df[df['startTime'].dt.dayofweek<5]['theirForecast'])
        data = [[r2_1, r2_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [r2_1, r2_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'R2 score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('r2 score of our forecast:', r2_1)
        st.write('r2 score of their forecast:', r2_2)
        if r2_1 > r2_2:
            st.write('We have a better forecast!')
        elif r2_1 < r2_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')
    if choose2 == 'RMSE':
        rmse_1 = mean_squared_error(df[df['startTime'].dt.dayofweek<5]['actualValue'], df[df['startTime'].dt.dayofweek<5]['ourForecast'], squared=False)
        rmse_2 = mean_squared_error(df[df['startTime'].dt.dayofweek<5]['actualValue'], df[df['startTime'].dt.dayofweek<5]['theirForecast'], squared=False)
        data = [[rmse_1, rmse_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [rmse_1, rmse_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'RMSE score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('rmse score of our forecast:', rmse_1)
        st.write('rmse score of their forecast:', rmse_2)
        if rmse_1 < rmse_2:
            st.write('We have a better forecast!')
        elif rmse_1 > rmse_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')
    if choose2 == 'MAE':
        mae_1 = mean_absolute_error(df[df['startTime'].dt.dayofweek<5]['actualValue'], df[df['startTime'].dt.dayofweek<5]['ourForecast'])
        mae_2 = mean_absolute_error(df[df['startTime'].dt.dayofweek<5]['actualValue'], df[df['startTime'].dt.dayofweek<5]['theirForecast'])
        data = [[mae_1, mae_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [mae_1, mae_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'MAE score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('mae score of our forecast:', mae_1)
        st.write('mae score of their forecast:', mae_2)
        if mae_1 < mae_2:
            st.write('We have a better forecast!')
        elif mae_1 > mae_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')
if choose1 == 'Weekend':
    if choose2 == 'R2':
        r2_1 = r2_score(df[df['startTime'].dt.dayofweek>4]['actualValue'], df[df['startTime'].dt.dayofweek>4]['ourForecast'])
        r2_2 = r2_score(df[df['startTime'].dt.dayofweek>4]['actualValue'], df[df['startTime'].dt.dayofweek>4]['theirForecast'])
        data = [[r2_1, r2_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [r2_1, r2_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'R2 score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('r2 score of our forecast:', r2_1)
        st.write('r2 score of their forecast:', r2_2)
        if r2_1 > r2_2:
            st.write('We have a better forecast!')
        elif r2_1 < r2_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')
    if choose2 == 'RMSE':
        rmse_1 = mean_squared_error(df[df['startTime'].dt.dayofweek>4]['actualValue'], df[df['startTime'].dt.dayofweek>4]['ourForecast'], squared=False)
        rmse_2 = mean_squared_error(df[df['startTime'].dt.dayofweek>4]['actualValue'], df[df['startTime'].dt.dayofweek>4]['theirForecast'], squared=False)
        data = [[rmse_1, rmse_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [rmse_1, rmse_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'RMSE score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('rmse score of our forecast:', rmse_1)
        st.write('rmse score of their forecast:', rmse_2)
        if rmse_1 < rmse_2:
            st.write('We have a better forecast!')
        elif rmse_1 > rmse_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')
    if choose2 == 'MAE':
        mae_1 = mean_absolute_error(df[df['startTime'].dt.dayofweek>4]['actualValue'], df[df['startTime'].dt.dayofweek>4]['ourForecast'])
        mae_2 = mean_absolute_error(df[df['startTime'].dt.dayofweek>4]['actualValue'], df[df['startTime'].dt.dayofweek>4]['theirForecast'])
        data = [[mae_1, mae_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [mae_1, mae_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'MAE score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('mae score of our forecast:', mae_1)
        st.write('mae score of their forecast:', mae_2)
        if mae_1 < mae_2:
            st.write('We have a better forecast!')
        elif mae_1 > mae_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')

if choose1 == 'Holiday':
    cal = calendar()
    holidays = cal.holidays(start=df['startTime'].min(), end=df['startTime'].max())
    if choose2 == 'R2':
        r2_1 = r2_score(df[df['startTime'].isin(holidays)]['actualValue'], df[df['startTime'].isin(holidays)]['ourForecast'])
        r2_2 = r2_score(df[df['startTime'].isin(holidays)]['actualValue'], df[df['startTime'].isin(holidays)]['theirForecast'])
        data = [[r2_1, r2_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [r2_1, r2_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'R2 score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('r2 score of our forecast:', r2_1)
        st.write('r2 score of their forecast:', r2_2)
        if r2_1 > r2_2:
            st.write('We have a better forecast!')
        elif r2_1 < r2_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')
    if choose2 == 'RMSE':
        rmse_1 = mean_squared_error(df[df['startTime'].isin(holidays)]['actualValue'], df[df['startTime'].isin(holidays)]['ourForecast'], squared=False)
        rmse_2 = mean_squared_error(df[df['startTime'].isin(holidays)]['actualValue'], df[df['startTime'].isin(holidays)]['theirForecast'], squared=False)
        data = [[rmse_1, rmse_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [rmse_1, rmse_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'RMSE score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('rmse score of our forecast:', rmse_1)
        st.write('rmse score of their forecast:', rmse_2)
        if rmse_1 < rmse_2:
            st.write('We have a better forecast!')
        elif rmse_1 > rmse_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')
    if choose2 == 'MAE':
        mae_1 = mean_absolute_error(df[df['startTime'].isin(holidays)]['actualValue'], df[df['startTime'].isin(holidays)]['ourForecast'])
        mae_2 = mean_absolute_error(df[df['startTime'].isin(holidays)]['actualValue'], df[df['startTime'].isin(holidays)]['theirForecast'])
        data = [[mae_1, mae_2]]
        chart_data = pd.DataFrame(data, columns = ['Our Forecast', 'Their Forecast'])
        # plot
        fig = px.bar(x = [mae_1, mae_2], y = ['Our Forecast', 'Their Forecast'], orientation='h')
        fig.update_layout(xaxis_title = 'MAE score', yaxis_title = '')
        st.plotly_chart(fig)
        # print
        st.write('mae score of our forecast:', mae_1)
        st.write('mae score of their forecast:', mae_2)
        if mae_1 < mae_2:
            st.write('We have a better forecast!')
        elif mae_1 > mae_2:
            st.write('They have a better forecast!')
        else:
            st.write('Both forecasts perform equally')


st.subheader('Part II: Define Your Timeframe')

## Range selector
format = 'YYYY-MM-DD hh:mm:ss'  # format output

# pd.to_datetime(df['startTime'])
min_time = df['startTime'].min().to_pydatetime()
start_date = min_time 
# start_date = dt.date(year=2021,month=1,day=1)-relativedelta(years=2)  #  I need some range in the past
max_time = df['startTime'].max().to_pydatetime()
# end_date = dt.datetime.now().date()-relativedelta(years=2)


slider_range = st.slider("Choose timeframe", min_value = min_time, value=[min_time, max_time], format=format)

st.write('Slider range:', slider_range[0], slider_range[1])

# metric values
# R2
our_r2 = r2_score(df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['actualValue'], df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['ourForecast'])
their_r2 = r2_score(df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['actualValue'], df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['theirForecast'])
# RMSE
our_rmse = mean_squared_error(df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['actualValue'], df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['ourForecast'], squared = False)
their_rmse = mean_squared_error(df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['actualValue'], df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['theirForecast'], squared = False)
# MAE
our_mae = mean_absolute_error(df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['actualValue'], df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['ourForecast'])
their_mae = mean_absolute_error(df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['actualValue'], df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['theirForecast'])

# Table formatting
data_table = [[our_r2, their_r2, "We are better" if our_r2 > their_r2 else "They are better"], [our_rmse, their_rmse, "We are better" if our_rmse < their_rmse else "They are better"], [our_mae, their_mae, "We are better" if our_mae < their_mae else "They are better"]]
df_table = pd.DataFrame(data_table, columns = ['Our Forecast', 'Their Forecast', "Who has a better forecast?"], index = ['r2', 'rmse', 'mae'])
st.dataframe(df_table)

# st.write('r2 score of our forecast:', our_r2)
# st.write('r2 score of their forecast:', their_r2)

# if our_r2 > their_r2:
#     st.write('We have a better forecast!')
# elif our_r2 < their_r2:
#     st.write('They have a better forecast!')
# else:
#     st.write('Both forecasts perform equally')


fig = px.line(df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])], x='startTime', y=df.columns[1:4])
st.plotly_chart(fig)

st.subheader('Part III: Forecast Outliers')

st.write("This part lists top 20 of our prediction ourliers based on absolute difference")

df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['our_abs'] = abs(df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['actualValue'] - df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])]['ourForecast'])
# df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])].sort_values('our_abs',ascending=False)
st.dataframe(df[(df['startTime']<=slider_range[1])&(df['startTime']>=slider_range[0])].sort_values('our_abs',ascending=False).head(20))
