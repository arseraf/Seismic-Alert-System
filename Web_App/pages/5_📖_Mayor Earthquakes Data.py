import streamlit as st
import pandas as pd
from google.cloud import bigquery
import os
import time
from streamlit_lottie import st_lottie
import requests
import plotly_express as px
import plotly.graph_objects as go
import datetime as dt


st.set_page_config(
    page_title = 'Working towards global standardization of seismological networks and effective communication to the civilian community.',
    page_icon = '📖',
    layout = 'wide'
)



@st.cache(allow_output_mutation=True)
def load_data():

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'seismic-alert-system-cbd570f67095.json' # google credentials file
    client = bigquery.Client()

    sql_query = 'SELECT * FROM seismic-alert-system.seismic_datawarehouse.earthquake_damage'

    query_job = client.query(sql_query)

    while query_job.state != 'DONE':
        query_job.reload()
        time.sleep(3)

    if query_job.state == 'DONE':
        df = query_job.to_dataframe()
    else:
        print(query_job.result())

    df.Date = pd.to_datetime(df.Date)

    df = df.sort_values(by='Date', ascending=False)

    df.dropna(axis=0, inplace=True)

    df.drop(columns=['ID'], inplace=True)

    if any(df.columns.str.contains('index')):
        df.drop('index', axis=1, inplace=True)
    else:

        df.reset_index(inplace=True)

        return (df)



df = load_data()

df.Tsunami.replace({1.0 : 'Likely',
                    0.0 : 'Unlikely'}, inplace=True)


@st.cache(allow_output_mutation=True)
def load_lottie_url(url : str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading = load_lottie_url('https://assets7.lottiefiles.com/private_files/lf30_x8aowqs9.json')

lottie_sidebar = load_lottie_url('https://assets10.lottiefiles.com/private_files/lf30_esvwgu42.json')
#st_lottie(lottie_loading, loop=True, width=200, height=200)




page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://static.vecteezy.com/system/resources/previews/004/296/173/original/gray-world-map-isolated-on-white-background-world-map-flat-earth-template-for-web-site-pattern-anual-report-inphographics-vector.jpg");
background-size: 110%;
background-position: center;
background-repeat: no-repeat;
background-attachment: local;
}}"""

st.markdown(page_bg_img, unsafe_allow_html=True)

titles_column, filler_colum,  gif_column = st.columns(3)
with titles_column:
    st.title("Mayor Earthquakes Data")
    st.write('## From US, Japan and Chile')
with filler_colum:
    st.write('')
with gif_column:
    st_lottie(lottie_loading, loop=True, width=200, height=200)


#st.title("Mayor Earthquakes Data")
#st.write('## From US, Japan and Chile')

st.markdown('-----')

with st.sidebar:
    st_lottie(lottie_sidebar, loop=True, width=200, height=200)

# Filter side bar

st.sidebar.header('Filter Here:')

country = st.sidebar.multiselect(
    'Select the Country',
    options=df['Country'].unique(),
    default=df['Country'].unique()
)


year = st.sidebar.multiselect(
    'Year',
    options=df['Year'].unique(),
    default=df['Year'].unique()
)

tsunami = st.sidebar.multiselect(
    'Tsunami',
    options=df['Tsunami'].unique(),
    default=df['Tsunami'].unique()
)

df_selection = df[['Year', 'Date', 'Country', 'Location', 'Latitude', 'Longitude',
       'Focal_Depth', 'Class_Depth', 'Primary_Magnitude', 'Class_Mag',
       'Intensity', 'Death_Description', 'Deaths', 'Damage__in_M__',
       'Damage_Description', 'Tsunami', 'Houses_Affected_Description',
       'Houses_Affected']].query(
    'Country == @country & Year == @year & Tsunami == @tsunami'
)



# TOP KPIs

total_costs = float(df_selection.Damage__in_M__.sum())
average_mag = int(df_selection.Primary_Magnitude.mean())
deaths_total = int(df_selection.Deaths.sum())

left_column, middle_column, rigth_column = st.columns(3)
with left_column:
    st.subheader('Total Costs:')
    st.subheader(f'${total_costs} M.')
with middle_column:
    st.subheader('Average Mag:')
    st.subheader(f'{average_mag}')
with rigth_column:
    st.subheader('Total Deaths:')
    st.subheader(f'{deaths_total}')

st.markdown('----')

total_tsunamis_by_country = (
    df_selection.groupby(by=['Country']).count()[['Tsunami']]
    )


fig_tsunamis_pie = px.pie(
    total_tsunamis_by_country, values='Tsunami', names=total_tsunamis_by_country.index,
    color=total_tsunamis_by_country.index,
    title='<b>Total Tsunamis by Country</b>',
    color_discrete_map={'USA':'darkblue',
                                 'CHILE':'cyan',
                                 'JAPAN':'royalblue'}
)
fig_tsunamis_pie.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=(dict(showgrid=False))
)

houses_affected = (
    df_selection.groupby(by=['Country']).sum()[['Houses_Affected']]
)


fig_houses_bar = px.bar(
    houses_affected,
    x='Houses_Affected',
    y=houses_affected.index,
    orientation='h',
    title='<b>Total Houses Affected by Country</b>',
    color_discrete_sequence=['#80DCC9'] * len(houses_affected),
    template='plotly_white'
)
fig_houses_bar.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=(dict(showgrid=False))
)


bar_column, pie_column = st.columns(2)
with bar_column:
    st.plotly_chart(fig_houses_bar)
with pie_column:
    st.plotly_chart(fig_tsunamis_pie)

total_costs_by_country = (
    df_selection.groupby(['Country','Year']).agg({'Damage__in_M__':'sum'}).reset_index()
)

total_costs_by_country.sort_values(by='Year', inplace=True)
total_costs_by_country.reset_index(inplace=True)

costs_line = px.line(
    total_costs_by_country, x='Year', y='Damage__in_M__',
    color='Country'
)




#st.dataframe(total_costs_by_country)

#fig = go.Figure()
#for c in df_selection['Country'].unique()[:3]:
#    dfp = df_selection[df_selection['Country']==c].pivot(index='Year', columns='Country', values='Damage__in_M__') 
#    fig.add_traces(go.Scatter(x=dfp.index, y=dfp[c], mode='lines', name = c))
#costs_line.add_trace(px.line(
#    df_s
#))

#st.plotly_chart(costs_line, theme='streamlit')
#st.plotly_chart(fig, theme='streamlit')


#st.dataframe(df[['Year', 'Date', 'Country', 'Location', 'Latitude', 'Longitude',
#       'Focal_Depth', 'Class_Depth', 'Primary_Magnitude', 'Class_Mag',
#       'Intensity', 'Death_Description', 'Deaths', 'Damage__in_M__',
#       'Damage_Description', 'Tsunami', 'Houses_Affected_Description',
#       'Houses_Affected']], width=1500)

st.markdown('-----')

st.markdown('## Detailed Data')

st.dataframe(df_selection)

#col2, col3, col4, col5 = st.columns((1, 1, 1, 1))


#col2.header("Distance From Earthquake")
#col2.metric(label='Total Earthquakes:', value=len(df), delta=None)
#col3.header("Earthquake Intensity Category")
#col3.metric(label='Total Earthquakes in US', value=f'%{us_percentage}')
#col4.header("Days Since Ocurred")
#col4.metric(label='Total Earthquakes in Japan', value=f'%{japan_percentage}')
#col5.header("Hours Since Ocurred")
#col5.metric(label='Total Earthquakes in Chile', value=f'%{chile_percentage}')