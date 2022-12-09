import streamlit as st
import pandas as pd
from google.cloud import bigquery
import os
import time
from streamlit_lottie import st_lottie
import requests

st.set_page_config(
    page_title = 'Working towards global standardization of seismological networks and effective communication to the civilian community.',
    page_icon = '🌊',
    layout = 'wide'
)




def load_data():

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'seismic-alert-system-cbd570f67095.json' # Google credentials file
    client = bigquery.Client()

    sql_query = 'SELECT * FROM seismic-alert-system.seismic_datawarehouse.earthquake_deduped WHERE time >= DATE_ADD(CURRENT_DATE(), INTERVAL -5 DAY)'

    query_job = client.query(sql_query)

    while query_job.state != 'DONE':
        query_job.reload()
        time.sleep(3)

    if query_job.state == 'DONE':
        df = query_job.to_dataframe()
    else:
        print(query_job.result())

    df.drop_duplicates(inplace=True)

    df = df[df.place.str.contains('Islands|Tonga|Indonesia|Philippines|Cyprus|Guinea|Rico|Colombia|Venezuela|Iran|Vanuatu|Ridge') == False]

    df = df.sort_values(by='time', ascending=False)

    df.dropna(axis=0, inplace=True)

    df.depth = df.depth.round(1)
    df.mag = df.mag.round(1)

    if any(df.columns.str.contains('index')):
        df.drop('index', axis=1, inplace=True)
    else:

        df.reset_index(inplace=True)

        return (df)



df = load_data()



df_japan = df[df.place.str.contains('Japan')]
df = df[df.place.str.contains('Japan') == False]

def correct_depth(depths):
    corrected_depths = []
    for e in depths:
        if len(str(e)) >= 8:
            corrected_depth = str(e)[:3]
            corrected_depths.append(corrected_depth)
        else:
            corrected_depth = str(e)[:2]
            corrected_depths.append(corrected_depth)
    return corrected_depths

depths = df_japan.depth.to_list()

corrected_depths = correct_depth(depths)

df_japan.depth = corrected_depths

df = df.append(df_japan, ignore_index=True)

df.depth = df.depth.astype(float)





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

st.title("Last 5 days earthquakes")

st.markdown('-----')

st.dataframe(df[['time', 'place', 'mag', 'magType', 'depth', 'latitude', 'longitude']], width=1500)



us_percentage = int(len(df[df.place.str.contains('Japan|Chile') == False])*100/len(df))
japan_percentage = int(len(df[df.place.str.contains('Japan')])*100/len(df))
chile_percentage = int(len(df[df.place.str.contains('Chile')]))



st.markdown('-----')

col2, col3, col4, col5 = st.columns((1, 1, 1, 1))



col2.metric(label='Total Earthquakes:', value=len(df), delta=None)

col3.metric(label='Total Earthquakes in US', value=f'%{us_percentage}')

col4.metric(label='Total Earthquakes in Japan', value=f'%{japan_percentage}')

col5.metric(label='Total Earthquakes in Chile', value=f'%{chile_percentage}')


def load_lottie_url(url : str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading = load_lottie_url('https://assets2.lottiefiles.com/packages/lf20_zn4imzfv.json')


with st.sidebar:
    st_lottie(lottie_loading, loop=True, width=200, height=200)