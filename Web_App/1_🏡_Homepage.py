import streamlit as st
import time
from streamlit_lottie import st_lottie
import requests

st.set_page_config(
    page_title = 'Working towards global standardization of seismological networks and effective communication to the civilian community.',
    page_icon = '🌊',
    layout = 'centered'
)


def load_lottie_url(url : str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading = load_lottie_url('https://assets8.lottiefiles.com/packages/lf20_hzwndued.json')


#st_lottie(lottie_loading, loop=False)


st.sidebar.success('Select a page above')
with st.sidebar:
    st_lottie(lottie_loading, loop=True, width=200, height=200)


st.markdown("<h1 style='text-align: center; color: black;'>Working towards global standardization of seismological networks and effective communication to the civilian community.</h1>", unsafe_allow_html=True)
#st.image("https://user-images.githubusercontent.com/107011436/205526521-f9056409-4798-449e-8961-d202f21a1215.png")
st.markdown('-----')
st.markdown("<h2 style='text-align: center; color: black;'>Latest Seismic data and predictions for the US, Japan and Chile </h2>", unsafe_allow_html=True)


#filler_column, gif_column = st.columns(2)
#with filler_column:
#    st.write('')
#with gif_column:
    #st_lottie(lottie_loading, loop=True, width=200, height=200)


#with st.sidebar:
#    st_lottie(lottie_loading, loop=True, width=200, height=200)


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://user-images.githubusercontent.com/107011436/205959719-8cf04b75-00b6-4a7d-98cd-6cc8e5bd7b62.png");
background-size: 100%;
background-position: center;
background-repeat: no-repeat;
background-attachment: local;
}}"""

st.markdown(page_bg_img, unsafe_allow_html=True)



time.sleep(1)