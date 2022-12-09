import streamlit as st
from streamlit_lottie import st_lottie
import requests


st.set_page_config(
    page_title = 'Working towards global standardization of seismological networks and effective communication to the civilian community.',
    page_icon = '🔎',
    layout = 'wide'
)

st.title('Earthquakes Analysis in Chile')
st.markdown('-----')

def load_lottie_url(url : str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading = load_lottie_url('https://assets3.lottiefiles.com/packages/lf20_acryqbdv.json')

with st.sidebar:
    st_lottie(lottie_loading, loop=True, width=200, height=200)


st.markdown("""<iframe title="Dashboard_EQ_EARTHSOLUTIONS" width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiMzY5YmY4YTAtODFmNi00ZTkwLTg4NWMtMDNiNmU0YmZjYTUxIiwidCI6IjBlMGNiMDYwLTA5YWQtNDlmNS1hMDA1LTY4YjliNDlhYTFmNiIsImMiOjR9&pageName=ReportSection1906f259d08088731104" frameborder="0" allowFullScreen="true"></iframe>""", unsafe_allow_html=True)
