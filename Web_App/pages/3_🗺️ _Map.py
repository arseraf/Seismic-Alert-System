import pandas as pd
import numpy as np
import streamlit as st
import os
import time
import requests
from streamlit_lottie import st_lottie
from google.cloud import bigquery
from geopy.geocoders import ArcGIS
from geopy import distance
import datetime as dt
import folium
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu
from folium.plugins import HeatMap
import branca

#----------------------------------------------------------------------Page Configuration-----------------------------------------------------------------------

st.set_page_config(
    page_title = 'Working towards global standardization of seismological networks and effective communication to the civilian community.',
    page_icon = '🗺',
    layout = 'wide'
)

#----------------------------------------------------------------------Load Data from Data Warehouse------------------------------------------------------------


def load_data(): # Function to retrieve data and transform it

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'seismic-alert-system-cbd570f67095.json'
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



    df = df[df.place.str.contains('Argentina|Ridge|Iran|Bouvet|Indonesia|Vanuatu|Zealand|Tonga|Venezuela|Puerto Rico|MX|Fiji|Kermadec|Italy') == False]

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

    df = df.sort_values(by='time', ascending=False)

    df.dropna(axis=0, inplace=True)

    if any(df.columns.str.contains('index')):
        df.drop('index', axis=1, inplace=True)
    else:

        df.reset_index(inplace=True)

        return (df)

df = load_data()
#time.sleep(1)

#----------------------------------------------------------------------Calculate nearest earthquake-------------------------------------------------------------------

@st.cache(allow_output_mutation=True)
def nearest_earthquake(location1, locations): # function to calculate distances of the user from the earthquakes
    kms = []
    for location in locations:
        kms.append(distance.distance(location1, location))
        if distance.distance(location1, location) == min(kms):
            nearest_location = location
    return min(kms), nearest_location

#----------------------------------------------------------------------Data Processing--------------------------------------------------------------------------------


@st.cache(allow_output_mutation=True)
def preprocess_data_kmeans(df): # Function to preprocess data for the Kmeans model
    from sklearn.preprocessing import StandardScaler
    import joblib

    df_japan = df[df.place.str.contains('Japan')]
    df = df[df.place.str.contains('Japan') == False]

    def correct_depth(depths): # Function to correct the depth field from the Japan API
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
    
    df.depth = df.depth.astype(float).round(1)
    df.mag = df.mag.round(1)


    conditions = [
    (df.depth >= -10 ) & (df.depth < 70),
    (df.depth >= 70 ) & (df.depth < 300),
    (df.depth >= 300)]
    choices = ['shallow','intermediate', 'deep']
    df['depth_class'] = np.select(conditions, choices)

    conditions = [
    (df.mag >= -5 ) & (df.mag < 4.0),
    (df.mag >= 4.0 ) & (df.mag < 5.0),
    (df.mag >= 5.0 ) & (df.mag < 6.0),
    (df.mag >= 6.0 ) & (df.mag < 7.0),
    (df.mag >= 7.0 ) & (df.mag < 8.0),
    (df.mag >= 8.0 )]
    choices = ['minor', 'light', 'moderate', 'strong', 'major', 'great']
    df['mag_class'] = np.select(conditions, choices)

    outlier_depth = df[df.depth_class == '0'].index
    df.drop(outlier_depth, inplace=True)

    df = df[['mag', 'depth', 'depth_class', 'mag_class']]


    def one_hot_encode(df, columns, prefixes): # One hot encoding function
        df = df.copy()
        for column, prefix in zip(columns, prefixes):
            dummies = pd.get_dummies(df[column], prefix=prefix)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)
        return df

    data = one_hot_encode(df, ['depth_class', 'mag_class'], ['d_class', 'm_class'])

    def complete_encoding(df): # Complete encoding function
        encoded_columns = ['d_class_deep','d_class_intermediate', 'd_class_shallow', 'm_class_great', 'm_class_light', 'm_class_major', 'm_class_minor', 'm_class_moderate', 'm_class_strong']
        for column in encoded_columns:
            if column not in df.columns:
                df[column] = 0
        return df

    data = complete_encoding(data)

    #return data

    scaler = StandardScaler()

    X = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    X.dropna(axis=0, inplace=True)

    X.reset_index(inplace=True)

    X.drop('index', axis=1, inplace=True)

    model = joblib.load('KMeans_v9')

    pred = model.predict(X)

    return pred


@st.cache(allow_output_mutation=True)
def proprocess_data_gradient(df): # Function to preprocess data for the Gradient Boost model
    from sklearn.preprocessing import StandardScaler
    import joblib

    df = df.copy()

    df = df[['mag', 'depth']]

    # Scale X

    scaler = StandardScaler()

    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

    model = joblib.load('Gradient_boosting_tsunami_v2')

    pred = model.predict(df)

    return pred

df['category'] = preprocess_data_kmeans(df)

Damage = {1 : 'Mayor', # KMeans labels dictionary
	  0 : 'Moderate',
	  2 : 'Minor'}

df.category.replace(Damage, inplace=True)

if any(df.columns.str.contains('index')):
        df.drop('index', axis=1, inplace=True)
else:
    pass

df['tsunami_prob'] = proprocess_data_gradient(df)

tsunami_dict = {1.0 : 'Likely', # GradientBoost dictionary
                0.0 : 'Unlikely'}

df.tsunami_prob.replace(tsunami_dict, inplace=True)

df.time = pd.to_datetime(df.time)

df['date'] = df.time.dt.date

@st.cache(allow_output_mutation=True)
def safe_distance(df): # Function to calculate safe distance of the user from its nearest earthquake
    if any(df.mag) <= 2.0:
        if kms < 300:
            return 'Close'
        else: return 'Safe Distance'
    if any(df.mag) >= 3.0:
        if kms < 600:
            return 'Close'
        else: return 'Safe Distance'

map_data = df[['latitude', 'longitude', 'mag']]

@st.cache(allow_output_mutation=True)
def load_lottie_url(url : str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading = load_lottie_url('https://assets7.lottiefiles.com/packages/lf20_bnrh8zog.json')


with st.sidebar:
    st_lottie(lottie_loading, loop=True, width=200, height=200)

#----------------------------------------------------------------------Filter map sidebar--------------------------------------------------------------------------------


st.sidebar.header('Map Filters:')


date = st.sidebar.multiselect(
    'Date',
    options=df['date'].unique(),
    default=df['date'].unique()

)

tsunami = st.sidebar.multiselect(
    'Tsunami Probability',
    options=df['tsunami_prob'].unique(),
    default=df['tsunami_prob'].unique()
)

intensity = st.sidebar.multiselect(
    'Intensity',
    options=['Mayor', 'Moderate', 'Minor'],
    default=['Mayor', 'Moderate', 'Minor']
)

df_selection = df.query('date == @date & tsunami_prob == @tsunami & category == @intensity')

#st.dataframe(df_selection)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

most_recent_earthquake = df[df.time == df.time.max()]

#--------------------------------------------------------------------------------Map Popups------------------------------------------------------------------------------

@st.cache(allow_output_mutation=True)
def fancy_html(row): # Popup for every earthquake
    i = row
    
    Date = df_selection['time'].iloc[i]  
    Earthquake_Location = df_selection['place'].iloc[i]
    Distance_From_Earthquake = str(round(float(str(kms).replace('km', '')))) + ' km'                           
    Earthquake_Intensity_Category = df_selection['category'].iloc[i]                           
    Tsunami_Alert = df_selection['tsunami_prob'].iloc[i]  
    Title = 'Earthquake Data'  
    Mag = df_selection['mag'].iloc[i]   
    Depth = df_selection['depth'].iloc[i]                                     

    
    left_col_colour = "#80DCC9"
    right_col_colour = "#F0F2F6"
    
    html = """<!DOCTYPE html>
    <html>

    <head>
    <h4 style="margin-bottom:0"; width="300px">{}</h4>""".format(Title) + """

    </head>
        <table style="height: 90px; width: 350px;">
    <tbody>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Time</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Date) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Earthquake Location</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Earthquake_Location) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Magnitude</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Mag) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Depth</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Depth) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Distance From Earthquake</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Distance_From_Earthquake) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Earthquake Intensity</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Earthquake_Intensity_Category) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Tsunami Alert</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Tsunami_Alert) + """
    </tr>
    </tbody>
    </table>
    </html>
    """
    return html

@st.cache(allow_output_mutation=True)
def fancy_html_n():  # Popup for nearest earthquake
    #i = row
    
    Date = str(nearest_earthquake_data['time'].item()) 
    Earthquake_Location = str(nearest_earthquake_data['place'].item()) 
    Distance_From_Earthquake = dist_label                         
    Earthquake_Intensity_Category = str(nearest_earthquake_data['category'].item())                          
    Tsunami_Alert = str(df.tsunami_prob.iloc[df.category[(df.latitude == nearest_loc[0]) & (df.longitude == nearest_loc[1])].index].item())
    Title = 'Earthquake Data'  
    Mag = str(nearest_earthquake_data['mag'].item()) 
    Depth = str(nearest_earthquake_data['depth'].item())                                  

    
    left_col_colour = "#80DCC9"
    right_col_colour = "#F0F2F6"
    
    html = """<!DOCTYPE html>
    <html>

    <head>
    <h4 style="margin-bottom:0"; width="300px">{}</h4>""".format(Title) + """

    </head>
        <table style="height: 90px; width: 350px;">
    <tbody>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Time</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Date) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Earthquake Location</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Earthquake_Location) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Magnitude</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Mag) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Depth</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Depth) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Distance From Earthquake</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Distance_From_Earthquake) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Earthquake Intensity</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Earthquake_Intensity_Category) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Tsunami Alert</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Tsunami_Alert) + """
    </tr>
    </tbody>
    </table>
    </html>
    """
    return html

@st.cache(allow_output_mutation=True)
def fancy_html_r():  # Popup for most recent earthquake
    #i = row
    
    Date = str(most_recent_earthquake['time'].item()) 
    Earthquake_Location = str(most_recent_earthquake['place'].item()) 
    Distance_From_Earthquake = str(round(float(str(kms).replace('km', '')))) + ' km'                           
    Earthquake_Intensity_Category = str(most_recent_earthquake['category'].item())                          
    Tsunami_Alert = str(most_recent_earthquake.tsunami_prob.item())
    Title = 'Earthquake Data'  
    Mag = str(most_recent_earthquake['mag'].item()) 
    Depth = str(most_recent_earthquake['depth'].item())                                  

    
    left_col_colour = "#80DCC9"
    right_col_colour = "#F0F2F6"
    
    html = """<!DOCTYPE html>
    <html>

    <head>
    <h4 style="margin-bottom:0"; width="300px">{}</h4>""".format(Title) + """

    </head>
        <table style="height: 90px; width: 350px;">
    <tbody>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Time</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Date) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Earthquake Location</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Earthquake_Location) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Magnitude</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Mag) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Depth</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Depth) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Distance From Earthquake</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Distance_From_Earthquake) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Earthquake Intensity</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Earthquake_Intensity_Category) + """
    </tr>
    <tr>
    <td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Tsunami Alert</span></td>
    <td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Tsunami_Alert) + """
    </tr>
    </tbody>
    </table>
    </html>
    """
    return html


#---------------------------------------------------------------------------------Maps-----------------------------------------------------------------------------------


#@st.cache(allow_output_mutation=True)
def display_nearest_map(map_data):  # Earthquake map
    hm = folium.Map(location=[location[0], location[1]], # user location
                tiles='OpenStreetMap',
                zoom_start=5.0, max_zoom=7.0, min_zoom=2.0
                )

    for i in range(0,len(map_data)):
        html = fancy_html(i)
    
        iframe = branca.element.IFrame(html=html,width=400,height=300)
        popup = folium.Popup(iframe,parse_html=True)
        
        folium.Marker([map_data['latitude'].iloc[i],map_data['longitude'].iloc[i]],
                    popup=popup,tooltip=f'<b>Earthquake</b><b><br></b> {map_data.place.iloc[i]}<br>', icon=folium.Icon(color='orange', icon='warning')).add_to(hm)

    for i in range(0,len(map_data)):
            folium.Circle(
            location=[map_data.iloc[i]['latitude'], map_data.iloc[i]['longitude']],
            radius=float(map_data.iloc[i]['mag'])*20000,
            color='crimson',
            fill=True,
            fill_color='crimson'
            ).add_to(hm)


    folium.Marker(location=[location[0], location[1]], tooltip=f'<b>You</b><b><br></b> {location_place}<br>', icon=folium.Icon(color='blue', icon='house')).add_to(hm)
    #folium.Marker(location=[nearest_loc[0], nearest_loc[1]], popup=(f'<b>Place:</b> {nearest_earthquake_data.place.item()}<br><b>Magnitude:</b> {nearest_earthquake_data.mag.item()}<br><b>Depth:</b> {nearest_earthquake_data.depth.item()} <b>Distance From Earthquake:</b> {dist_label} <b>Earthquake Intensity Category:</b> {nearest_earthquake_category} <b>Days Since Ocurred:</b> {time_since_ocurred.item().days} <b>Hours Since Ocurred:</b> {time_since_ocurred.dt.components.hours} <b>Tsunami Alert:</b> {str(df.tsunami_prob.iloc[df.category[(df.latitude == nearest_loc[0]) & (df.longitude == nearest_loc[1])].index].item())}'), tooltip='Nearest Earthquake', icon=folium.Icon(color='red', icon='warning')).add_to(hm)
    html_n = fancy_html_n()
    iframe_n = branca.element.IFrame(html=html_n,width=400,height=300)
    popup_n = folium.Popup(iframe_n,parse_html=True)
    folium.Marker(location=[nearest_loc[0], nearest_loc[1]], popup=popup_n, tooltip=f'<b>Nearest Earthquake</b><b><br></b> {nearest_earthquake_data.place.item()}<br>', icon=folium.Icon(color='red', icon='warning')).add_to(hm)
    html_r = fancy_html_r()
    iframe_r = branca.element.IFrame(html=html_r,width=400,height=300)
    popup_r = folium.Popup(iframe_r,parse_html=True)
    folium.Marker(location=[most_recent_earthquake.latitude, most_recent_earthquake.longitude], popup=popup_r, tooltip=f'<b>Most Recent Earthquake</b><b><br></b> {most_recent_earthquake.place.item()}<br>', icon=folium.Icon(color='green', icon='warning')).add_to(hm)

    
    hm.add_child(folium.LatLngPopup())



    st_hetmap = folium_static(hm, width=1754, height=1080)



def display_heatmap(map_data):  # Heatmap map
    hm = folium.Map(location=[17.482508098310195, -36.750479057838284], #Center of USA
            tiles='stamentoner',
            zoom_start=3.0
            )
    HeatMap(map_data, 
        min_opacity=0.4,
        blur = 18
            ).add_to(folium.FeatureGroup(name='Heat Map').add_to(hm))
    folium.LayerControl().add_to(hm)

    st_hetmap = folium_static(hm, width=1754, height=1080)


#-------------------------------------------------------------------------Horizontal Map Selection-----------------------------------------------------------------------

selected = option_menu(
    menu_title='Select Map',
    menu_icon='map',
    options=['Nearest Earthquake', 'Heatmap Map'],
    icons=['geo-alt', 'geo-alt'],
    orientation='horizontal'
)

if selected == 'Nearest Earthquake':

    location_place = st.text_input('Your location', 'Buenos Aires')

    df_locs = list(zip(df.latitude, df.longitude))

    loc = ArcGIS()
    coordinates = loc.geocode(location_place, out_fields='location')
    location = (coordinates.latitude, coordinates.longitude)

    kms, nearest_loc = nearest_earthquake(location, df_locs)

    nearest_earthquake_category = str(df.category.iloc[df.category[(df.latitude == nearest_loc[0]) & (df.longitude == nearest_loc[1])].index].item())

    current_time = dt.datetime.now()
    time_since_ocurred = current_time - df.time[(df.latitude == nearest_loc[0]) & (df.longitude == nearest_loc[1])]

    nearest_earthquake_data = df[(df.latitude == nearest_loc[0]) & (df.longitude == nearest_loc[1])]

    dist_label = safe_distance(nearest_earthquake_data)



    display_nearest_map(df_selection)

    days = time_since_ocurred.item().days

    tsunami = df.tsunami_prob.iloc[df.category[(df.latitude == nearest_loc[0]) & (df.longitude == nearest_loc[1])].index].item()

    def user_status(dist_label, nearest_earthquake_category, days, tsunami):
        if days >= 5:
            return "Safe, it's very unlikely to experience any aftershock or tsunami alert as of right now"
        elif days < 5:
            if nearest_earthquake_category == 'Mayor':
                if dist_label == 'Close':
                    if tsunami == 'Likely':
                        return 'There is a high chance of a tsunami alert'
                    elif tsunami == 'Unlikely':
                        return 'You might experience a slight aftershock'
                elif dist_label == 'Safe Distance':
                    if tsunami == 'Likely':
                        return 'There is a high chance of a tsunami alert'
            elif nearest_earthquake_category == 'Moderate':
                if dist_label == 'Close':
                    return 'You might experience a slight aftershock'
                elif dist_label == 'Safe Distance':
                    return 'Safe, low probability of an aftershock'
            elif nearest_earthquake_category == 'Minor':
                if dist_label == 'Close':
                    return "Safe, it's very unlikely to experience a slight aftershock"
                elif dist_label == 'Safe Distance':
                    return "Safe, it's very unlikely to experience a slight aftershock"

    user = user_status(dist_label, nearest_earthquake_category, days, tsunami)

    st.markdown('## Status base on location')
    st.write(user)

if selected == 'Heatmap Map':
    display_heatmap(map_data)