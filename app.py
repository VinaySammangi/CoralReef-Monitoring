#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vinaysammangi
"""

import streamlit as st
import json
from io import StringIO
from PIL import Image
import ee
import geemap
import pandas as pd
import geopandas as gpd
import plotly.express as px
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from datetime import date
from datetime import datetime
import eemont
from joblib import Parallel, delayed
import subprocess
import sys
import plotly.graph_objects as go
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


# Landsat8 bands and their descriptions
l8_bands_dict = {"SR_B1":"ULTRABLUE","SR_B2":"BLUE","SR_B3":"GREEN","SR_B4":"RED","SR_B5":"NIR","SR_B6":"SIR1","SR_B7":"SIR2",
                 "SR_QA_AEROSOL":"AEROSOL_ATTRIBUTES", "ST_B10":"SURFACE_TEMPERATURE","ST_ATRAN":"ATMOSPHERIC_TRANSMITTANCE",
                 "ST_CDIST":"PIXEL_DISTANCE","ST_DRAD":"DOWNWELLED_RADIANCE","ST_EMIS":"BAND10_EMISSIVITY",
                 "ST_EMSD":"EMISSIVITY_STD", "ST_QA":"SURFACE_TEMPERATURE_UNCERTAINTY", "ST_TRAD":"RADIANCE_THERMALBAND",
                 "ST_URAD":"UPWELLED_RADIANCE", "QA_PIXEL":"CLOUD", "QA_RADSAT":"RADIOMETRIC_SATURATION"}

# Landsat8 spectral indices available in eemont library
spectral_indices = ["AFRI1600","AFRI2100","ARVI","ATSAVI","AVI","AWEInsh","AWEIsh","BAI","BAIM","BCC",
                    "BI","BLFEI","BNDVI","BWDRVI","BaI","CIG","CSI","CSIT","CVI","DBI","DBSI","DVI","DVIplus",
                    "EBBI","EMBI","EVI","EVI2","ExG","ExGR","ExR","FCVI","GARI","GBNDVI","GCC","GDVI","GEMI",
                    "GLI","GNDVI","GOSAVI","GRNDVI","GRVI","GSAVI","GVMI","IAVI","IBI","IKAW","IPVI","LSWI",
                    "MBI","MBWI","MCARI1","MCARI2","MGRVI","MIRBI","MLSWI26","MLSWI27","MNDVI","MNDWI","MNLI",
                    "MRBVI","MSAVI","MSI","MSR","MTVI1","MTVI2","MuWIR","NBLI","NBR","NBR2","NBRT1","NBRT2",
                    "NBRT3","NBSIMS","NBUI","NDBI","NDBaI","NDDI","NDGI","NDGlaI","NDII","NDISIb","NDISIg",
                    "NDISImndwi","NDISIndwi","NDISIr","NDMI","NDPI","NDSI","NDSII","NDSInw","NDSaII",
                    "NDVI","NDVIMNDWI","NDVIT","NDWI","NDWIns","NDYI","NGRDI","NIRv","NIRvH2","NLI",
                    "NMDI","NRFIg","NRFIr","NSDS","NWI","NormG","NormNIR","NormR","OCVI","OSAVI","PISI","RCC",
                    "RDVI","RGBVI","RGRI","RI","S3","SARVI","SAVI","SAVI2","SAVIT","SI","SIPI","SR","SR2","SWI",
                    "SWM","TDVI","TGI","TSAVI","TVI","TriVI","UI","VARI","VI6T","VIG","VgNIRBI","VrNIRBI","WDRVI",
                    "WDVI","WI1","WI2","WRI","kEVI","kIPVI","kNDVI","kRVI","kVARI"]
spectral_bands = list(l8_bands_dict.keys())
l8_dates = {"start_date":"2013-04-01","end_date":"2022-08-01"}

# Data from the downloaded JSON file
json_data = '''
{
  "type": "service_account",
  "project_id": "practicum-project-363013",
  "private_key_id": "9ed5f5c2a9d6f5ab4f495fa438e5a7f9e972fd52",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC2WTXfYD/ezdP1\n1+P6n+dvBkFVptsHv/3KaKbaXBJRtMiItPndzwgdsgjAU8T1VqU+CJc38FWb7G9a\nTWFn8V2VWIJbuUzgEUNyjoMkfIRx3WxmyceqyMOMQxbM9G96MUp5gmUE8lno+o/N\nvFmz9O+Gua48xbR89E5PJgyl+jLEmDr+7OYj/srh0Ja18HvtqrkJHAdoHlNS6CdM\nA7Da+DDPPQfuOM8HOB8ZoTUgewPcXYxC9hbmzQl81NfUibu6NbA6IiAfGcDM5bfh\nrQMwpKbbQqye361NVg6PW91mWBXTnV+xNd0MFXsVygv89kcZ2EnTy/77KOcXnu1v\n5hv8mCjnAgMBAAECggEALRTYI04N7F0Rsp150Qv4cTPoMi9Kxls6ePCvk5ugsc+S\npm2ruqFFHeZWkIoFTyxpNPF1xVAnMiHdk8M+ui5rlxEnRVsF/P13odpG5N3d9rKp\n6q2nLfttkP9DI0+pQdnu0iShKfxqqxVLOS+AM+Px1eqQ/5hXW28g7yN2jBBTvdOC\n+hmHBGx5xoM+x7/SstHMtUX14MTqmcgWzslvUBFyvxUs07ksQuLjnpmz4V3F9NCF\nBOe/L+0BhtDMCeZ8WY0Up4ip/d039Kj/h4W/ErFJGI4pqE+69iC2A+sauvQbpzif\ngywQsPK/F1dg5UFhh4yr9qRCUgO90rkPinrF63uSkQKBgQD6tspwBW9DM9h7fssu\n9fYK9dgiR28oEBNBJ3etNLUIj1rQKlURaolwvAcX9IQwWZKeWNZuib5nhEAxyBQW\nNx607QdEaM7B1OSPQMy8i2SCizRGbJM8m+gsjg/RxyVCcT83zokSWB+1DgpFPuUa\nAhhCM7t8rCA+N9SiujesIGis8QKBgQC6MWwIS0xumOI5/nQcrjI3/BN1KDlfuNC7\n+XTXiSWLAdSNXxHSW0aXJcySV/sKK3NUJltpc/TM5wugCqJfWfimdguwl+X1JqWe\nRGjx2QOBMhprb0HDUlN8kyEjWD4/WTsjpXjqIQdVrMPfaX45NGBoZT7PxZ63EjGy\n6QbK1SeTVwKBgCniwf1nGwiKL9+p9j4ZP4rjOcG4V3zE+sKG2nqodJpCgPSILgAj\n4WRhNXouEquVO2aTBvgesR3QPX1TpO91M/8cHnuyWuCNNcYtGEdjrl4U7Z3aY9rb\nXTWcYk40zCfGjb5AFixnZpy0BMk+0b2/ndfplqgkhZp/b1nkbIqoO3SxAoGBAKMW\nPwZUzjHhf+ZEVvf4LMyU44YvIXISs+KycgGIg3XquH7L0xRqFr61wSY+IgmaXX5L\nyq3nf3kqtygLqIXUjNNheoPHyQiePVsPmMydxVAYzsNjxDqNlcr8JH6NAJkEU6S5\nf9uz6nTEyxyZjpIUqo1GgWoEMy0vppCLRAPOCMgpAoGBAM9yTqMfxpo3K5hWqbI8\naL7prFVK3EVlhRqBrpnjfnqBfN0XEGp6oVCPOhXvVWIF4w4l4ASFyEuivqmICvqd\ngE1ZRvLo1s/lYIKL96W2n/aiwehA0c//Ahb+ongl3dcugnr13xxPKzBO5YRnx07s\nhZ+Q1i4L1xTIF9oqngNzB1Q+\n-----END PRIVATE KEY-----\n",
  "client_email": "earth-engine-resource-viewer@practicum-project-363013.iam.gserviceaccount.com",
  "client_id": "110078860645894310593",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/earth-engine-resource-viewer%40practicum-project-363013.iam.gserviceaccount.com"
}
'''
# Preparing values
json_object = json.loads(json_data, strict=False)
service_account = json_object['client_email']
json_object = json.dumps(json_object)
# Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)

BACKGROUND_COLOR = 'white'
COLOR = 'black'

def v_spacer(height,elem):
    for _ in range(height):
        elem.write('\n')
        
def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = True,
        padding_top: int = 0, padding_right: int = 0.5, padding_left: int = 0.5, padding_bottom: int = 0,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {padding_top}rem;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

im = Image.open("imgs/page_image.png")
st.set_page_config(page_title="Vinay Sammangi - Project Portfolio",page_icon=im,layout="wide")

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
                .css-18e3th9 {
                    padding-top: 1rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
                .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)


st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """, unsafe_allow_html=True)
            
st.markdown("<i><text style='text-align: left; color:orange;'> Please view this website on either desktop or laptop</text></i>",unsafe_allow_html=True)
st.markdown("<h1 width: fit-content; style='text-align: center; color: white; background-color:#0083b8;'>Coral Reef Health Monitoring System</h1>", unsafe_allow_html=True)        
set_page_container_style()

@st.cache(allow_output_mutation=True)
def _load_data():
    data_dict = {}
    data_dict["coral_df"] = pd.read_csv("Output/coral_df.csv")
    final_df = pd.read_pickle('Output/FinalData.pkl')
    final_gdf = gpd.GeoDataFrame(final_df, geometry='geometry')
    final_gdf["long"] = final_gdf.geometry.centroid.x
    final_gdf["lat"] = final_gdf.geometry.centroid.y
    data_dict["final_gdf"] = final_gdf
    data_dict["boundaries_regions"] = gpd.read_file('Output/boundaries_regions.geojson')
    data_dict["global_model"] = joblib.load("Output/global_model.sav")
    data_dict["ModelSummaries"] = df = pd.read_excel("Output/ModelSummaries.xlsx")
    return data_dict

def _data_exploration():
    tab1_1, tab1_2, tab1_3 = st.tabs(["1.1. Coral Databases","1.2. Satellite Instruments","1.3. Data Integration"])
    with tab1_1:
        st.write("**Technical Area 1:** Cross-Validate the Open-Source Reef Databases")
        tab1_1_1, tab1_1_2, tab1_1_3, tab1_1_4, tab1_1_5 = st.tabs(["1.1.1. Allen Coral Atlas","1.1.2. Reef Base","1.1.3. AIMS","1.1.4. GCBD","1.1.5. NOAA NCEI"])
    with tab1_2:
        st.write("**Technical Area 2:** Time-Align and Geo-Align with Corresponding Instrument Data")
        tab1_2_1, tab1_2_2 = st.tabs(["1.2.1. LANDSAT","1.2.2. CALIPSO"])
    with tab1_3:
        _coral_databases()

def _coral_databases():
    st1_31, st1_32 = st.columns([2,1],gap="medium")
    data_file = _load_data()
    coral_df = data_file["coral_df"].copy()
    boundaries_regions = data_file["boundaries_regions"].copy()
    final_gdf = data_file["final_gdf"].copy()
    final_gdf = final_gdf.rename(columns={"Coral_Class":"Class"})
    regions = list(boundaries_regions["name"])
    selected_regions = st1_31.multiselect('Regions to consider',regions,default=regions)
    final_gdf_filtered = final_gdf.loc[(final_gdf["ReefRegion"].isin(selected_regions)),]
    boundaries_regions_filtered = boundaries_regions.loc[boundaries_regions["name"].isin(selected_regions)].reset_index(drop=True)
    layers_ = []
    for i in range(boundaries_regions_filtered.shape[0]):
      main_dict = {}
      temp = boundaries_regions_filtered.iloc[i:(i+1)]
      main_dict["source"] = json.loads(temp.geometry.to_json())
      main_dict["type"] = "line"
      main_dict["below"] = "traces"
      main_dict["color"] = list(temp["color"])[0]
      layers_.append(main_dict)
    
    fig = px.scatter_mapbox(final_gdf_filtered, lat="lat", lon="long", hover_data=["ReefRegion","year", "month","hour"],color = "Class",
                            color_discrete_map={"Coral":"green","NonCoral":"red"}, opacity=0.2)
    fig.update_layout(
        title_text="Region-Wise Coral & NonCoral Data",title_x=0.5,title_y=0.95,
        mapbox = {
            'style': "open-street-map",
            'center': { 'lon': -130, 'lat':-8},
            'zoom': 2, 'layers': layers_},
        legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        )),
        margin = {'l':0, 'r':0, 'b':0, 't':50},height=500
        )
    st1_31.plotly_chart(fig,use_container_width=True)
    
    counts_df = coral_df.groupby("ReefRegion").agg({"ID":"count"}).reset_index()
    counts_df["Cluster"] = counts_df["ReefRegion"].map({"Northern Caribbean - Florida,Bahamas":"Cluster1","Great Barrier Reef and Torres Strait":"Cluster2",
     "Southeastern Caribbean":"Cluster1","Mesoamerica":"Cluster1",
     "Central South Pacific":"Cluster2","Subtropical Eastern Australia":"Cluster2",
     "Coral Sea":"Cluster2","Bermuda":"Cluster1",
     "Eastern Tropical Pacific":"Cluster1"})
    counts_df.columns = ["Region","Coral Count","Cluster"]
    counts_df = counts_df.sort_values("Coral Count",ascending=False)
    st1_32.markdown("<h3 width: fit-content; style='text-align: center; color: gray;'>Region-Wise Coral Data</h3>", unsafe_allow_html=True)
    st1_32.markdown("*Clusters are defined based on the closest regions and the number of data points in each cluster.*")
    gb = GridOptionsBuilder.from_dataframe(counts_df)
    # gb.configure_pagination(paginationAutoPageSize=True)
    # gb.configure_side_bar()
    gridOptions = gb.build()
    
    with st1_32.container():
        AgGrid(counts_df, gridOptions=gridOptions,
                data_return_mode='AS_INPUT',
                update_mode='MODEL_CHANGED',
                fit_columns_on_grid_load=True,
                theme='dark',height=300
                )
    
    
def _real_time_prediction():
    st3_1_1, st3_1_2, st3_1_3, st3_1_4,st3_1_5, st3_1_6 = st.columns([2,2,2,2,5,1])
    lat_ = st3_1_1.number_input('Latitude', -90.0, 90.0,32.270120,format="%.6f",key="lat_3",help='Select a value between -90 and 90')
    long_ = st3_1_2.number_input('Longitude', -180.0, 180.0,-64.776210,format="%.6f",key="long_3",help='Select a value between -180 and 180')
    radius_ = st3_1_3.number_input('Radius', 30, 1000, 100, key="radius_3",help='Select a value between 30 and 1000 meters',step=10)    
    # date_ = st3_1_4.date_input("Date",date.today())
    def start_capture():
        geometry = ee.Geometry.Point([long_, lat_]).buffer(radius_)
        if lat_==32.270120 and long_==-64.776210 and radius_==100:
            landsat8_df = pd.read_pickle("Output/landsat8_df_default.pkl")
        else:
            fc = ee.FeatureCollection([ee.Feature(geometry)])
            bands_ = spectral_bands + spectral_indices
            s_date = date.today()
            e_date = datetime.strptime(l8_dates["start_date"], '%Y-%m-%d')
            year_end = (s_date.year)
            year_start = (e_date.year)
            start_dates, end_dates = [l8_dates["start_date"]], ["2013-12-31"]
            for year in range(year_start+1,year_end+1):
                start_dates.append(str(year)+"-01-01")
                end_dates.append(str(year)+"-12-31")
            
            landsat8_list = Parallel(n_jobs=-1)(delayed(download_landsat8)(fc,start_dates[i],
                                                                          end_dates[i],bands_) 
                                                for i in range(len(start_dates)))
            landsat8_df = pd.concat(landsat8_list).reset_index(drop=True)
            landsat8_df["date"] = pd.to_datetime(landsat8_df["date"])
            landsat8_df = landsat8_df.sort_values("date",ascending=True).reset_index(drop=True)
            landsat8_df = landsat8_df.dropna().reset_index(drop=True)
            # landsat8_df.to_pickle("../Output/landsat8_df_default.pkl")
        return landsat8_df,geometry
    
    def run_cap():
        cap_button = st3_1_6.button("Predict") # Give button a variable name
        if cap_button: # Make button a condition.
            with st.spinner('Wait for it...'):
                landsat8_df,geometry = start_capture()
                data_file = _load_data()
                final_gdf = data_file["final_gdf"].copy()
                st3_2_1, st3_2_2 = st.columns([5,2])

                fig = go.Figure(go.Scattermapbox(
                    mode = "markers",
                    lon = [], lat = [],
                    marker = {'size': 0, 'color': ["cyan"]}))
                temp = (geometry.getInfo()["coordinates"][0])
                x,y = list(np.mean(np.array(temp[:-1]),axis=0))
                fig.update_layout(
                    title_text="Bounding Box",title_x=0.5,title_y=0.95,
                    mapbox = {
                        'style': "open-street-map",
                        'center': { 'lon': x, 'lat':y},
                        'zoom': 12, 'layers': [{
                            'source': geometry.getInfo(),
                            'type': "line", 'below': "traces", 'color': "#0042FF"},
                {'source': json.loads(final_gdf.loc[final_gdf["Coral_Class"]=="Coral",].geometry.to_json()),
                'type': "fill", 'below': "traces", 'color': "green"},
                {'source': json.loads(final_gdf.loc[final_gdf["Coral_Class"]=="NonCoral",].geometry.to_json()),
                'type': "fill", 'below': "traces", 'color': "red"}]},
                        margin = {'l':0, 'r':0, 'b':0, 't':50},height=500
                        )                
                st3_2_2.plotly_chart(fig,use_container_width=True)
                
                global_model = data_file["global_model"]
                x_vars  = list(global_model.feature_names_in_)                
                predict_probs = global_model.predict_proba(landsat8_df[x_vars])              
                predict_probs = (list(predict_probs[:,1]))
                predictions = list(global_model.predict(landsat8_df[x_vars]))
                predictions_text = ["Coral" if i==0 else "NonCoral" for i in predictions]
                dates = list(landsat8_df["date"])
                fig = go.Figure(data=go.Scatter(x=dates,
                                y=predict_probs,
                                mode='markers',
                                marker = {'color': predictions,
                                  'colorscale': [[0, 'red'],[1.0, 'green']],
                                  'size': 7
                                 },
                                hovertemplate =
                    'Date: <b>%{x}</b>'+
                    '<br>Probability: <b>%{y}</b>'+
                    '<br>Class: <b>%{marker.color}</b><br>'+                                
                    '<b>%{text}</b>',text = dates))
                fig.update_layout(margin=dict(l=0,r=0,b=0,t=50))
                fig.update_layout(title_text="Model Predictions Over Time",title_x=0.5,title_y=0.95,
                                 width=1200,height=500,template="plotly_white",font=dict(size=16))
                st3_2_1.plotly_chart(fig,use_container_width=True)
            st3_1_6.success('', icon="✅")
            
    run_cap()

    
def download_landsat8(fc_,start_date,end_date,bands_):
    import eemont
    json_object = json.loads(json_data, strict=False)
    service_account = json_object['client_email']
    json_object = json.dumps(json_object)
    # Authorising the app
    credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
    ee.Initialize(credentials)

    S2 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(fc_)
            .filterDate(start_date,end_date)
            .preprocess()
            .select(spectral_bands)
            .spectralIndices(spectral_indices))

    ts = S2.getTimeSeriesByRegions(reducer = [ee.Reducer.median()],
                                              collection=fc_,
                                              bands = bands_,
                                              scale = 30)
    time_df = geemap.ee_to_pandas(ts)
    time_df_filtered = time_df.loc[time_df["QA_PIXEL"]!=-9999,]
    return time_df_filtered

    
def _model_results():
    tab2_1, tab2_2, tab2_3 = st.tabs(["2.1. Landsat","2.2. Calipso","2.3. Landsat + Calipso"])
    with tab2_1:
        _landsat8_experiments()
    pass

def _landsat8_experiments():
    st.markdown("<h3 width: fit-content; style='text-align: center; color: gray;'>Model Results : Summary</h3>", unsafe_allow_html=True)
    data_file = _load_data()
    model_summaries = data_file["ModelSummaries"].copy()
    gb = GridOptionsBuilder.from_dataframe(model_summaries)
    # gb.configure_pagination(paginationAutoPageSize=True)
    # gb.configure_side_bar()
    gridOptions = gb.build()
    with st.container():
        AgGrid(model_summaries, gridOptions=gridOptions,
                data_return_mode='AS_INPUT',
                update_mode='MODEL_CHANGED',
                fit_columns_on_grid_load=False,
                theme='dark',height=270
                )
    # gb = GridOptionsBuilder.from_dataframe(model_summaries)
    # # gb.configure_pagination(enabled=False)
    # gridOptions = gb.build()
    # grid_table = AgGrid(model_summaries,
    #                     gridOptions=gridOptions,
    #                     fit_columns_on_grid_load=False,
    #                     height=270,
    #                     theme='dark',
    #                     update_mode=GridUpdateMode.GRID_CHANGED,
    #                     reload = True,
    #                     allow_unsafe_jscode=True,
    #             )
        
        # alpine,material
    st.markdown('**Cluster1 Regions:** Northern Caribbean - Florida,Bahamas , Southeastern Caribbean , Mesoamerica , Bermuda , Eastern Tropical Pacific')
    st.markdown('**Cluster2 Regions:** Great Barrier Reef and Torres Strait , Central South Pacific , Subtropical Eastern Australia , Coral Sea')
    
    st.markdown("<h3 width: fit-content; style='text-align: center; color: gray;'>Model Results : Details</h3>", unsafe_allow_html=True)

    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'>1. Model 1</h4>", unsafe_allow_html=True)
    st.markdown('Model trained and predicted on the global dataset using Stratified KFold Cross Validation')
    
    image1 = Image.open('imgs/global_cm.png')
    image2 = Image.open('imgs/global_fi.png')
    image1 = image1.resize((600, 550))
    image2 = image2.resize((600, 550))
    st2_1_1, st2_1_2 = st.columns([2,2],gap="large")
    st2_1_1.image(image1, caption='Classification Report')
    st2_1_2.image(image2, caption='Feature Importance')
    
    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'>2. Model 2</h4>", unsafe_allow_html=True)
    st.markdown('Model trained and predicted on the Cluster 1 regions using Stratified KFold Cross Validation')
    
    image1 = Image.open('imgs/cluster1_cm.png')
    image2 = Image.open('imgs/cluster1_fi.png')
    image1 = image1.resize((600, 550))
    image2 = image2.resize((600, 550))
    st2_1_3, st2_1_4 = st.columns([2,2],gap="large")
    st2_1_3.image(image1, caption='Classification Report')
    st2_1_4.image(image2, caption='Feature Importance')
    
    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'>3. Model 3</h4>", unsafe_allow_html=True)
    st.markdown('Model trained and predicted on the Cluster 2 regions using Stratified KFold Cross Validation')
    
    image1 = Image.open('imgs/cluster2_cm.png')
    image2 = Image.open('imgs/cluster2_fi.png')
    image1 = image1.resize((600, 550))
    image2 = image2.resize((600, 550))
    st2_1_5, st2_1_6 = st.columns([2,2],gap="large")
    st2_1_5.image(image1, caption='Classification Report')
    st2_1_6.image(image2, caption='Feature Importance')

    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'>4. Model 4</h4>", unsafe_allow_html=True)
    st.markdown('Model trained on Cluster 1 regions and predicted on Cluster 2 regions using Train-Test Strategy')
    
    image1 = Image.open('imgs/cluster1_cluster2_cm.png')
    image2 = Image.open('imgs/cluster1_cluster2_fi.png')
    image1 = image1.resize((600, 550))
    image2 = image2.resize((600, 550))
    st2_1_7, st2_1_8 = st.columns([2,2],gap="large")
    st2_1_7.image(image1, caption='Classification Report')
    st2_1_8.image(image2, caption='Feature Importance')

    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'>5. Model 5</h4>", unsafe_allow_html=True)
    st.markdown('Model trained on Cluster 2 regions and predicted on Cluster 1 regions using Train-Test Strategy')
    
    image1 = Image.open('imgs/cluster2_cluster1_cm.png')
    image2 = Image.open('imgs/cluster2_cluster1_fi.png')
    image1 = image1.resize((600, 550))
    image2 = image2.resize((600, 550))
    st2_1_9, st2_1_10 = st.columns([2,2],gap="large")
    st2_1_9.image(image1, caption='Classification Report')
    st2_1_10.image(image2, caption='Feature Importance')

    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'>6. Model 6</h4>", unsafe_allow_html=True)
    st.markdown('Model trained on Northern Caribbean - Florida,Bahamas region and predicted on other Cluster 1 regions using Train-Test Strategy')
    
    image1 = Image.open('imgs/ncfb_c1_cm.png')
    image2 = Image.open('imgs/ncfb_c1_fi.png')
    image1 = image1.resize((600, 550))
    image2 = image2.resize((600, 550))
    st2_1_11, st2_1_12 = st.columns([2,2],gap="large")
    st2_1_11.image(image1, caption='Classification Report')
    st2_1_12.image(image2, caption='Feature Importance')

    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'>7. Model 7</h4>", unsafe_allow_html=True)
    st.markdown('Model trained on Great Barrier Reef and Torres Strait region and predicted on other Cluster 2 regions using Train-Test Strategy')
    
    image1 = Image.open('imgs/gbr_c2_cm.png')
    image2 = Image.open('imgs/gbr_c2_fi.png')
    image1 = image1.resize((600, 550))
    image2 = image2.resize((600, 550))
    st2_1_13, st2_1_14 = st.columns([2,2],gap="large")
    st2_1_13.image(image1, caption='Classification Report')
    st2_1_14.image(image2, caption='Feature Importance')

# tab2, tab1, tab3, tab4 = st.tabs(["2. Model Results","1. Data Exploration","3. Real-time Prediction","4. Vitality Prediction"])
tab1, tab2, tab3 = st.tabs(["1. Data Exploration","2. Model Results","3. Real-time Prediction"])

with tab1:
    _data_exploration()
    pass

with tab2:
    _model_results()
    pass

with tab3:
    _real_time_prediction()
    pass

# with tab4:
#     _vitality_prediction()
#     pass