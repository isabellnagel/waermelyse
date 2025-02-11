import os
from PIL import Image
import streamlit as st
import requests
import xml.etree.ElementTree as ET
from tiff_to_png import split_tif_to_tiles
from wmsdownload import download_image
import sys
import nbimporter
import skimage.io
from skimage.transform import resize
import cv2
import json
import numpy as np

import folium
from streamlit_folium import st_folium
import geopandas as gpd

current_dir = os.getcwd()
working_dir = os.path.join(current_dir, "machine_learning", "roof_indentification")
mask_rcnn_dir = os.path.join(working_dir, "Mask_RCNN")
modelling_py = os.path.join(working_dir, "modelling.py")
mask_rcnn_dir = os.path.join(working_dir, "Mask_RCNN")


modelling_py = os.path.join(working_dir, "modelling.py")
print(modelling_py)
sys.path.append(modelling_py)
sys.path.append(r"C:\\Users\\bilge\\OneDrive\\Masaüstü\\waermelyse\\machine_learning\\roof_indentification\\modelling.py")
if mask_rcnn_dir in working_dir:
    pass
else:
    sys.path.append(mask_rcnn_dir)


# App Header
app_header = "Wärmylator"
st.header(app_header)

# Helper function to manage session state
def add_to_session_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].extend(value)

# Define tabs
DOP_header = "Einladen der Luftbilder"
ml_header = "KI Erkennung"
geodata_header = "Einladen der Geodaten"
gis_header = "GIS Workflows"
output_header = "Endergebnis"
DOP_tab, ml_tab, geodata_tab, gis_tab, output_tab = st.tabs([DOP_header, ml_header, geodata_header, gis_header, output_header])

import streamlit as st
import requests
import xml.etree.ElementTree as ET
import wmsdownload  # Importiere dein Modul

with DOP_tab:
    st.info("Wähle eine Option, um Daten zu laden:")
    data_source = st.radio("Datenquelle auswählen:", ["WMS-Link", "Satellitenbilder (Verzeichnis)", "GeoTIFF (Verzeichnis)"])

# Content in ML tab
with ml_tab:
    st.subheader("KI-Erkennung")
    image_dir = st.text_input("Gib das Verzeichnis der Bilder zur Verarbeitung an:")

  
with geodata_tab:
    st.info("Lade alle Geodaten hoch")
    
    geodata1 = st.text_input("Gib den Pfad zu den ALKIS-Daten an:")
    geodata2 = st.text_input("Gib den Pfad zu den WEA-Daten an:")


    if geodata1:
        if geodata1.endswith((".xml", ".gml", ".shp", ".gpkg", ".dxf")):
            st.success("ALKS-Datenformat erkannt: " + os.path.splitext(geodata1)[1])
        else:
            st.error("Ungültiges Format für ALKIS-Daten. Erlaubt: XML, GML, SHP, GeoPackage, DXF")

    if geodata2:
        if geodata2.endswith((".shp", ".gpkg", ".geojson")):
            st.success("WEA-Datenformat erkannt: " + os.path.splitext(geodata2)[1])
        else:
            st.error("Ungültiges Format für WEA-Daten. Erlaubt: SHP, GeoPackage, GeoJSON")

with gis_tab:
    detecting_potentials = st.multiselect(
        "1. Wähle die gesuchten Potentialflächen",
        ["Geothermie", "PV", "Wärme"]
    )

    if "Geothermie" in detecting_potentials:
        st.success("Hier wird der Code des Geothermie-Workflows durchlaufen")


    if "PV" in detecting_potentials:
        st.success("Hier wird der Code des PV-Workflows durchlaufen")

    if "Wärme" in detecting_potentials:
        st.success("Hier wird der Code des Wärme-Workflows durchlaufen")

    #Angenommen, dein GIS-Skript heißt gis_workflow.py und hat eine Funktion run_gis_workflow(alkis_path, ml_path, output_path), 
    # dann müssen die noch oben impoirtiert werden

with output_tab:
    # Lade die Geodaten als Geopandas-Datenrahmen (ersetze die Dateipfade)
    gdf_daecher = gpd.read_file("pfad_zu_daecher.geojson")
    gdf_baeume = gpd.read_file("pfad_zu_baeume.geojson")
    gdf_alkis = gpd.read_file("pfad_zu_alkis.geojson")

    # Beispielhafte Koordinaten für den Kartenmittelpunkt
    map_center = [52.0, 10.0]  # Beispiel für Niedersachsen

    # Streamlit-Oberfläche
    st.title("Ergebnisdarstellung der GIS-Potentialanalyse")

    # Erstelle die Grundkarte mit folium
    m = folium.Map(location=map_center, zoom_start=12, control_scale=True)

    # Füge Layer hinzu (ersetzte `gdf.to_crs(epsg=4326)` falls Daten bereits im WGS84-Format sind)
    if "gdf_daecher" in locals():
        folium.GeoJson(gdf_daecher.to_crs(epsg=4326), name="Dächer").add_to(m)

    if "gdf_baeume" in locals():
        folium.GeoJson(gdf_baeume.to_crs(epsg=4326), name="Bäume").add_to(m)

    if "gdf_alkis" in locals():
        folium.GeoJson(gdf_alkis.to_crs(epsg=4326), name="ALKIS-Daten").add_to(m)

    # Layer-Kontrolle hinzufügen
    folium.LayerControl().add_to(m)

    # Zeige die Karte in Streamlit
    st_folium(m, width=800, height=600)






