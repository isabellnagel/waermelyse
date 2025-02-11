import os
from PIL import Image
from pycocotools import mask as mask_utils
import streamlit as st
import requests
import xml.etree.ElementTree as ET
from tiff_to_png import split_tif_to_tiles
from wmsdownload import download_image
from machine_learning.roof_indentification.modelling import SetupConfig
from machine_learning.roof_indentification.modelling import process_folder_and_save
from machine_learning.Georeferencing_WMS_1.georeferencing_wms import create_geojson
import sys
import nbimporter
import skimage.io
from skimage.transform import resize
import cv2
import json
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium

current_dir = os.getcwd()
working_dir = os.path.join(current_dir, "machine_learning", "roof_indentification")
mask_rcnn_dir = os.path.join(working_dir, "Mask_RCNN")
modelling_py = os.path.join(working_dir, "modelling.py")
mask_rcnn_dir = os.path.join(working_dir, "Mask_RCNN")


modelling_py = os.path.join(working_dir, "modelling.py")
print(modelling_py)
sys.path.append(modelling_py)
#sys.path.append(r"C:\\Users\\bilge\\OneDrive\\Masaüstü\\waermelyse\\machine_learning\\roof_indentification\\modelling.py")
if mask_rcnn_dir in working_dir:
    pass
else:
    sys.path.append(mask_rcnn_dir)
#sys.path.append(r"C:\Users\MSI\Uni\master_projekt\waermelyse\machine_learning\roof_indentification\Mask_RCNN")


from mrcnn import utils
from mrcnn.utils import Dataset
from mrcnn.model import log
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.config import Config


#COCO_MODEL_PATH = r"C:\Users\MSI\Uni\master_projekt\waermelyse\machine_learning\roof_indentification\data\transfer_pretrained_weights.h5"

#sys.path.append(r"C:\Users\MSI\Uni\master_projekt\waermelyse\machine_learning\roof_indentification\Mask_RCNN\samples\coco")  # To find local version
coco_path = os.path.join(mask_rcnn_dir, "samples", "coco")

if coco_path in sys.path:
    pass
else:
    sys.path.append(coco_path)

import coco
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


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

    if data_source == "WMS-Link":
        wms_url = st.text_input("Gib den WMS-Link ein:", "")
        if wms_url:
            if st.button("Abrufen"):
                try:
                    capabilities_url = f"{wms_url}?service=WMS&request=GetCapabilities"
                    response = requests.get(capabilities_url)
                    if response.status_code == 200:
                        root = ET.fromstring(response.content)
                        layers = [
                            layer.find("Name").text
                            for layer in root.findall(".//{http://www.opengis.net/wms}Layer")
                            if layer.find("Name") is not None
                        ]
                        st.session_state["wms_layers"] = layers
                        #st.success(f"{len(layers)} Layer erfolgreich abgerufen!")
                    else:
                        st.error(f"Fehler beim Abrufen der Layer: HTTP {response.status_code}")
                except Exception as e:
                    st.error(f"Ein Fehler ist aufgetreten: {e}")

        # Auswahl der Layer
        if "wms_layers" in st.session_state:
            selected_layer = st.selectbox("Wähle einen Layer aus:", st.session_state["wms_layers"])

            # Eingabe der Bounding Box
            st.write("Gib die Bounding Box (xmin, ymin, xmax, ymax) ein:")
            col1, col2, col3, col4 = st.columns(4)
            xmin = col1.number_input("xmin", value=0.0)
            ymin = col2.number_input("ymin", value=0.0)
            xmax = col3.number_input("xmax", value=0.0)
            ymax = col4.number_input("ymax", value=0.0)

            if st.button("Bild herunterladen"):
                try:
                    bbox = (xmin, ymin, xmax, ymax)
            
                    # Bilder herunterladen und Speicherort erhalten
                    download_path = download_image(wms_url, "DOP10_2023_HB", bbox)
                    # Alle heruntergeladenen Bilder aus dem Ordner anzeigen
                    image_files = [os.path.join(download_path, f) for f in os.listdir(download_path) if f.endswith(".png")]
                    if image_files:
                        for image_file in image_files:
                            st.image(image_file, caption=f"Kachel: {os.path.basename(image_file)}", use_column_width=True)
                        
                        st.success(f"{len(image_files)} Kacheln erfolgreich heruntergeladen!")
                    else:
                        st.warning("Keine Bilder gefunden.")

                except Exception as e:
                    st.error(f"Ein Fehler ist aufgetreten: {e}")
        
        elif data_source == "Satellitenbilder":
            image_dir = st.text_input("Gib das Verzeichnis mit Satellitenbildern an:")
            if image_dir and os.path.exists(image_dir):
                st.success(f"Satellitenbilder werden aus {image_dir} geladen.")
            else:
                st.warning("Gültiges Verzeichnis angeben.")
        
        elif data_source == "GeoTIFF":
            geotiff_dir = st.text_input("Gib das Verzeichnis mit GeoTIFF-Dateien an:")
            output_folder = os.path.join(geotiff_dir, "tiles")
            
            if geotiff_dir and os.path.exists(geotiff_dir):
                st.info("GeoTIFF-Dateien werden verarbeitet...")
                os.makedirs(output_folder, exist_ok=True)
                try:
                    split_tif_to_tiles(geotiff_dir, output_folder)
                    st.success(f"GeoTIFF erfolgreich in {len(os.listdir(output_folder))} PNG-Kacheln zerlegt.")
                except Exception as e:
                    st.error(f"Fehler beim Zerlegen des GeoTIFF: {e}")
            else:
                st.warning("Gültiges Verzeichnis angeben.")

# Content in ML tab
with ml_tab:
    st.subheader("KI-Erkennung")
    image_dir = st.text_input("Gib das Verzeichnis der Bilder zur Verarbeitung an:")
    detecting_objects = st.multiselect("1. Wähle das zu erkennende Objekt", ["Bäume", "Dächer"])

    if image_dir and os.path.exists(image_dir) and st.button("Starte Modellerkennung"):
        config = SetupConfig()
        combined_results = []

        for obj in detecting_objects:
            model_type = "roof" if obj == "Dächer" else "tree"
            st.info(f"Lade Modell für {obj}...")

            # Dynamische Festlegung des Modellpfads
            COCO_MODEL_PATH = os.path.join(working_dir, "gewichte_modell", model_type, f"mask_rcnn_coco_00{'10' if model_type == 'roof' else '05'}.h5")

            if os.path.exists(COCO_MODEL_PATH):
                ROOT_DIR = os.path.join(working_dir, "model")
                model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)
                model.load_weights(COCO_MODEL_PATH, by_name=True)

                output_json = os.path.join(image_dir, f"output_predictions_{model_type}.json")
                st.info(f"Starte die Erkennung für {obj}...")

                process_folder_and_save(image_dir, model, output_json)

                # Ergebnisse sammeln
                with open(output_json, 'r') as f:
                    results = json.load(f)
                    combined_results.extend(results)

                st.success(f"Erkennung für {obj} abgeschlossen!")

                st.info("Starte die Georeferenzierung...")

                # Metadaten-Datei aus demselben Verzeichnis laden
                metadata_file = os.path.join(image_dir, "metadata.json")
                
                if os.path.exists(metadata_file):
                    geojson_result = create_geojson(output_json, metadata_file)

                    # Speichern der GeoJSON-Datei im gleichen Verzeichnis
                    output_geojson_path = os.path.join(image_dir, f"georeferenced_results_{model_type}.geojson")
                    with open(output_geojson_path, 'w') as f:
                        json.dump(geojson_result, f)

                    st.success(f"Georeferenzierung abgeschlossen! Datei gespeichert unter: {output_geojson_path}")
                else:
                    st.error(f"Metadaten-Datei {metadata_file} nicht gefunden! Bitte überprüfe das Verzeichnis.")
            else:
                st.error(f"Das Modell für {obj} ({COCO_MODEL_PATH}) wurde nicht gefunden!")

with geodata_tab:
    st.info("Lade alle Geodaten hoch")
    GebaeudeBauwerk = st.text_input("Gib den Pfad zu den GebaeudeBauwerk-Daten an:")
    Baeume = st.text_input("Gib den Pfad zu den Baumdaten an:")
    Wsg = st.text_input("Gib den Pfad zu den Wasserschutzgebieten an:")
    Nsg = st.text_input("Gib den Pfad zu den Naturschutzgebieten an:")
    Vsg = st.text_input("Gib den Pfad zu den Vogelschutzgebieten an:")
    Lsg = st.text_input("Gib den Pfad zu den Landschaftschutzgebieten an:")
    FFh = st.text_input("Gib den Pfad zu den FFH-Daten an:")
    NutzungFlurstueck = st.text_input("Gib den Pfad zu den NutzungFlurstueks-Daten an:")
    Windenergieanlagen = st.text_input("Gib den Pfad zu den Windenergiedaten an:")
    Zielgebiet = st.text_input("Gib den Pfad zu den Daten des Zielgebietes an:")


    if GebaeudeBauwerk:
        if GebaeudeBauwerk.endswith((".xml", ".gml", ".shp", ".gpkg", ".dxf")):
            st.success("ALKS-Datenformat erkannt: " + os.path.splitext(GebaeudeBauwerk)[1])
        else:
            st.error("Ungültiges Format für ALKIS-Daten. Erlaubt: XML, GML, SHP, GeoPackage, DXF")

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
      # Dropdown-Menü für die Auswahl
    option = st.selectbox("Wähle die gewünschte Flächenart:", ["Geothermie-Potenzialflächen", "PV-Flächen"])

    # Dateipfade für beide Optionen
    #geothermie_path = "/Users/hannesrottger/Desktop/Waermelyse/waermelyse/Geothermie_Potenzialflächen_final_area_file.GeoJSON/Geothermie_Potenzialflächen_final_area_file.shp"
    #pv_path1 = "/Users/hannesrottger/Desktop/Waermelyse/waermelyse/PV ergebnisse/Dachflächen des Solarkatasters.shp"
    #pv_path2 = "/Users/hannesrottger/Desktop/Waermelyse/waermelyse/PV Ergebnisse/Neue Dachflächen (Machine Learning).shp"
    # Erstelle eine Folium-Karte
    m = folium.Map(location=[53.0793, 8.8017], zoom_start=10)  # Bremen als Standard

    path_available = False #das hier kann rausgenommen, sobald die richtigen ergebnisse genereriert werden können
    if path_available:
        if option == "Geothermie-Potenzialflächen":
            # Lade Geothermie-Daten
            gdf = gpd.read_file(geothermie_path)

            # Falls nötig, umprojizieren
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            # Geothermie-Flächen als Layer hinzufügen
            folium.GeoJson(gdf.to_json(), name="Geothermie-Potenzialflächen", tooltip="Geothermie").add_to(m)

            # Automatische Kartenanpassung
            bounds = gdf.total_bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        elif option == "PV-Flächen":
            # Lade beide PV-Datensätze
            gdf1 = gpd.read_file(pv_path1)
            gdf2 = gpd.read_file(pv_path2)

            # Falls nötig, umprojizieren
            if gdf1.crs != "EPSG:4326":
                gdf1 = gdf1.to_crs("EPSG:4326")
            if gdf2.crs != "EPSG:4326":
                gdf2 = gdf2.to_crs("EPSG:4326")

            # PV-Flächen (Ergebnis 1) als Layer hinzufügen
            folium.GeoJson(gdf1.to_json(), name="PV-Flächen Ergebnis 1", tooltip="PV 1", style_function=lambda x: {"color": "yellow"}).add_to(m)

            # PV-Flächen (Ergebnis 2) als Layer hinzufügen
            folium.GeoJson(gdf2.to_json(), name="PV-Flächen Ergebnis 2", tooltip="PV 2", style_function=lambda x: {"color": "red"}).add_to(m)

                # Automatische Kartenanpassung auf beide Layer
            bounds = [
                min(gdf1.total_bounds[1], gdf2.total_bounds[1]),  # min Y
                min(gdf1.total_bounds[0], gdf2.total_bounds[0]),  # min X
                max(gdf1.total_bounds[3], gdf2.total_bounds[3]),  # max Y
                max(gdf1.total_bounds[2], gdf2.total_bounds[2])   # max X
            ]
            m.fit_bounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]])
        
        

    # Layer-Control hinzufügen, damit man Layer an-/ausschalten kann
    folium.LayerControl().add_to(m)

    # Karte in Streamlit anzeigen
    st_folium(m, width=800, height=500)






