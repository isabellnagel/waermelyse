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
import sys
import nbimporter
import skimage.io
from skimage.transform import resize
import cv2
import json
import numpy as np

current_dir = os.getcwd()
working_dir = os.path.join(current_dir, "machine_learning", "roof_indentification")
mask_rcnn_dir = os.path.join(working_dir, "Mask_RCNN")
modelling_py = os.path.join(working_dir, "modelling.py")
mask_rcnn_dir = os.path.join(working_dir, "Mask_RCNN")


modelling_py = os.path.join(working_dir, "modelling.py")
print(modelling_py)
sys.path.append(modelling_py)
sys.path.append(r"C:\Users\MSI\Uni\master_projekt\waermelyse\machine_learning\roof_indentification\modelling.py")
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
gis_header = "GIS Workflows"
output_header = "Endergebnis"
DOP_tab, ml_tab, gis_tab, output_tab = st.tabs([DOP_header, ml_header, gis_header, output_header])

# Content in DOP tab
with DOP_tab:
    st.info("Wähle eine Option, um Daten zu laden:")

    # Auswahl zwischen WMS-Link, Bild-Upload oder GeoTIFF
    data_source = st.radio("Datenquelle auswählen:", ["WMS-Link", "Satellitenbilder hochladen", "GeoTIFF-Datei hochladen"])

    # Option: WMS-Link
    if data_source == "WMS-Link":
        wms_url = st.text_input("Gib den WMS-Link ein:", "")
        
        if wms_url:
            # Abruf der Layer
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
                        st.success(f"{len(layers)} Layer erfolgreich abgerufen!")
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
                    image = download_image(wms_url, selected_layer, bbox)
                    st.image(image, caption=f"Layer: {selected_layer}", use_column_width=True)
                    st.success("Bild erfolgreich heruntergeladen!")
                except Exception as e:
                    st.error(f"Ein Fehler ist aufgetreten: {e}")

    # Option: Satellitenbilder hochladen
    elif data_source == "Satellitenbilder hochladen":
        uploaded_files = st.file_uploader("Bilder hochladen (.jpg, .png)", type=["jpg", "png"], accept_multiple_files=True)
        if uploaded_files:
            # Speichern der hochgeladenen Dateien in Session State
            add_to_session_state("uploaded_files", uploaded_files)
            total_files = len(st.session_state["uploaded_files"])
            st.info(f"{total_files} Datei(en) insgesamt hochgeladen.")
            for file in uploaded_files:
                st.write(f"Dateiname: {file.name}")

    # Option: GeoTIFF-Datei hochladen
    elif data_source == "GeoTIFF-Datei hochladen":
        geotiff_files = st.file_uploader("GeoTIFF-Datei hochladen (.tif, .tiff)", type=["tif", "tiff"], accept_multiple_files=True)

        if geotiff_files:
            for geotiff_file in geotiff_files:
                st.success(f"GeoTIFF-Datei '{geotiff_file.name}' erfolgreich hochgeladen.")

                # Temporärer Pfad für die GeoTIFF-Datei
                temp_input_path = os.path.join("/tmp", geotiff_file.name)

                # Datei speichern
                with open(temp_input_path, "wb") as f:
                    f.write(geotiff_file.getbuffer())

                # Zielordner für die Kacheln
                output_folder = os.path.join("/tmp", f"tiles_{os.path.splitext(geotiff_file.name)[0]}")
                os.makedirs(output_folder, exist_ok=True)

                # GeoTIFF in Kacheln zerlegen
                st.info("Verarbeite GeoTIFF...")
                try:
                    # Nutze split_tif_to_tiles mit dem temporären Pfad
                    split_tif_to_tiles(temp_input_path, output_folder)

                    # Lade Kacheln in den Session State
                    tile_paths = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder)]
                    add_to_session_state("tiff_tiles", tile_paths)
                    st.success(f"GeoTIFF erfolgreich in {len(tile_paths)} PNG-Kacheln zerlegt.")
                except Exception as e:
                    st.error(f"Fehler beim Zerlegen des GeoTIFF: {e}")

# Content in ML tab
with ml_tab:
    st.subheader("KI-Erkennung")
    detecting_object = st.radio("1. Wähle das zu erkennende Objekt", ["Bäume", "Dächer"])
    config = SetupConfig()
    #config.display()
    COCO_MODEL_PATH = os.path.join(working_dir,"coco20250116T1546", "mask_rcnn_coco_0010.h5")
    ROOT_DIR = os.path.join(working_dir,"model")
    st.info("loading mask R-CNN model")
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)
    #-------------- Load weights trained on MS-COCO -------------------------------
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Folder containing images
    input_folder = r"C:\Users\MSI\Uni\master_projekt\waermelyse\machine_learning\320x320_Annotationen\valid_dataset\images"
    # Output JSON file
    output_json = "output_predictions.json"
    process_folder_and_save(input_folder, model, output_json)

    # Verarbeite hochgeladene Satellitenbilder
    if "uploaded_files" in st.session_state and st.session_state["uploaded_files"]:
        st.write("Verfügbare Satellitenbilder zur Verarbeitung:")
        for file in st.session_state["uploaded_files"]:
            print(file)
            st.image(file, caption=f"Bild: {file.name}", use_column_width=True)
            st.success(f"Erkennung für '{file.name}' abgeschlossen!")

    # Verarbeite GeoTIFF-Kacheln
    if "tiff_tiles" in st.session_state and st.session_state["tiff_tiles"]:
        st.write("Verfügbare GeoTIFF-Kacheln zur Verarbeitung:")
        for tile_path in st.session_state["tiff_tiles"]:
            st.image(tile_path, caption=f"Kachel: {os.path.basename(tile_path)}", use_column_width=True)
            st.success(f"Erkennung für Kachel '{os.path.basename(tile_path)}' abgeschlossen!")
    else:
        st.warning("Es wurden noch keine GeoTIFF-Kacheln oder Satellitenbilder erstellt/hochgeladen.")
