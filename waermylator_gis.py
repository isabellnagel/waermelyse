import os
import streamlit as st
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



with DOP_tab:
    st.info("Wähle eine Option, um Daten zu laden:")
    data_source = st.radio("Datenquelle auswählen:", ["WMS-Link", "Satellitenbilder (Verzeichnis)", "GeoTIFF (Verzeichnis)"])

# Content in ML tab
with ml_tab:
    st.subheader("KI-Erkennung")
    image_dir = st.text_input("Gib das Verzeichnis der Bilder zur Verarbeitung an:")

  
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
    geothermie_path = "/Users/hannesrottger/Desktop/Waermelyse/waermelyse/Geothermie_Potenzialflächen_final_area_file.GeoJSON/Geothermie_Potenzialflächen_final_area_file.shp"
    pv_path1 = "/Users/hannesrottger/Desktop/Waermelyse/waermelyse/PV ergebnisse/Dachflächen des Solarkatasters.shp"
    pv_path2 = "/Users/hannesrottger/Desktop/Waermelyse/waermelyse/PV Ergebnisse/Neue Dachflächen (Machine Learning).shp"

    # Erstelle eine Folium-Karte
    m = folium.Map(location=[53.0793, 8.8017], zoom_start=10)  # Bremen als Standard

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
