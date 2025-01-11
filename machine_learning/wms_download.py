import streamlit as st
from owslib.wms import WebMapService
from PIL import Image
from io import BytesIO
import requests
import os
import zipfile

# Funktion, um WMS-Bilder herunterzuladen
def download_image(wms_url, layer, bbox, width, height, format="image/png"):
    """Lade ein Bild von einem WMS-Server herunter."""
    wms = WebMapService(wms_url)
    response = wms.getmap(
        layers=[layer],
        srs="EPSG:4326",
        bbox=bbox,
        width=width,
        height=height,
        format=format,
        transparent=True
    )
    return Image.open(BytesIO(response.read()))

# Funktion, um Kacheln zu erstellen und herunterzuladen
def download_tiles(wms_url, layer, bbox, tile_size):
    """Teile die Bounding Box in Kacheln und lade sie herunter."""
    minx, miny, maxx, maxy = bbox
    tiles = []
    x_range = range(int(minx), int(maxx), tile_size)
    y_range = range(int(miny), int(maxy), tile_size)

    for x in x_range:
        for y in y_range:
            tile_bbox = (x, y, min(x + tile_size, maxx), min(y + tile_size, maxy))
            try:
                tile = download_image(wms_url, layer, tile_bbox, tile_size, tile_size)
                tiles.append((tile, tile_bbox))
            except Exception as e:
                st.error(f"Fehler beim Laden der Kachel {tile_bbox}: {e}")

    return tiles

# UI-Komponenten für den WMS-Tab
st.header("WMS Bild herunterladen und kacheln")

# WMS-Server und Layer-Auswahl
wms_url = st.text_input("WMS-Server URL", "https://ows.terrestris.de/osm/service")
layer = st.text_input("Layer", "OSM-WMS")

# Bounding Box und Bildgröße
bbox = st.text_input("Bounding Box (minx, miny, maxx, maxy)", "-180,-90,180,90")
tile_size = st.slider("Kachelgröße (Pixel)", 256, 1024, 512)

if st.button("Kacheln herunterladen"):
    try:
        bbox_values = [float(coord) for coord in bbox.split(",")]
        with st.spinner("Bilder werden heruntergeladen..."):
            tiles = download_tiles(wms_url, layer, bbox_values, tile_size)

        # ZIP-Datei erstellen
        zip_path = "tiles.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for idx, (tile, tile_bbox) in enumerate(tiles):
                tile_path = f"tile_{idx}.png"
                tile.save(tile_path)
                zipf.write(tile_path)
                os.remove(tile_path)

        # Download-Link für ZIP-Datei
        with open(zip_path, "rb") as f:
            st.download_button("Kacheln herunterladen", f, file_name="tiles.zip")
        os.remove(zip_path)

        # Kacheln anzeigen
        st.success("Kacheln erfolgreich heruntergeladen!")
        for tile, tile_bbox in tiles:
            st.image(tile, caption=f"Kachel {tile_bbox}", use_column_width=True)

    except Exception as e:
        st.error(f"Fehler: {e}")
