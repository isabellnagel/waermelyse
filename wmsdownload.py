import os
import json
from owslib.wms import WebMapService
from PIL import Image
from io import BytesIO

# Speicherort für Bilder
DOWNLOAD_DIR = "wms_tiles"

def ensure_directory(directory):
    """Stellt sicher, dass das Verzeichnis existiert."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_all_metadata(metadata_list, filename):
    """Speichert die Metadaten als JSON."""
    with open(filename, "w") as f:
        json.dump(metadata_list, f, indent=4)

def download_image(wms_url, layer, bbox, tile_size=(320, 320), crs="EPSG:25832", resolution=0.1):
    """Holt Kartenbilder vom WMS-Server und speichert sie in Kacheln."""
    wms = WebMapService(wms_url, version="1.3.0")

    # Breite und Höhe der Bounding Box in Metern
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    # Anzahl der Kacheln in x- und y-Richtung
    num_tiles_x = int(bbox_width / (tile_size[0] * resolution))
    num_tiles_y = int(bbox_height / (tile_size[1] * resolution))

    # Breite und Höhe jeder Kachel in Metern
    tile_width = bbox_width / num_tiles_x
    tile_height = bbox_height / num_tiles_y

    # Sicherstellen, dass der Download-Ordner existiert
    ensure_directory(DOWNLOAD_DIR)

    metadata_list = []

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            tile_bbox = (
                bbox[0] + j * tile_width,
                bbox[1] + i * tile_height,
                bbox[0] + (j + 1) * tile_width,
                bbox[1] + (i + 1) * tile_height
            )

            filename = os.path.join(DOWNLOAD_DIR, f"tile_{int(tile_bbox[0])}_{int(tile_bbox[1])}.png")

            # WMS-Request für die Kachel
            response = wms.getmap(
                layers=[layer],
                srs=crs,
                bbox=tile_bbox,
                size=tile_size,
                format="image/png",
                transparent=True
            )

            # Bild speichern
            img = Image.open(BytesIO(response.read()))
            img.save(filename)

            # Metadaten speichern
            metadata_list.append({
                "filename": filename,
                "bbox": tile_bbox,
                "size": tile_size,
                "crs": crs,
                "resolution": resolution
            })

    # Metadaten als JSON speichern
    save_all_metadata(metadata_list, os.path.join(DOWNLOAD_DIR, "metadata.json"))

    return DOWNLOAD_DIR  # Gibt das Verzeichnis zurück
