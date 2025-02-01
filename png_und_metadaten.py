import os
from owslib.wms import WebMapService
import json

# Verzeichnis für Bilder und Metadaten
output_folder = "output/tiles"
os.makedirs(output_folder, exist_ok=True)

# Liste für alle Metadaten
metadata_list = []

# Iteriere über alle Kacheln
for i in range(num_tiles_y):
    for j in range(num_tiles_x):
        tile_bbox = (
            bbox[0] + j * tile_width,
            bbox[1] + i * tile_height,
            bbox[0] + (j + 1) * tile_width,
            bbox[1] + (i + 1) * tile_height
        )

        # Pfad für die Kachel
        filename = os.path.join(output_folder, f"tile_{j}_{i}.png")

        # Bild und Metadaten herunterladen
        download_image(
            wms, layer, tile_bbox, 
            (tile_size, tile_size), 
            "image/png", filename, crs, resolution, metadata_list
        )

# Speichere Metadaten
save_all_metadata(metadata_list, "output/metadata.json")
print("Alle Bilder und Metadaten erfolgreich gespeichert!")