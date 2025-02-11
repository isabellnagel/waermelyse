import json
import os
from shapely.geometry import Polygon
from pyproj import Transformer
import re
from pyproj import CRS

def extract_coordinates(filename):
    # Extrahiert die Koordinaten aus dem Dateinamen
    match = re.search(r'tile_(\d+)_(\d+)', filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None

def find_matching_metadata(metadata_file, coordinates):
    # Sucht nach einer passenden Metadaten in der großen JSON-Datei
    with open(metadata_file, 'r') as f:
        metadata_list = json.load(f)
    
    # Durchsuche die Metadaten und finde das passende Element
    for metadata in metadata_list:
        if coordinates in metadata['filename']:  # Hier wird nach den Koordinaten im Dateinamen gesucht
            return metadata
    return None

def pixel_to_coord(px, py, bbox, img_width, img_height):
    world_width = bbox[2] - bbox[0]
    world_height = bbox[3] - bbox[1]
    x = bbox[0] + (px / img_width) * world_width
    y = bbox[3] - (py / img_height) * world_height
    return x, y

def create_geojson(model_output_file, metadata_file):
    with open(model_output_file, 'r') as f:
        model_output = json.load(f)

    features = []

    # Durchlaufe alle Bilder und füge entsprechende Geodaten hinzu
    for image in model_output['images']:
        image_file = os.path.basename(image['file_name'])
        coordinates = extract_coordinates(image_file)
        if not coordinates:
            print(f"Konnte keine Koordinaten aus {image_file} extrahieren. Überspringe dieses Bild.")
            continue

        metadata = find_matching_metadata(metadata_file, coordinates)
        if not metadata:
            print(f"Keine passenden Metadaten für {image_file} gefunden. Überspringe dieses Bild.")
            continue

        bbox = metadata['bbox']
        crs = metadata['crs']

        # Hier ist die Änderung:
        transformer = Transformer.from_crs(CRS.from_string(crs), CRS.from_epsg(25832), always_xy=True)

        annotations = [ann for ann in model_output['annotations'] if ann['image_id'] == image['id']]

        for ann in annotations:
            if ann['category_id'] == 1 and ann['segmentation']:
                # Konvertiere die Segmentation in Weltkoordinaten
                world_coords = [
                    pixel_to_coord(x, y, bbox, image['width'], image['height'])
                    for x, y in zip(ann['segmentation'][0][::2], ann['segmentation'][0][1::2])
                ]
                epsg25832_coords = [transformer.transform(x, y) for x, y in world_coords]

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [epsg25832_coords]
                    },
                    "properties": {
                        "image_id": ann['image_id'],
                        "annotation_id": ann['id'],
                        "area": ann['area']
                    }
                }
                features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:25832"  # Setze das CRS des GeoJSON auf EPSG:25832
            }
        },
        "features": features
    }

    return geojson


# Verwendung der Funktion
model_output_file = r'C:\Users\bilge\OneDrive\Masaüstü\waermelyse\machine_learning\roof_indentification\output_predictions_roofs.json'
metadata_file = r'C:\Users\bilge\OneDrive\Masaüstü\waermelyse\machine_learning\Georeferencing_WMS_1\metadata.json'  # Hier auf die große JSON-Datei verweisen
geojson_result = create_geojson(model_output_file, metadata_file)

# Speichern der GeoJSON-Datei
with open('roofs_georeferenced.geojson', 'w') as f:
    json.dump(geojson_result, f)
