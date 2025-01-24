from PIL import Image
import os

# Deaktiviert die Begrenzung für große Bilder
Image.MAX_IMAGE_PIXELS = None

def split_tif_to_tiles(geotiff_file, output_folder, tile_size=(320, 320)):
    """
    Zerlegt eine TIFF-Datei in kleinere Kacheln und speichert sie als Bilder.

    :param geotiff_file: BytesIO oder Dateipfad der Eingabe-TIFF-Datei
    :param output_folder: Ordner, in dem die Kacheln gespeichert werden
    :param tile_size: Tuple mit der Breite und Höhe der Kacheln (Standard: 320x320)
    """
    # Sicherstellen, dass der Ausgabeordner existiert
    os.makedirs(output_folder, exist_ok=True)

    # Öffne die TIFF-Datei
    with Image.open(geotiff_file) as img:
        width, height = img.size
        tile_width, tile_height = tile_size

        # Zähle die Kacheln
        tile_count = 0

        # Schleife über die Koordinaten
        for top in range(0, height, tile_height):
            for left in range(0, width, tile_width):
                # Berechne die Kachelgrenzen
                right = min(left + tile_width, width) 
                bottom = min(top + tile_height, height)

                # Schneide die Kachel aus
                tile = img.crop((left, top, right, bottom))

                # Speicher die Kachel
                tile_filename = f"tile_{tile_count:04d}.png"
                tile.save(os.path.join(output_folder, tile_filename))

                tile_count += 1

    print(f"{tile_count} Kacheln gespeichert im Ordner '{output_folder}'")
