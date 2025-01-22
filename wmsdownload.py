import requests
from PIL import Image
from io import BytesIO

def download_image(wms_url, layer, bbox, width=512, height=512, srs="EPSG:4326", image_format="image/png"):
    """
    Lädt ein Bild von einem WMS-Server herunter.
    """
    bbox_str = ",".join(map(str, bbox))
    request_url = (
        f"{wms_url}?service=WMS&request=GetMap&layers={layer}"
        f"&bbox={bbox_str}&width={width}&height={height}"
        f"&srs={srs}&format={image_format}"
    )
    response = requests.get(request_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        raise Exception(f"Fehler beim Abrufen des Bildes: HTTP {response.status_code}")
