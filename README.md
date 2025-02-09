# Wärmelyse

## Kurzbeschreibung

Dieses Repository enthält Skripte zu **Tranfer Learning**, um anhand Satellitenbildern **Dächer und Bäume zu erkennen**, sowie aus **ArcGIS Pro** exportierten Code, welche **Open Source übersetzt** wurde, um Geothermie und Solarenergie **Potenzialflächen** zu analysieren.
Der Quellecode wurde in eine webbasierte streamlit app verpackt.

## Voraussetzungen
- Python 3.7.3
- Wenn Pip und requirements.txt benutze wird: Pip 21.0 für psutil, ansonsten Pip 19.0.3
- Zusätzlich zu Requirements.txt: Matterports Mask-RCNN [https://github.com/matterport/Mask_RCNN]

## Struktur
- Übersetzug der Potenzialflächenanalyse: waermelyse\GIS
- Transfer Learning: waermelyse\roof_indentification
- Streamlit App: waermelyse\waermylator.py

## Benutzung der App
- Über das Terminal: streamlit run waermylator.py
- Das vollständige Laden der App im Browser kann einige Sekunden dauern