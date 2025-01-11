import streamlit as st

# App Header
app_header = "Wärmylator"
st.header(app_header)

# Define tabs
wms_header = "WMS Export"
ml_header = "KI Erkennung"
gis_header = "GIS Workflows"
output_header = "Endergebnis"
wms_tab, ml_tab, gis_tab, output_tab = st.tabs([wms_header, ml_header, gis_header, output_header])

# Content in WMS tab
with wms_tab:
    st.info("Hier evtl interaktive Maske zu WMS Export")
    bar = st.progress(25)

# Content in ML tab
with ml_tab:
    st.subheader("1. Wähle das zu erkennende Objekt")  # Adding a subheader for clarity
    detecting_object = st.radio("1. Wähle das zu erkennende Objekt", ["Bäume", "Dächer"])
    
    # Display uploaded files if selected
    st.subheader("2. Wähle einen Ordner mit Satellitenbildern (.jpg)") 
    uploaded_files = st.file_uploader("2. Wähle einen Ordner mit Satellitenbildern (.jpg)", type=['png', 'jpg'], accept_multiple_files=True)

    bar = st.progress(50)