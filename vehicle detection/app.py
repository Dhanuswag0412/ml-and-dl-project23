import sys
import os
from streamlit import runtime

# Ensure the directory containing vehicle_detection is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vehicle_detection.gui import main

if runtime.exists():
    # Running inside Streamlit server, render the UI
    main()
else:
    # Running as a regular Python script, launch Streamlit server pointing to this file
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())