# PCA Visualizer Streamlit Page
# (This file imports and runs the main visualizer app)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PCA_Visualizer import visualizer
import streamlit as st