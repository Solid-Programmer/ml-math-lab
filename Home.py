import streamlit as st

st.set_page_config(page_title="ML Math Lab", layout="centered")

st.title("ML Math Lab")
st.write("""
Welcome to the ML Math Lab!  
Explore interactive tools and visualizations to deepen your understanding of the mathematics behind machine learning.
""")

st.sidebar.title("Navigation")


# Add navigation link to PCA Visualizer page
st.sidebar.markdown("[PCA Visualizer](./PCA_Visualizer.py)")