import streamlit as st

st.set_page_config(page_title="ML Math Lab", layout="centered")

st.title("ML Math Lab")
st.write("""
Welcome to the ML Math Lab!  
Explore interactive tools and visualizations to deepen your understanding of the mathematics behind machine learning.
""")

st.sidebar.title("Navigation")
st.sidebar.markdown("## Math-For-ML")

st.markdown("---")
st.header("Math-For-ML Interactive Apps")

col1, col2 = st.columns(2)

with col1:
    st.subheader("PCA Visualizer")
    st.markdown("""
    - Step-by-step Principal Component Analysis
    - 2D/3D visualizations and projections
    - Explore dimensionality reduction interactively
    """)
    st.markdown(
        "<span style='color:gray;'>Use the sidebar to open this app.</span>", unsafe_allow_html=True
    )

with col2:
    st.subheader("Gradient Descent Playground")
    st.markdown("""
    - Visualize and experiment with gradient descent
    - Adjust learning rate, epochs, and see optimizer paths
    - Understand MAE loss and optimization dynamics
    """)
    st.markdown(
        "<span style='color:gray;'>Use the sidebar to open this app.</span>", unsafe_allow_html=True
    )