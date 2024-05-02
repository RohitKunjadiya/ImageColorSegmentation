import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image,ImageDraw

st.title('Image Color Segmentation')

st.write('Color segmentation is a technique used in computer vision to identify and distinguish different objects or regions in an image based on their colors. Clustering algorithms can automatically group similar colors together, without the need to specify threshold values for each color. This can be useful when working with images that have a large range of colors, or when the exact threshold values are not known in advance.')

st.subheader("Let's Start")
def main():

    img = st.file_uploader('Upload An Image:')

    if img is not None:
        img1 = Image.open(img)
        st.image(img1, caption='Uploaded Image', use_column_width=True)
        img_array = np.array(img1)

        img2 = img_array
        # print(img2.shape)

        x = img2.reshape(-1,3)
        # print(x.shape)

        km = KMeans(n_clusters=3,init='k-means++',n_init=10)
        km.fit(x)

        dominant_colors = km.cluster_centers_.astype(int)
        palette_size = (750, 100)

        # Create an image to display the colors
        palette = Image.new("RGB",palette_size)
        draw = ImageDraw.Draw(palette)

        # Calculate the width of each color swatch
        swatch_width = palette_size[0] // len(dominant_colors)

        # Draw each color as a rectangle on the palette
        for i,color in enumerate(dominant_colors):
            draw.rectangle([i*int(swatch_width),0,(i+1)*int(swatch_width),palette_size[1]],fill=tuple(color))

        btn = st.button('Show Output')

        if btn:
            st.write('Top-3 Frequent Colour is:')
            st.image(palette)

main()