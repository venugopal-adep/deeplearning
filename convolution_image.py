import streamlit as st
import numpy as np
from PIL import Image

def convolve(image, kernel):
    image_height, image_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def get_filters():
    filters = {
        "Identity": {
            "kernel": [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ],
            "explanation": "The identity filter leaves the image unchanged. It is useful for testing and comparison purposes.",
            "example": "Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nOutput: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]"
        },
        "Edge Detection (Horizontal)": {
            "kernel": [
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ],
            "explanation": "The horizontal edge detection filter highlights horizontal edges in the image. It is commonly used for edge detection and feature extraction.",
            "example": "Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nOutput: [[-12, -12, -12], [0, 0, 0], [12, 12, 12]]"
        },
        "Edge Detection (Vertical)": {
            "kernel": [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ],
            "explanation": "The vertical edge detection filter highlights vertical edges in the image. It is commonly used for edge detection and feature extraction.",
            "example": "Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nOutput: [[-6, 0, 6], [-6, 0, 6], [-6, 0, 6]]"
        },
        "Sharpen": {
            "kernel": [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ],
            "explanation": "The sharpen filter enhances the edges and details in the image, making it appear sharper.",
            "example": "Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nOutput: [[0, -3, 0], [-3, 45, -3], [0, -3, 0]]"
        },
        "Box Blur": {
            "kernel": [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ],
            "explanation": "The box blur filter smooths the image by averaging the pixel values within a rectangular neighborhood.",
            "example": "Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nOutput: [[12, 18, 24], [30, 36, 42], [48, 54, 60]]"
        }
    }
    return filters

def main():
    st.set_page_config(page_title="Convolution Playground", layout="wide")
    st.title("Convolution Playground")
    st.markdown("""
    Welcome to the Convolution Playground! This interactive tool allows you to explore and understand the concept of convolution, which is widely used in computer vision and image processing.

    ## How to Use the Tool
    1. Upload an image using the file uploader in the sidebar.
    2. Select a filter from the dropdown menu in the sidebar.
    3. Click the "Apply Filter" button to perform the convolution operation.
    4. Observe the original image and the resulting filtered image side by side.
    5. Experiment with different filters to understand how convolution affects the image.

    Let's dive in and explore the world of convolution!
    """)

    # Sidebar controls
    st.sidebar.title("Controls")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    selected_filter = st.sidebar.selectbox("Select a filter:", list(get_filters().keys()))

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2).astype(np.uint8)
        st.session_state.image = image_array
    else:
        if 'image' not in st.session_state:
            st.warning("Please upload an image first.")
            return

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.image, use_column_width=True)

    with col2:
        st.subheader("Filtered Image")
        if st.sidebar.button("Apply Filter"):
            filters = get_filters()
            kernel = np.array(filters[selected_filter]["kernel"])
            filtered_image = convolve(st.session_state.image, kernel)
            st.image(filtered_image, use_column_width=True)
        else:
            st.empty()

    # Selected filter details
    st.subheader("Selected Filter")
    filters = get_filters()
    st.write(f"**Filter:** {selected_filter}")
    st.write(f"**Kernel:**")
    st.code(np.array(filters[selected_filter]["kernel"]))
    st.write(f"**Explanation:** {filters[selected_filter]['explanation']}")
    st.write(f"**Example:**")
    st.code(filters[selected_filter]["example"])

if __name__ == "__main__":
    main()