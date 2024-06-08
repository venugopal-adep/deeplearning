import streamlit as st
import numpy as np
import cv2
from PIL import Image

def apply_convolution(image, kernel, stride, padding):
    """
    Apply convolution operation to an image using OpenCV.
    """
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    result = cv2.filter2D(image, -1, kernel)
    return result

def create_kernel(kernel_type, kernel_size):
    """
    Create a kernel based on the selected kernel type and size.
    """
    if kernel_type == 'Box':
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    elif kernel_type == 'Gaussian':
        kernel = cv2.getGaussianKernel(kernel_size, 0)
        kernel = kernel * kernel.T
    elif kernel_type == 'Laplacian':
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    elif kernel_type == 'Sobel':
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    return kernel

# Streamlit interface
st.set_page_config(page_title='Convolution Operation Explorer', layout='wide')
st.title('Convolution Operation Explorer')
st.write('**Developed by : Venugopal Adep**')

# Sidebar
st.sidebar.title('Convolution Parameters')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])
kernel_type = st.sidebar.selectbox('Kernel Type', ['Box', 'Gaussian', 'Laplacian', 'Sobel'])
kernel_size = st.sidebar.slider('Kernel size', 1, 7, 3, step=2)
stride = st.sidebar.slider('Stride', 1, 5, 1)
padding = st.sidebar.slider('Padding', 0, 5, 0)

# Custom kernel input
st.sidebar.subheader('Custom Kernel')
custom_kernel = st.sidebar.text_area('Enter custom kernel (comma-separated values)')

# Apply convolution button
apply_button = st.sidebar.button('Apply Convolution')

# Main content
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create two columns for displaying images side by side
    col1, col2 = st.columns(2)
    
    # Display original image in the first column
    with col1:
        st.subheader('Original Image')
        st.image(image, use_column_width=True)
    
    # Create kernel
    if custom_kernel:
        kernel = np.array([[float(x.strip()) for x in row.split(',')] for row in custom_kernel.split('\n')], np.float32)
    else:
        kernel = create_kernel(kernel_type, kernel_size)

    # Display kernel
    st.subheader('Kernel')
    st.write(kernel)

    # Apply convolution
    if apply_button:
        conv_image = apply_convolution(image, kernel, stride, padding)
        
        # Display convolved image in the second column
        with col2:
            st.subheader('Convolved Image')
            st.image(conv_image, use_column_width=True, clamp=True)
# 0, 1, 0, 0, 1, 0, 0, 1, 0
# 0, 0, 0, 1, 1, 1, 0, 0, 0