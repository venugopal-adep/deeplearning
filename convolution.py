import streamlit as st
import numpy as np
import plotly.graph_objects as go

def convolve(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

def plot_image(image, title):
    fig = go.Figure(data=go.Heatmap(z=image, colorscale='Viridis'))
    fig.update_layout(title=title, xaxis_title='Width', yaxis_title='Height')
    return fig

def main():
    st.title("Convolution Playground")
    st.write("**Developed by : Venugopal Adep**")
    st.markdown("""
    Welcome to the Convolution Playground! This interactive tool allows you to explore and understand the concept of convolution, which is widely used in computer vision and image processing.

    ## What is Convolution?
    Convolution is a mathematical operation that combines two functions to produce a third function. In the context of image processing, convolution is used to apply filters or kernels to an image to extract features or perform operations like blurring, edge detection, and sharpening.

    ## How to Use the Tool
    1. Select an image size and create a random image using the "Create Random Image" button.
    2. Define a kernel by entering values in the kernel matrix.
    3. Click the "Apply Convolution" button to perform the convolution operation.
    4. Observe the original image, kernel, and the resulting convolved image.
    5. Experiment with different image sizes and kernel values to understand how convolution affects the image.

    Let's dive in and explore the world of convolution!
    """)

    image_size = st.sidebar.slider("Select image size:", min_value=5, max_value=10, value=5)
    
    if st.sidebar.button("Create Random Image"):
        image = np.random.rand(image_size, image_size)
        st.session_state.image = image
    else:
        if 'image' not in st.session_state:
            st.warning("Please create a random image first.")
            return

    st.subheader("Original Image")
    fig_image = plot_image(st.session_state.image, "Original Image")
    st.plotly_chart(fig_image)

    st.subheader("Kernel")
    kernel_size = 3
    kernel_values = []
    for i in range(kernel_size):
        row = st.sidebar.text_input(f"Enter values for kernel row {i+1} (comma-separated):", value="1,1,1")
        kernel_values.append(list(map(float, row.split(','))))
    kernel = np.array(kernel_values)

    fig_kernel = plot_image(kernel, "Kernel")
    st.plotly_chart(fig_kernel)

    if st.sidebar.button("Apply Convolution"):
        convolved_image = convolve(st.session_state.image, kernel)
        st.subheader("Convolved Image")
        fig_convolved = plot_image(convolved_image, "Convolved Image")
        st.plotly_chart(fig_convolved)

    st.markdown("""
    ## Example Kernels
    Here are some example kernels commonly used in image processing:

    - Blurring Kernel:
      ```
      1 1 1
      1 1 1
      1 1 1
      ```
    - Edge Detection Kernel (Horizontal):
      ```
      -1 -1 -1
       0  0  0
       1  1  1
      ```
    - Edge Detection Kernel (Vertical):
      ```
      -1 0 1
      -1 0 1
      -1 0 1
      ```
    - Sharpening Kernel:
      ```
       0 -1  0
      -1  5 -1
       0 -1  0
      ```

    Try experimenting with these kernels and observe how they affect the image!
    """)

if __name__ == "__main__":
    main()
