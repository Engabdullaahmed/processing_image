# # Image Processing GUI with Python
## Overview
This project is a graphical user interface (GUI) application for image processing using Python. It provides functionalities to apply various image filters and transformations on single images or entire datasets. Users can load images, apply filters like negative, Sobel, Gaussian, and others, and save the processed images.
## output
-- procssing single image
![image](https://github.com/user-attachments/assets/9dcf4348-2579-4822-ab68-599ec8daaebb)
![image](https://github.com/user-attachments/assets/74da795d-8208-4d40-bd31-eeb60624a7a5)
![image](https://github.com/user-attachments/assets/9aec68db-bdc1-4e67-bb68-b8ebca783e79)
-- procssing dataset 
![image](https://github.com/user-attachments/assets/8a178da7-fdd0-4801-85c7-29b202f560cd)
![image](https://github.com/user-attachments/assets/cb2d7da0-1117-4314-a877-5973c7606d32)
![Uploading image.png…]()
![Uploading image.png…]()

## Features
- Load single images or a folder of images.
- Apply various filters and transformations:
  - Negative filter
  - Power law transformation
  - Log transformation
  - Histogram equalization
  - Histogram matching
  - Sobel and Prewitt edge detection
  - Gaussian filter
  - Average, median, minimum, and maximum filters
  - Brightness increase/decrease
- Reset images to the original state.
- Save processed images.
- Add visual lines to images (e.g., diagonal lines).
- Interactive interface using Tkinter.

## Technologies and Libraries Used
### Libraries
1. **OpenCV (cv2)**:
   - For image processing and transformations.
   - Example functions:
     - `cvtColor` for color space conversion.
     - `resize` for resizing images.
   - Documentation: [OpenCV Documentation](https://docs.opencv.org/)

2. **Pillow (PIL)**:
   - For handling image formats and conversions.
   - Example functions:
     - `Image.open` to load images.
     - `Image.fromarray` to convert NumPy arrays to images.
   - Documentation: [Pillow Documentation](https://pillow.readthedocs.io/)

3. **NumPy**:
   - For efficient image data manipulation.
   - Example usage:
     - `np.clip` to handle pixel intensity constraints.
     - `np.bincount` for histogram calculations.
   - Documentation: [NumPy Documentation](https://numpy.org/)

4. **Tkinter**:
   - For GUI development.
   - Widgets used:
     - `Button` for actions.
     - `Canvas` for displaying images.
     - `Scrollbar` for navigating datasets.
   - Documentation: [Tkinter Documentation](https://docs.python.org/3/library/tkinter.html)

5. **Matplotlib** (optional):
   - For visualizing histogram data if needed.
   - Documentation: [Matplotlib Documentation](https://matplotlib.org/)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required libraries:
   ```bash
   pip install opencv-python pillow numpy matplotlib
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Usage
### Single Image Processing
1. Click **"Load Image"** to upload an image.
2. Choose and apply filters from the left and right panels.
3. Save the processed image using **"Save Image"**.

### Dataset Processing
1. Click **"Upload Folder"** to select a folder containing images.
2. Apply filters to all images in the dataset.
3. Save the processed images to a desired directory.

### Filters and Transformations
#### 1. Negative Filter
- Inverts pixel values to create a negative image.
- Formula: `NewPixel = 255 - OriginalPixel`

#### 2. Power Law Transformation
- Enhances image contrast.
- Formula: `NewPixel = c * (OriginalPixel ** gamma)`

#### 3. Log Transformation
- Highlights darker areas of an image.
- Formula: `NewPixel = c * log(1 + OriginalPixel)`

#### 4. Sobel and Prewitt Filters
- Detects edges in the image.
- Uses kernels to compute gradients in both horizontal and vertical directions.

#### 5. Gaussian Filter
- Smoothens the image using a Gaussian kernel.

#### 6. Histogram Equalization
- Improves the contrast by spreading pixel intensity values.

#### 7. Histogram Matching
- Adjusts the image histogram to match a target distribution.

#### 8. Average, Median, Maximum, Minimum Filters
- Applies respective statistical operations on local regions of the image.

#### 9. Brightness Adjustment
- Multiplies or divides pixel intensity values to increase or decrease brightness.

## Resources
1. OpenCV Tutorials: [Learn OpenCV](https://opencv.org/)
2. Pillow Documentation: [Pillow Docs](https://pillow.readthedocs.io/)
3. NumPy for Image Processing: [NumPy Tutorials](https://numpy.org/learn/)
4. Sobel and Prewitt Filters:
   - [Sobel Filter Explanation](https://en.wikipedia.org/wiki/Sobel_operator)
   - [Prewitt Filter Explanation](https://en.wikipedia.org/wiki/Prewitt_operator)
5. Gaussian Filter Basics: [Gaussian Filtering](https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm)
6. tkinter Basics:
   - [tkinter](https://www.geeksforgeeks.org/python-gui-tkinter/)
   - ((https://www.youtube.com/watch?v=NzEh3Dfa4Vg&list=PLSiLeKadTQ7nLJxpQo1-944miQKlheu-v))
   - ((https://www.youtube.com/watch?v=yQSEXcf6s2I&list=PLCC34OHNcOtoC6GglhF3ncJ5rLwQrLGnV))
  

