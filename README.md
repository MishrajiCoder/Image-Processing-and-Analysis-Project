# Image Processing Assignment

A Python-based image processing project developed using OpenCV, NumPy, and Matplotlib. This project demonstrates fundamental digital image processing techniques through a menu-driven program.

## Features

1. **Original Image Analysis** — Display the original image and analyze its properties.
2. **Grayscale Conversion** — Convert the original image into grayscale.
3. **Image Negative** — Generate the negative of an image.
4. **Gamma Correction** — Apply gamma correction using different gamma values (0.5, 1.0, and 2.0).
5. **Contrast Stretching** — Enhance image contrast.
6. **Histogram Equalization** — Improve image contrast using histogram equalization.
7. **Histogram Visualization** — Display image intensity histograms.
8. **Gaussian Smoothing** — Reduce image noise using Gaussian filters with different kernel sizes.
9. **Sobel Edge Detection** — Detect horizontal and vertical edges and calculate gradient magnitude.

## Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib

## Project Structure

```text
IPA Assignment/
├── images/
│   └── original.jpeg
├── outputs/
├── main.py
├── README.md
└── .venv/
```

## Installation

Clone the repository:

```bash
git clone "https://github.com/MishrajiCoder/Image-Processing-and-Analysis-Project"
cd "IPA Project"
```

Install the required libraries:

```bash
python -m pip install opencv-python numpy matplotlib
```

## How to Run

Execute the main Python program:

```bash
python main.py
```

A menu will appear in the terminal. Select an option to perform the corresponding image processing operation.

## Output

Processed images and analysis results are saved in the `outputs/` directory.

## Author

Priyansh Mishra
