
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# FETCH AND STORING OF IMAGE PROCESSING OUTPUTS
# ==========================================

IMAGE_PATH = r"C:\Users\priya\OneDrive\Desktop\IPA Assignment\images\original.jpeg"  
OUTPUT_DIR = r"C:\Users\priya\OneDrive\Desktop\IPA Assignment\outputs"              
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load original image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}. Check the file path."
    )

# Convert image formats
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def save_image(filename, img):
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)
    print(f"Saved: {filename}")


def display_image(img, title, color=False):
    plt.figure(figsize=(7, 6))

    if color:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def compare_images(images, titles, heading):
    fig, axes = plt.subplots(
        1, len(images), figsize=(5 * len(images), 5)
    )

    if len(images) == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)

        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(heading)
    plt.tight_layout()
    plt.show()


# ==========================================
# TASK 1: ORIGINAL IMAGE ANALYSIS
# ==========================================

def task1():
    height, width, channels = image.shape
    size = os.path.getsize(IMAGE_PATH)

    print("\n--- ORIGINAL IMAGE ANALYSIS ---")
    print("Width:", width, "pixels")
    print("Height:", height, "pixels")
    print("Dimensions:", width, "x", height)
    print("Channels:", channels)
    print("File size:", round(size / 1024, 2), "KB")

    display_image(image, "Original Image", color=True)
    save_image("original.png", image)


# ==========================================
# TASK 2: GRAYSCALE CONVERSION
# ==========================================

def task2():
    save_image("grayscale.png", gray)

    compare_images(
        [image_rgb, gray],
        ["Original Image", "Grayscale Image"],
        "Grayscale Conversion"
    )


# ==========================================
# TASK 3: IMAGE NEGATIVE
# ==========================================

def task3():
    negative = 255 - gray

    save_image("negative.png", negative)

    compare_images(
        [gray, negative],
        ["Original Grayscale", "Negative Image"],
        "Image Negative"
    )


# ==========================================
# TASK 4: GAMMA CORRECTION
# ==========================================

def task4():
    normalized = gray.astype(np.float32) / 255.0
    gamma_values = [0.5, 1.0, 2.0]
    results = [gray]

    for gamma in gamma_values:
        corrected = np.power(normalized, gamma)
        corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)

        results.append(corrected)
        save_image(f"gamma_{gamma}.png", corrected)

    compare_images(
        results,
        ["Original"] + [f"Gamma = {g}" for g in gamma_values],
        "Gamma Correction"
    )


# ==========================================
# TASK 5: CONTRAST STRETCHING
# ==========================================

def task5():
    minimum = int(gray.min())
    maximum = int(gray.max())

    if maximum > minimum:
        stretched = (
            (gray.astype(np.float32) - minimum)
            * 255 / (maximum - minimum)
        )
    else:
        stretched = gray.copy()

    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    save_image("contrast_stretched.png", stretched)

    compare_images(
        [gray, stretched],
        ["Original Grayscale", "Contrast Stretched"],
        "Contrast Stretching"
    )


# ==========================================
# TASK 6: HISTOGRAM EQUALIZATION
# ==========================================

def task6():
    equalized = cv2.equalizeHist(gray)

    save_image("equalized.png", equalized)

    compare_images(
        [gray, equalized],
        ["Original Grayscale", "Histogram Equalized"],
        "Histogram Equalization"
    )


# ==========================================
# TASK 7: DISPLAY HISTOGRAMS
# ==========================================

def task7():
    equalized = cv2.equalizeHist(gray)

    hist1 = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist1)
    plt.title("Original Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.xlim(0, 255)

    plt.subplot(1, 2, 2)
    plt.plot(hist2)
    plt.title("Equalized Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.xlim(0, 255)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms.png"))
    plt.show()


# ==========================================
# TASK 8: GAUSSIAN SMOOTHING
# ==========================================

def task8():
    # CHANGE KERNEL SIZE OR SIGMA HERE IF REQUIRED
    gaussian3 = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.8)
    gaussian7 = cv2.GaussianBlur(gray, (7, 7), sigmaX=1.5)

    save_image("gaussian_3x3.png", gaussian3)
    save_image("gaussian_7x7.png", gaussian7)

    compare_images(
        [gray, gaussian3, gaussian7],
        ["Original", "Gaussian 3x3", "Gaussian 7x7"],
        "Gaussian Smoothing"
    )


# ==========================================
# TASK 9: SOBEL EDGE DETECTION
# ==========================================

def task9():
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize outputs for display
    sobel_x_display = cv2.normalize(
        np.abs(sobel_x), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    sobel_y_display = cv2.normalize(
        np.abs(sobel_y), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    magnitude_display = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    save_image("sobel_x.png", sobel_x_display)
    save_image("sobel_y.png", sobel_y_display)
    save_image("sobel_magnitude.png", magnitude_display)

    compare_images(
        [gray, sobel_x_display, sobel_y_display, magnitude_display],
        ["Original", "Sobel X", "Sobel Y", "Gradient Magnitude"],
        "Sobel Edge Detection"
    )


# ==========================================
# RUN ALL OPERATIONS
# ==========================================

def run_all():
    for operation in [
        task1, task2, task3, task4, task5,
        task6, task7, task8, task9
    ]:
        operation()


# ==========================================
# MENU
# ==========================================

def menu():
    operations = {
        "1": task1,
        "2": task2,
        "3": task3,
        "4": task4,
        "5": task5,
        "6": task6,
        "7": task7,
        "8": task8,
        "9": task9,
        "10": run_all
    }

    while True:
        print("\n========================================")
        print("      IMAGE PROCESSING ASSIGNMENT")
        print("========================================")
        print("1. Original Image Analysis")
        print("2. Grayscale Conversion")
        print("3. Image Negative")
        print("4. Gamma Correction")
        print("5. Contrast Stretching")
        print("6. Histogram Equalization")
        print("7. Display Histograms")
        print("8. Gaussian Smoothing")
        print("9. Sobel Edge Detection")
        print("10. Run All Operations")
        print("0. Exit")
        print("========================================")

        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print("Program terminated.")
            break

        if choice not in operations:
            print("Invalid choice. Please select 0-10.")
            continue

        try:
            operations[choice]()
        except Exception as error:
            print("Error:", error)

        input("\nPress Enter to return to the menu...")


# ==========================================
# PROGRAM START
# ==========================================

if __name__ == "__main__":
    menu()