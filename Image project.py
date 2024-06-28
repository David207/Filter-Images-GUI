import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, color, img_as_ubyte, filters

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        self.image_path = None
        self.original_image = None
        self.processed_image = None

        self.create_widgets()

    def create_widgets(self):
        # Load Image Button
        load_button = Button(self.root, text="Load Image", command=self.load_image)
        load_button.pack(pady=10)

        # Enhancement Options Dropdown
        options = [
            "Resizing",
            "Color Conversion (BGR to Grayscale)",
            "Color Elimination",
            "Color Channel Swapping to Red",
            "Color Channel Swapping to Blue",
            "Color Channel Swapping to Green",
            "Image Complementing",
            "Changing Brightness",
            "Brightness Color Change",
            "Plotting Image’s Histogram",
            "Lowering or Increasing Contrast",
            "Smoothing the Image",
            "Sharpening the Image",
            "Dilation",
            "Erosion",
            "Opening",
            "Closing",
            "Thresholding Segmentation (Global)",
            "Thresholding Segmentation (Adaptive)"
        ]

        self.selected_option = StringVar()
        self.selected_option.set(options[0])

        option_menu = OptionMenu(self.root, self.selected_option, *options)
        option_menu.pack(pady=10)

        # Process Button
        process_button = Button(self.root, text="Process", command=self.process_image)
        process_button.pack(pady=10)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if not hasattr(self, "panel"):
            self.panel = Label(self.root, image=image)
            self.panel.image = image
            self.panel.pack(side="bottom", padx=10, pady=10)
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def process_and_display(self, processed_image):
        self.display_image(processed_image)
        self.processed_image = processed_image

    def process_image(self):
        if self.original_image is not None:
            selected_option = self.selected_option.get()

            if selected_option == "Resizing":
                self.resize_image()

            elif selected_option == "Color Conversion (BGR to Grayscale)":
                self.convert_to_grayscale()

            elif selected_option == "Color Elimination":
                self.eliminate_color()

            elif selected_option == "Color Channel Swapping to Red":
                self.swap_color_channel(2)  # Red channel

            elif selected_option == "Color Channel Swapping to Blue":
                self.swap_color_channel(0)  # Blue channel

            elif selected_option == "Color Channel Swapping to Green":
                self.swap_color_channel(1)  # Green channel

            elif selected_option == "Image Complementing":
                self.complement_image()

            elif selected_option == "Changing Brightness":
                self.change_brightness()

            elif selected_option == "Brightness Color Change":
                self.change_brightness_color()

            elif selected_option == "Plotting Image’s Histogram":
                self.plot_histogram()

            elif selected_option == "Lowering or Increasing Contrast":
                self.adjust_contrast()

            elif selected_option == "Smoothing the Image":
                self.smooth_image()

            elif selected_option == "Sharpening the Image":
                self.sharpen_image()

            elif selected_option in ["Dilation", "Erosion", "Opening", "Closing"]:
                self.morphological_operations(selected_option.lower())

            elif selected_option.startswith("Thresholding Segmentation"):
                self.threshold_segmentation(selected_option)

    def resize_image(self):
        if self.original_image is not None:
            scale_factor = 0.5  # Adjust as needed
            resized_image = cv2.resize(self.original_image, None, fx=scale_factor, fy=scale_factor)
            self.process_and_display(resized_image)

    def convert_to_grayscale(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.process_and_display(gray_image)

    def eliminate_color(self):
        if self.original_image is not None:
            color_to_eliminate = 'green'  # Change as needed
            eliminated_image = self.original_image.copy()
            eliminated_image[:, :, self.get_channel(color_to_eliminate)] = 0
            self.process_and_display(eliminated_image)

    def swap_color_channel(self, channel):
        if self.original_image is not None:
            swapped_image = self.original_image.copy()
            swapped_image[:, :, (0, 1, 2)] = 0
            swapped_image[:, :, channel] = self.original_image[:, :, channel]
            self.process_and_display(swapped_image)

    def complement_image(self):
        if self.original_image is not None:
            complemented_image = 255 - self.original_image
            self.process_and_display(complemented_image)

    def change_brightness(self):
        if self.original_image is not None:
            brightness_factor = 1.5  # Adjust as needed
            brighter_image = cv2.convertScaleAbs(self.original_image, alpha=brightness_factor, beta=0)
            self.process_and_display(brighter_image)

    def change_brightness_color(self):
        if self.original_image is not None:
            color_to_change = 'red'  # Change as needed
            brightness_factor = 1.5  # Adjust as needed
            changed_image = self.original_image.copy()
            changed_image[:, :, self.get_channel(color_to_change)] = cv2.convertScaleAbs(
                self.original_image[:, :, self.get_channel(color_to_change)], alpha=brightness_factor, beta=0)
            self.process_and_display(changed_image)

    def plot_histogram(self):
        if self.original_image is not None:
            plt.hist(self.original_image.flatten(), bins=256, range=[0, 256], color='r', alpha=0.5)
            plt.title('Histogram')
            plt.show()

    def adjust_contrast(self):
        if self.original_image is not None:
            contrasted_image = exposure.rescale_intensity(self.original_image)
            self.process_and_display(contrasted_image)

    def smooth_image(self):
        if self.original_image is not None:
            smooth_type = 'average'  # Change as needed
            kernel_size = 5  # Adjust as needed
            smooth_image = self.apply_filter(smooth_type, kernel_size)
            self.process_and_display(smooth_image)

    def sharpen_image(self):
        if self.original_image is not None:
            sharpened_image = filters.unsharp_mask(self.original_image, radius=1, amount=1)
            self.process_and_display(sharpened_image)

    def morphological_operations(self, operation):
        if self.original_image is not None:
            kernel_size = 5  # Adjust as needed
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if operation == "dilation":
                processed_image = cv2.dilate(self.original_image, kernel, iterations=1)
            elif operation == "erosion":
                processed_image = cv2.erode(self.original_image, kernel, iterations=1)
            elif operation == "opening":
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
            elif operation == "closing":
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
            self.process_and_display(processed_image)

    def threshold_segmentation(self, option):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            if option == "Thresholding Segmentation (Global)":
                _, segmented_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
            elif option == "Thresholding Segmentation (Adaptive)":
                segmented_image = cv2.adaptiveThreshold(
                    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            self.process_and_display(segmented_image)

    def apply_filter(self, filter_type, kernel_size):
        if filter_type == 'average':
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
            return cv2.filter2D(self.original_image, -1, kernel)
        elif filter_type == 'median':
            return cv2.medianBlur(self.original_image, kernel_size)
        elif filter_type == 'max':
            return cv2.dilate(self.original_image, None, iterations=kernel_size)
        elif filter_type == 'min':
            return cv2.erode(self.original_image, None, iterations=kernel_size)

    def get_channel(self, color):
        channels = {'blue': 0, 'green': 1, 'red': 2}
        return channels[color]


if __name__ == "__main__":
    root = Tk()
    processor = ImageProcessor(root)
    root.mainloop()
