import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt

# Function to display intermediate images
def show_intermediate_images(original):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(original, (5, 5), 0)
    
    # Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours based on x-coordinate of bounding rectangle
    contours_with_rects = [(contour, cv2.boundingRect(contour)) for contour in contours]
    contours_with_rects.sort(key=lambda cr: cr[1][0])  # Sort by x-coordinate
    
    # Unpack sorted contours and their bounding rectangles
    sorted_contours = [cr[0] for cr in contours_with_rects]
    
    # Create images to draw contours on
    contours_images = [original.copy() for _ in sorted_contours]
    for i, contour in enumerate(sorted_contours):
        cv2.drawContours(contours_images[i], [contour], -1, (0, 255, 0), 2)
    
    # Plot original, blurred, grayscale, and threshold images
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blur")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(thresh, cmap='gray')
    plt.title("Thresholding")
    plt.axis('off')
    
    plt.show()

    # Plot contours
    num_contours = len(sorted_contours)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_contours + cols - 1) // cols  # Calculate the number of rows needed

    plt.figure(figsize=(15, 5 * rows))
    for i, contour_image in enumerate(contours_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Contour {i + 1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()




# Function to segment the image
def segment_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found.")
        return [], image, []

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    image_with_boxes = image.copy()
    segments = []
    bounding_boxes = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 0 and h > 0:
            segment = image[y:y+h, x:x+w]
            if segment.size > 0:
                segments.append(segment)
                bounding_boxes.append((x, y, w, h))
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if not segments:
        print("No valid segments found.")
    
    # Show intermediate images
    # show_intermediate_images(image, blurred, gray, thresh, image_with_boxes)
    
    return segments, image_with_boxes, bounding_boxes

# Function to preprocess each segment of the image
def preprocess_image(segment):
    if segment is None or not isinstance(segment, np.ndarray):
        print("Segment is not a valid NumPy array.")
        return None
    resized = cv2.resize(segment, (100, 100))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray

# Function to preprocess templates from multiple directories
def preprocess_templates(template_dirs):
    preprocessed_templates = {}
    for template_dir in template_dirs:
        for alphabet_file in os.listdir(template_dir):
            template_path = os.path.join(template_dir, alphabet_file)
            template = cv2.imread(template_path)
            if template is not None:
                segments, _, _ = segment_image(template)  # Unpack the values but ignore the third one
                if segments:
                    template_preprocessed = [preprocess_image(segment) for segment in segments if preprocess_image(segment) is not None]
                    preprocessed_templates[alphabet_file] = template_preprocessed
            else:
                print(f"Error reading image file: {template_path}")
    return preprocessed_templates

# Function to compare images using cross-correlation
def compare_images(input_processed, template):
    input_height, input_width = input_processed.shape
    template_height, template_width = template.shape
    template_mean = np.mean(template)
    max_correlation_value = -float('inf')

    for y in range(input_height - template_height + 1):
        for x in range(input_width - template_width + 1):
            roi = input_processed[y:y+template_height, x:x+template_width]
            roi_mean = np.mean(roi)
            denominator = np.sqrt(np.sum((roi - roi_mean)**2) * np.sum((template - template_mean)**2))
            if denominator > 0:
                correlation_value = np.sum((roi - roi_mean) * (template - template_mean)) / denominator
                if correlation_value > max_correlation_value:
                    max_correlation_value = correlation_value

    return max_correlation_value

# Function to recognize alphabets from the input image
def recognize_alphabets(input_image, preprocessed_templates):
    segments, image_with_boxes, bounding_boxes = segment_image(input_image)  # Get segments and image with bounding boxes

    show_intermediate_images(input_image)
    recognized_alphabets = []
    match_coords = []

    for idx, segment in enumerate(segments):
        input_processed = preprocess_image(segment)
        max_match_val = -1
        recognized_alphabet = None
        best_match_coord = None

        for alphabet_file, templates in preprocessed_templates.items():
            for template in templates:
                match_val = compare_images(input_processed, template)
                if match_val > max_match_val:
                    max_match_val = match_val
                    recognized_alphabet = os.path.splitext(alphabet_file)[0]  # Extract the alphabet from the filename
                    best_match_coord = bounding_boxes[idx]

        recognized_alphabets.append((recognized_alphabet, max_match_val))
        match_coords.append(best_match_coord)

    return recognized_alphabets, match_coords, image_with_boxes

# Function to plot the matched templates over the selected image
def plot_matched_templates(input_image, recognized_alphabets, match_coords, preprocessed_templates):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    for i, (alphabet, match_val) in enumerate(recognized_alphabets):
        if alphabet not in preprocessed_templates:
            continue
        if alphabet is not None:
            if "small_" in alphabet:
                alphabet = alphabet.replace("small_", "")
            if match_coords[i]:
                template = preprocessed_templates[alphabet][0]
                template_height, template_width = template.shape
                x, y, _, _ = match_coords[i]
                ax.imshow(template, extent=[x, x + template_width, y, y + template_height], alpha=0.5)

    plt.show()

# Function to handle image selection
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        input_image_path.set(file_path)
        load_image(file_path)

# Function to load and display the selected image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        # Optionally resize the image to fit a specific display area if needed
        aspect_ratio = image.shape[1] / image.shape[0]
        max_height = 200
        target_width = int(max_height * aspect_ratio)
        image = cv2.resize(image, (target_width, max_height))
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
        image_label.config(image=photo)
        image_label.image = photo
    else:
        print("Error loading image.")

# Function to detect and display the recognized alphabets
def detect_alphabet():
    input_path = input_image_path.get()
    if input_path:
        input_image = cv2.imread(input_path)
        if input_image is None:
            print("Error loading image.")
            return

        preprocessed_templates = preprocess_templates(template_dirs)
        recognized_alphabets, match_coords, image_with_boxes = recognize_alphabets(input_image, preprocessed_templates)

        result_text = "Recognized Alphabets:\n"
        for i, (alphabet, match_val) in enumerate(recognized_alphabets):
            if alphabet is not None:
                if "small_" in alphabet:
                    alphabet = alphabet.replace("small_", "")
                result_text += alphabet
                
                # Draw the matched template over the selected alphabet
                if match_coords[i]:
                    x, y, w, h = match_coords[i]
                    cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 0, 128), 2)  # Navy blue color

        # Resize the image to fit the display area if necessary
        aspect_ratio = input_image.shape[1] / input_image.shape[0]
        max_height = 200
        target_width = int(max_height * aspect_ratio)
        resized_image = cv2.resize(input_image, (target_width, max_height))
        
        # Convert image to RGB and display it
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
        image_label.config(image=photo)
        image_label.image = photo
        
        result_label.config(text=result_text)
        
        # Plot matched templates in an external plot
        # plot_matched_templates(input_image, recognized_alphabets, match_coords, preprocessed_templates)
    else:
        print("Please select an input image.")

# Create the main application window
root = tk.Tk()
root.title("Alphabet Detection")

# Create a frame for the title
title_frame = tk.Frame(root)
title_frame.pack(pady=10)

# Add a label for the title
title_label = tk.Label(title_frame, text="Alphabet Detection", font=("Arial", 18))
title_label.pack()

# Create a frame for the input image selection
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Add a label and button for selecting the input image
input_label = tk.Label(input_frame, text="Select Input Image:")
input_label.grid(row=0, column=0)

input_image_path = tk.StringVar()
input_entry = tk.Entry(input_frame, textvariable=input_image_path, width=40)
input_entry.grid(row=0, column=1)

browse_button = tk.Button(input_frame, text="Browse", command=select_image)
browse_button.grid(row=0, column=2)

# Create a frame for displaying the selected image
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

image_label = tk.Label(image_frame)
image_label.pack()

# Create a frame for the result
result_frame = tk.Frame(root)
result_frame.pack(pady=10)

result_label = tk.Label(result_frame, text="", font=("Arial", 14))
result_label.pack()

# Create a frame for the detect button
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

detect_button = tk.Button(button_frame, text="Detect Alphabets", command=detect_alphabet)
detect_button.pack()

# Define template directories (update with actual paths)
template_dirs = ['datasets/templates/small', 'datasets/templates/capital']

# Start the main event loop
root.mainloop()
