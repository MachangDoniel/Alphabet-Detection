import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt

# Function to segment the image
# def segment_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
#     segments = [image[y:y+h, x:x+w] for x, y, w, h in [cv2.boundingRect(c) for c in contours]]
#     return segments

# py special function
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Copy the original image to draw bounding boxes
    image_with_boxes = image.copy()
    segments = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 0 and h > 0:  # Ensure valid width and height
            segment = image[y:y+h, x:x+w]
            if segment.size > 0:  # Ensure segment is not empty
                segments.append(segment)
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return segments, image_with_boxes



# Function to preprocess each segment of the image
# def preprocess_image(segment):
#     resized = cv2.resize(segment, (100, 100))
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     return gray

# py special function
def preprocess_image(segment):
    if segment is None or not isinstance(segment, np.ndarray):
        print("Segment is not a valid NumPy array.")
        return None
    resized = cv2.resize(segment, (100, 100))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


# Function to preprocess templates from multiple directories
# def preprocess_templates(template_dirs):
#     preprocessed_templates = {}
#     for template_dir in template_dirs:
#         for alphabet_file in os.listdir(template_dir):
#             template_path = os.path.join(template_dir, alphabet_file)
#             template = cv2.imread(template_path)
#             if template is not None:
#                 segments = segment_image(template)
#                 if segments:
#                     template_preprocessed = [preprocess_image(segment) for segment in segments]
#                     preprocessed_templates[alphabet_file] = template_preprocessed
#     return preprocessed_templates

# py special function
def preprocess_templates(template_dirs):
    preprocessed_templates = {}
    for template_dir in template_dirs:
        for alphabet_file in os.listdir(template_dir):
            template_path = os.path.join(template_dir, alphabet_file)
            template = cv2.imread(template_path)
            if template is not None:
                segments, _ = segment_image(template)
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
# def recognize_alphabets(input_image, preprocessed_templates):
#     segments = segment_image(input_image)
#     recognized_alphabets = []

#     for segment in segments:
#         input_processed = preprocess_image(segment)
#         max_match_val = -1
#         recognized_alphabet = None

#         for alphabet_file, templates in preprocessed_templates.items():
#             for template in templates:
#                 match_val = compare_images(input_processed, template)
#                 if match_val > max_match_val:
#                     max_match_val = match_val
#                     recognized_alphabet = os.path.splitext(alphabet_file)[0]  # Extract the alphabet from the filename

#         recognized_alphabets.append((recognized_alphabet, max_match_val))

#     return recognized_alphabets

# py special function
# Function to recognize alphabets from the input image
def recognize_alphabets(input_image, preprocessed_templates):
    segments, _ = segment_image(input_image)  # Get segments and image with bounding boxes
    recognized_alphabets = []

    for segment in segments:
        input_processed = preprocess_image(segment)
        max_match_val = -1
        recognized_alphabet = None

        for alphabet_file, templates in preprocessed_templates.items():
            for template in templates:
                match_val = compare_images(input_processed, template)
                if match_val > max_match_val:
                    max_match_val = match_val
                    recognized_alphabet = os.path.splitext(alphabet_file)[0]  # Extract the alphabet from the filename

        recognized_alphabets.append((recognized_alphabet, max_match_val))

    return recognized_alphabets


# Function to display debugging images
# def show_debug_images(test_image, recognized_alphabets, expected_alphabet, image_name):
#     # Display the test image
#     plt.figure(figsize=(2, 1))
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
#     plt.title(f"Image: {image_name}\nExpected: {expected_alphabet}")
#     plt.axis('off')
    
#     # Combine recognized segments into a single string
#     combined_result = ''.join([recognized_alphabet for recognized_alphabet, match_val in recognized_alphabets if recognized_alphabet is not None])
    
#     # Display combined result as a separate image
#     plt.figure(figsize=(3, 1))
#     plt.text(0.5, 0.7, f"Combined Result: {combined_result}\nExpected: {expected_alphabet}",
#              fontsize=14, ha='center', va='center', color='green' if combined_result.replace(' ', '') == expected_alphabet.replace(' ', '') else 'red',
#              bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
#     plt.title("Combined Recognition Result")
#     plt.axis('off')
#     plt.show()

# py special function
def show_debug_images(test_image, recognized_alphabets, expected_alphabet, image_name):
    # Display the test image with bounding boxes
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    image_with_boxes = segment_image(test_image)[1]
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title(f"Image: {image_name}\nExpected: {expected_alphabet}")
    plt.axis('off')
    
    # Combine recognized segments into a single string
    combined_result = ''.join([recognized_alphabet for recognized_alphabet, match_val in recognized_alphabets if recognized_alphabet is not None])
    
    # Display combined result as a separate image
    plt.figure(figsize=(8, 2))
    plt.text(0.5, 0.7, f"Combined Result: {combined_result}\nExpected: {expected_alphabet}",
             fontsize=14, ha='center', va='center', color='green' if combined_result.replace(' ', '') == expected_alphabet.replace(' ', '') else 'red',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    plt.title("Combined Recognition Result")
    plt.axis('off')
    plt.show()


# Function to test accuracy and display results
def test_accuracy(test_dir, preprocessed_templates):
    total_images = 0
    correct_images = 0

    for test_file in os.listdir(test_dir):
        test_image_path = os.path.join(test_dir, test_file)
        test_image = cv2.imread(test_image_path)

        if test_image is not None:
            recognized_alphabets = recognize_alphabets(test_image, preprocessed_templates)
            expected_alphabet = os.path.splitext(test_file)[0].split('-')[0].strip()

            combined_result = ''.join([recognized_alphabet for recognized_alphabet, match_val in recognized_alphabets if recognized_alphabet is not None])
            
            if combined_result.replace(' ', '') == expected_alphabet.replace(' ', ''):
                correct_images += 1

            total_images += 1

            # Show debugging images
            show_debug_images(test_image, recognized_alphabets, expected_alphabet, test_file)

    overall_accuracy = (correct_images / total_images) * 100 if total_images > 0 else 0

    plt.figure(figsize=(6, 2))
    plt.text(0.5, 0.5, f"Overall Accuracy: {overall_accuracy:.2f}%",
            fontsize=20, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    plt.title("Overall Accuracy")
    plt.axis('off')
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
    aspect_ratio = image.shape[1] / image.shape[0]
    max_height = 200
    target_width = int(max_height * aspect_ratio)
    image = cv2.resize(image, (target_width, max_height))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
    image_label.config(image=photo)
    image_label.image = photo

# Function to detect and display the recognized alphabets
def detect_alphabet():
    input_path = input_image_path.get()
    if input_path:
        input_image = cv2.imread(input_path)
        preprocessed_templates = preprocess_templates(template_dirs)
        recognized_alphabets = recognize_alphabets(input_image, preprocessed_templates)

        result_text = "Recognized Alphabets:\n"
        for i, (alphabet, match_val) in enumerate(recognized_alphabets):
            if alphabet is not None:
                if "small_" in alphabet:
                    alphabet = alphabet.replace("small_", "")
                result_text += alphabet
        
        result_label.config(text=result_text)
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

# Add a label for displaying the selected image
image_label = tk.Label(image_frame)
image_label.pack()

# Create a frame for displaying the results
result_frame = tk.Frame(root)
result_frame.pack(pady=10)

# Add a label for displaying the results
result_label = tk.Label(result_frame, text="Recognized Alphabets:")
result_label.pack()

# Add a button to perform alphabet detection
detect_button = tk.Button(root, text="Detect Alphabet", command=detect_alphabet)
detect_button.pack(pady=10)

# Directories containing template alphabet images
template_dirs = ['datasets/templates/small', 'datasets/templates/capital']  # Adjust this directory as needed

# Start the main event loop
root.mainloop()
