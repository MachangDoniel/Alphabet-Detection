def plot_matched_templates(input_image, recognized_alphabets, match_coords, preprocessed_templates):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    for i, (alphabet, match_val) in enumerate(recognized_alphabets):
        if alphabet is not None:
            if "small_" in alphabet:
                alphabet = alphabet.replace("small_", "")
            if match_coords[i]:
                template = preprocessed_templates[alphabet][0]
                template_height, template_width = template.shape
                x, y, _, _ = match_coords[i]
                ax.imshow(template, extent=[x, x + template_width, y, y + template_height], alpha=0.5)

    plt.show()