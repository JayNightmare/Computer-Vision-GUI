import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_measurements_table(measurements):
    """
    Display a table of measurement descriptors (Label, Area, Perimeter, Bounding Box, Centroid).
    """
    # Define column headers
    col_labels = ["Label", "Area", "Perimeter", "Bounding Box", "Centroid"]
    
    # Prepare table rows
    cell_text = []
    for m in measurements:
        row = [
            m['Label'],
            m['Area'],
            f"{m['Perimeter']:.1f}",
            m['BBox'],
            (f"{m['Centroid'][0]:.2f}", f"{m['Centroid'][1]:.2f}")
        ]
        cell_text.append(row)
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_axis_off()  # Hide axes
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    
    plt.title("Object Measurements")
    plt.tight_layout()
    plt.show()

def main():
    # 1) LOAD GRAYSCALE IMAGE
    img_gray = cv2.imread('images/Circle.png', cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error: Could not load image. Check 'your_image.png' path.")
        return

    # 2) OTSU THRESHOLDING
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 3) CONNECTED COMPONENTS (8-CONNECTIVITY)
    # This returns: num_labels, labels_im, stats, centroids.
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Create a color-labeled image for visualization.
    label_hue = np.uint8(179 * labels_im / (num_labels if num_labels > 1 else 1))
    blank_ch = 255 * np.ones_like(label_hue)
    colored_labels = cv2.merge([label_hue, blank_ch, blank_ch])
    colored_labels = cv2.cvtColor(colored_labels, cv2.COLOR_HSV2BGR)
    colored_labels[label_hue == 0] = 0  # Background set to black

    # 4) CALCULATE MEASUREMENTS & CREATE AN ANNOTATED IMAGE (LABEL NUMBERS ONLY)
    measurements = []
    annotated = colored_labels.copy()  # Create a copy for annotation
    for label_id in range(1, num_labels):  # Skip label 0 (background)
        x, y, w, h, area = stats[label_id]
        cx, cy = centroids[label_id]  # Centroid (floating point)
        
        # Get perimeter using contours.
        region_mask = (labels_im == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = 0.0
        if contours:
            perimeter = cv2.arcLength(contours[0], closed=True)
        
        measurements.append({
            'Label': label_id,
            'Area': area,
            'Perimeter': perimeter,
            'BBox': (x, y, w, h),
            'Centroid': (cx, cy)
        })
        
        # Draw bounding box in white.
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 255, 255), 2)
        # Put label number in red (using a moderate font size).
        text_str = f"{label_id}"
        text_pos = (x, y - 5)
        cv2.putText(annotated, text_str, text_pos,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255),  # Red color
                    thickness=1,
                    lineType=cv2.LINE_AA)

    # Convert BGR images to RGB for proper display in matplotlib.
    colored_rgb = cv2.cvtColor(colored_labels, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # 5) DISPLAY THE IMAGES SEPARATELY
    # Original Grayscale Image
    plt.figure(figsize=(6, 6))
    plt.imshow(img_gray, cmap='gray')
    plt.title("1) Original Grayscale Image")
    plt.axis('off')

    # Binary Image
    plt.figure(figsize=(6, 6))
    plt.imshow(binary, cmap='gray')
    plt.title("2) Binary Image (Otsu Threshold)")
    plt.axis('off')

    # Color-Labeled Image (No Annotation)
    plt.figure(figsize=(6, 6))
    plt.imshow(colored_rgb)
    plt.title("3) Color-Labeled Regions")
    plt.axis('off')

    # Annotated Image (Label Numbers Only, original size)
    plt.figure(figsize=(6, 6))
    plt.imshow(annotated_rgb)
    plt.title("4) Annotated Regions (Label Numbers)")
    plt.axis('off')

    plt.show()

    # 6) DISPLAY THE MEASUREMENTS TABLE
    show_measurements_table(measurements)

if __name__ == "__main__":
    main()
