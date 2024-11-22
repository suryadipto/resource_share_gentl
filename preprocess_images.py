import os
import cv2
import numpy as np
import pandas as pd

# Define the path to the dataset folder
input_file_path = "./data/data_as_jpg"  # jpg images folder
output_file_path = "./data/processed_data"  # images with contours

# Create the output folder if it doesn't exist
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# create a list to store rectangle coordinates
data_list = []

# Traverse the dataset folder
for root, _, files in os.walk(input_file_path):
    for file in files:
        if file.endswith('.jpg'):  # Check if the file is a .jpg file
            image_path = os.path.join(root, file)

            # Step 1: Read the image and convert to grayscale
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Step 2: Apply Otsu's thresholding, returns (threshold,binary image)
            _, mask = cv2.threshold(src=original_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Step 3: Find contours within the mask
            contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            # Step 4: Get the largest contour with area
            largest_contour = max(contours, key=cv2.contourArea)

            # Step 5: Calculate bounding rectangle around the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Step 6: Draw the outer rectangle on the original image
            cv2.rectangle(
                img=original_image,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(255, 255, 255),  # white border
                thickness=2
                )

            # Step 7: Fit an ellipse inside the bounding rectangle
            center = (x + w // 2, y + h // 2)  # Center of the rectangle
            axes = (w // 2, h // 2)  # Semi-major and semi-minor axes
            angle = 0  # No rotation
            # mention start_angle as 0 and end_angle as 360 to draw a complete ellipse
            cv2.ellipse(
                img=original_image,
                center=center,
                axes=axes,
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=(255, 255, 255),
                thickness=2
                )

            # Step 8: Draw a rectangle completely inside the ellipse
            inner_rect_w = int(axes[0] * np.sqrt(2))  # sqrt(2) * semi major axis = width
            inner_rect_h = int(axes[1] * np.sqrt(2))  # sqrt(2) * semi minor axis = height

            # Calculate the top-left corner of the inner rectangle
            inner_rect_x = center[0] - inner_rect_w // 2
            inner_rect_y = center[1] - inner_rect_h // 2

            cv2.rectangle(
                img=original_image,
                pt1=(inner_rect_x, inner_rect_y),
                pt2=(inner_rect_x + inner_rect_w, inner_rect_y + inner_rect_h),
                color=(255, 255, 255),
                thickness=2
                )

            # Step 9: Store image wise rectangle coordinates
            data_list.append(
                {"patient_id": file.split(".jpg")[0], "outer_rect_coordinates": (x, y, x + w, y + h),
                 "inner_rect_coordinates": (
                     inner_rect_x, inner_rect_y, inner_rect_x + inner_rect_w, inner_rect_y + inner_rect_h)}
                )

            # Step 10: Save the processed image to the output folder
            output_path = os.path.join(output_file_path, file)
            cv2.imwrite(output_path, original_image)
            print(f"Saved processed image to: {output_path}")

# Step 11: Save the coordinates in a csv with image-id as the index
processed_data_dataframe = pd.DataFrame(data_list)
processed_data_dataframe.to_csv("./data/processed_data.csv", index=False)
