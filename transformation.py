import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
# from display_image import display_image



    
def draw_bounding_boxes_and_transform(yolo_mat_file, image_path, H):
    """
    Draws bounding boxes on an image and transforms the image and bounding boxes using a homography matrix.
    
    Parameters:
    - yolo_mat_file: Path to the YOLO .mat file containing bounding box coordinates and object IDs.
    - image_path: Path to the image on which to draw the bounding boxes.
    - H: 3x3 Homography matrix for geometric transformation.
    
    Returns:
    - transformed_image: Transformed image with bounding boxes.
    - transformed_boxes: Transformed bounding box coordinates.
    """
    # Load the .mat file
    data = sio.loadmat(yolo_mat_file)
    xyxy = data['xyxy']
    ids = data['id']
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found or cannot be read: {image_path}")
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # # Draw the original bounding boxes on the image
    # for i in range(xyxy.shape[0]):
    #     xmin, ymin, xmax, ymax = xyxy[i]
    #     object_id = int(ids[i][0])  # Ensure proper indexing
    #     color = (255, 0, 0)  # Red
    #     image_rgb = cv2.rectangle(image_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)
    #     label = f'ID: {object_id}'
    #     cv2.putText(image_rgb, label, (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Apply the homography transformation to the image
    height, width = image.shape[:2]
    transformed_image = cv2.warpPerspective(image_rgb, H, (width, height))
    
    # Transform the bounding box coordinates using H
    transformed_boxes = []
    for i in range(xyxy.shape[0]):
        xmin, ymin, xmax, ymax = xyxy[i]
        
        # Corner points of the bounding box
        corners = np.array([[xmin, ymin, 1],
                            [xmax, ymin, 1],
                            [xmax, ymax, 1],
                            [xmin, ymax, 1]]).T  # Shape: (3, 4)
        
        # Transform corners
        transformed_corners = H @ corners  # Matrix multiplication
        transformed_corners /= transformed_corners[2]  # Normalize by the homogeneous coordinate
        transformed_corners = transformed_corners[:2].T  # Shape: (4, 2)
        
        # Get new bounding box (xmin, ymin, xmax, ymax)
        new_xmin = np.min(transformed_corners[:, 0])
        new_ymin = np.min(transformed_corners[:, 1])
        new_xmax = np.max(transformed_corners[:, 0])
        new_ymax = np.max(transformed_corners[:, 1])
        transformed_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
        
        # Draw transformed bounding box
        color = (0, 255, 0)  # Green
        transformed_image = cv2.rectangle(transformed_image,
                                          (int(new_xmin), int(new_ymin)),
                                          (int(new_xmax), int(new_ymax)),
                                          color, 3)
        label = f'Transformed ID: {int(ids[i][0])}'
        cv2.putText(transformed_image, label, (int(new_xmin), int(new_ymin)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    

    return transformed_image, transformed_boxes



# # Example usage
# yolo_mat_file = './data/yolo_0001.mat'
# image_path = './data/img_0001.jpg'
# H = np.array([
#     [-1.80993229e+00,  2.45954885e+00,  3.38956141e+03],
#     [ 4.18339032e-02, -3.83698497e+00,  3.40878223e+03],
#     [ 1.12300663e-04,  2.18391580e-03,  1.00000000e+00]
# ])  # Homography matrix

# # Call the function to draw and transform
# transformed_image, transformed_boxes = draw_bounding_boxes_and_transform(yolo_mat_file, image_path, H)

# Display the transformed image
# display_image(transformed_image)

# # Print transformed bounding boxes
# print("Transformed bounding boxes:")
# for box in transformed_boxes:
#     print(box)
