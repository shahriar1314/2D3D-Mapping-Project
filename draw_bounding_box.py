import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt

def draw_bounding_boxes(yolo_mat_file, image_path):
    """
    Draws bounding boxes on an image using the coordinates and object IDs from a YOLO .mat file.
    
    Parameters:
    - yolo_mat_file: Path to the YOLO .mat file containing bounding box coordinates and object IDs.
    - image_path: Path to the image on which to draw the bounding boxes.
    
    Returns:
    - image_rgb: Image with the bounding boxes drawn.
    """
    # Load the .mat file
    data = sio.loadmat(yolo_mat_file)
    
    # Extract bounding box coordinates and ids
    xyxy = data['xyxy']  # Bounding box coordinates (xmin, ymin, xmax, ymax)
    ids = data['id']     # Object IDs (can be used to differentiate between objects, e.g., car, person)
    
    # Load the image you want to overlay the bounding boxes onto
    image = cv2.imread(image_path)
    
    # Convert the image to RGB (since OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw the bounding boxes on the image
    for i in range(xyxy.shape[0]):
        xmin, ymin, xmax, ymax = xyxy[i]
        object_id = int(ids[i])  # Use this ID for labels or distinguishing objects
        
        # Draw the rectangle (BGR format for OpenCV)
        color = (255, 0, 0)  # Blue color for the box (you can choose any color)
        image_rgb = cv2.rectangle(image_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)
        
        # Optionally, you can also put text (object_id or class name) on the box
        label = f'ID: {object_id}'
        cv2.putText(image_rgb, label, (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image_rgb

    

# def display_image(image):
#     """
#     Displays the given image using matplotlib.
    
#     Parameters:
#     - image: The image to be displayed.
#     """
#     plt.imshow(image)
#     plt.axis('off')  # Hide axes
#     plt.show()

# # Example usage
# yolo_mat_file = './data/yolo_0044.mat'
# image_path = './data/img_0044.jpg'

# # Call the function to draw bounding boxes on the image
# result_image = draw_bounding_boxes(yolo_mat_file, image_path)


# # Display the image with bounding boxes
# display_image(result_image)

