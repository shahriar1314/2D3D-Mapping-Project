import numpy as np

def warp_perspective_manually(image, H, output_size):
    height, width = output_size
    transformed_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Compute the inverse of the transformation matrix
    H_inv = np.linalg.inv(H)
    
    for y in range(height):
        for x in range(width):
            # Create homogeneous coordinates for the output pixel
            output_coord = np.array([x, y, 1])
            
            # Map the output pixel to the input space
            input_coord = H_inv @ output_coord
            input_x, input_y, input_w = input_coord
            
            # Normalize homogeneous coordinates
            input_x = input_x / input_w
            input_y = input_y / input_w
            
            # Check if the mapped coordinates are within bounds of the input image
            if 0 <= input_x < image.shape[1] and 0 <= input_y < image.shape[0]:
                # Perform nearest-neighbor interpolation
                input_x_int = int(round(input_x))
                input_y_int = int(round(input_y))
                
                # Copy pixel value to the output image
                transformed_image[y, x] = image[input_y_int, input_x_int]
    
    return transformed_image
