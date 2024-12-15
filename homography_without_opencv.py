import numpy as np



def homography_without_opencv(x1, y1, x2, y2):
    """
    Computes the homography matrix H such that [x2; y2; 1] ~ H * [x1; y1; 1].
    
    Parameters:
    - x1, y1: Lists or arrays of points from the first image.
    - x2, y2: Lists or arrays of corresponding points from the second image.

    Returns:
    - H: The 3x3 homography matrix.
    """
    # Ensure the input lists/arrays have the same length
    if len(x1) != len(y1) or len(x2) != len(y2) or len(x1) != len(x2):
        raise ValueError("Input points must have the same length.")

    # There must be at least 4 point correspondences
    if len(x1) < 4:
        raise ValueError("At least 4 point correspondences are required to compute the homography.")

    # Construct the A matrix for the linear system
    num_points = len(x1)
    A = []
    for i in range(num_points):
        x1i, y1i = x1[i], y1[i]
        x2i, y2i = x2[i], y2[i]
        A.append([-x1i, -y1i, -1, 0, 0, 0, x2i * x1i, x2i * y1i, x2i])
        A.append([0, 0, 0, -x1i, -y1i, -1, y2i * x1i, y2i * y1i, y2i])

    A = np.array(A)

    # Solve for h using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]  # The last row of Vt corresponds to the smallest singular value
    H = h.reshape((3, 3))  # Reshape to 3x3 matrix

    H = H / H[2, 2]

    return H