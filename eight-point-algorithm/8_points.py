import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def detect_and_match_features(img1, img2, num_matches=100):
    """
    Detect and match features between two images using SIFT and FLANN-based matching.

    Parameters:
    -----------
    img1 : numpy.ndarray
        First input image
    img2 : numpy.ndarray
        Second input image
    num_matches : int, optional
        Maximum number of good matches to return (default: 100)

    Returns:
    --------
    pts1_homogeneous : numpy.ndarray
        Homogeneous coordinates of matched points in the first image
    pts2_homogeneous : numpy.ndarray
        Homogeneous coordinates of matched points in the second image
    match_img : numpy.ndarray
        Visualization of the matched keypoints between the two images
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Configure FLANN matcher parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(
        algorithm=FLANN_INDEX_KDTREE, trees=5
    )  # Use 5 randomized KD-trees for searching
    search_params = dict(checks=50)  # Number of times to check leaf nodes during search

    # FLANN = Fast Library for Approximate Nearest Neighbors
    # More efficient than brute force for large feature sets
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # For each descriptor in first image, find 2 closest matches in second image
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    # Ensures that the best match is significantly better than the second-best match
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # m is best match, n is second-best
            good_matches.append(m)

    # Sort matches by distance (lower is better) and take top num_matches
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    good_matches = good_matches[:num_matches]

    # Extract the coordinates of matched keypoints
    # queryIdx refers to the keypoint in the first image
    # trainIdx refers to the keypoint in the second image
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Convert to homogeneous coordinates by appending 1 as third coordinate
    pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homogeneous = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    # Create a visualization of the matches
    match_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return pts1_homogeneous, pts2_homogeneous, match_img


def normalize_points(points):
    """
    Normalize point coordinates for improved numerical stability.

    This performs:
    1. Translation so the centroid is at the origin
    2. Scaling so the average distance from origin is sqrt(2)

    Parameters:
    -----------
    points : numpy.ndarray
        Points in homogeneous coordinates (Nx3)

    Returns:
    --------
    normalized_points : numpy.ndarray
        Normalized points in homogeneous coordinates
    T : numpy.ndarray
        3x3 transformation matrix used for normalization
    """
    # Compute the centroid of the points (ignoring the homogeneous coordinate)
    centroid = np.mean(points[:, :2], axis=0)

    # Center the points by subtracting the centroid
    centered_points = points[:, :2] - centroid

    # Calculate the average distance from the origin
    mean_dist = np.mean(np.sqrt(np.sum(centered_points**2, axis=1)))

    # Compute scale factor to make average distance sqrt(2)
    # This is the standard normalization suggested by Hartley
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0

    # Create transformation matrix for normalization
    # [scale   0    -scale*centroid_x]
    # [0     scale  -scale*centroid_y]
    # [0       0            1        ]
    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )

    # Apply transformation to points
    normalized_points = np.dot(T, points.T).T

    return normalized_points, T


def compute_fundamental_matrix(pts1, pts2, normalized=True):
    """
    Compute the fundamental matrix using the eight-point algorithm.

    Parameters:
    -----------
    pts1 : numpy.ndarray
        Points from the first image in homogeneous coordinates (Nx3)
    pts2 : numpy.ndarray
        Corresponding points from the second image in homogeneous coordinates (Nx3)
    normalized : bool, optional
        Whether to apply normalization for better numerical stability (default: True)

    Returns:
    --------
    F : numpy.ndarray
        3x3 fundamental matrix
    """
    # Optionally normalize points for better numerical stability
    if normalized:
        pts1_norm, T1 = normalize_points(pts1)
        pts2_norm, T2 = normalize_points(pts2)
    else:
        pts1_norm, pts2_norm = pts1, pts2
        T1, T2 = np.eye(3), np.eye(3)  # Identity matrices

    # A is the constraint matrix with one row per point correspondence
    # Each row represents the equation x2^T * F * x1 = 0
    n_points = pts1_norm.shape[0]
    A = np.zeros((n_points, 9))

    # Build the constraint matrix
    for i in range(n_points):
        x1, y1, _ = pts1_norm[i]  # Coordinates in image 1
        x2, y2, _ = pts2_norm[i]  # Corresponding coordinates in image 2

        # Each row has the form [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

    # Solve the system Af = 0 using SVD
    # The solution is the eigenvector corresponding to the smallest eigenvalue
    # which is the last column of V^T
    _, _, Vt = np.linalg.svd(A)

    # The solution is the last row of Vt (last column of V)
    F_flat = Vt[-1]

    # Reshape the 9-element vector into a 3x3 matrix
    F_norm = F_flat.reshape(3, 3)

    # Enforce the rank-2 constraint (a valid fundamental matrix has rank 2)
    # This is done by setting the smallest singular value to zero
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0  # Set the smallest singular value to zero
    F_norm = U @ np.diag(S) @ Vt

    # Denormalize the fundamental matrix if normalization was applied
    if normalized:
        F = T2.T @ F_norm @ T1
    else:
        F = F_norm

    # Normalize F to make F[2,2] = 1 (common convention)
    F = F / F[2, 2] if F[2, 2] != 0 else F

    return F


def compute_epipolar_lines(F, points, which="left"):
    """
    Compute epipolar lines in one image given points from the other image.

    Parameters:
    -----------
    F : numpy.ndarray
        Fundamental matrix
    points : numpy.ndarray
        Points in homogeneous coordinates
    which : str, optional
        'left' to compute lines in left image, 'right' to compute in right image (default: 'left')

    Returns:
    --------
    lines : numpy.ndarray
        Epipolar lines in the form [a, b, c] where ax + by + c = 0
    """
    if which == "left":
        # Lines in the left image: l = F^T * x2
        lines = np.dot(F.T, points.T).T
    else:
        # Lines in the right image: l' = F * x1
        lines = np.dot(F, points.T).T

    # Normalize lines so that a^2 + b^2 = 1
    # This makes the distance from a point to a line equal to |ax + by + c|
    norm = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2).reshape(-1, 1)
    lines = lines / norm

    return lines


def draw_epipolar_lines(img1, img2, pts1, pts2, F, figsize=(15, 10)):
    """
    Visualize epipolar geometry by drawing corresponding points and epipolar lines.

    Parameters:
    -----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image
    pts1 : numpy.ndarray
        Points from the first image in homogeneous coordinates
    pts2 : numpy.ndarray
        Corresponding points from the second image in homogeneous coordinates
    F : numpy.ndarray
        Fundamental matrix
    figsize : tuple, optional
        Figure size (default: (15, 10))

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the visualization
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Show the images in RGB (OpenCV loads as BGR)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Compute epipolar lines in the right image for points in the left image
    lines2 = compute_epipolar_lines(F, pts1, which="right")

    # Create a color map to assign each point/line pair a unique color
    colors = cm.rainbow(np.linspace(0, 1, len(pts1)))

    # For each point in the left image and its corresponding epipolar line in the right image
    for i, (pt1, line, color) in enumerate(zip(pts1, lines2, colors)):
        # Plot the point in the left image
        x1, y1, _ = pt1
        ax1.scatter(x1, y1, c=color.reshape(1, -1), s=50)

        # Draw the epipolar line in the right image
        a, b, c = line
        if abs(b) > 1e-8:  # Non-vertical line: y = (-ax - c) / b
            x_start, x_end = 0, w2
            y_start = (-c - a * x_start) / b
            y_end = (-c - a * x_end) / b
        else:  # Vertical line: x = -c / a
            y_start, y_end = 0, h2
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0

        ax2.plot([x_start, x_end], [y_start, y_end], c=color)

        # Plot the corresponding point in the right image
        x2, y2, _ = pts2[i]
        ax2.scatter(x2, y2, c=color.reshape(1, -1), s=50)

    # Compute epipolar lines in the left image for points in the right image
    lines1 = compute_epipolar_lines(F, pts2, which="left")

    # For each point in the right image and its corresponding epipolar line in the left image
    for i, (pt2, line, color) in enumerate(zip(pts2, lines1, colors)):
        # Draw the epipolar line in the left image
        a, b, c = line
        if abs(b) > 1e-8:  # Non-vertical line
            x_start, x_end = 0, w1
            y_start = (-c - a * x_start) / b
            y_end = (-c - a * x_end) / b
        else:  # Vertical line
            y_start, y_end = 0, h1
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0

        ax1.plot([x_start, x_end], [y_start, y_end], c=color)

    # Add titles and format the figure
    ax1.set_title("Image 1 with Epipolar Lines")
    ax2.set_title("Image 2 with Epipolar Lines")

    ax1.axis("off")
    ax2.axis("off")

    plt.tight_layout()

    return fig


def compute_geometric_error(pts1, pts2, F):
    """
    Compute the geometric error (Sampson distance) for point correspondences.

    The error is the distance from a point to its corresponding epipolar line,
    averaged between both directions.

    Parameters:
    -----------
    pts1 : numpy.ndarray
        Points from the first image in homogeneous coordinates
    pts2 : numpy.ndarray
        Corresponding points from the second image in homogeneous coordinates
    F : numpy.ndarray
        Fundamental matrix

    Returns:
    --------
    errors : numpy.ndarray
        Array of geometric errors for each point correspondence
    """
    # Compute epipolar lines in both directions
    lines1 = compute_epipolar_lines(F, pts2, which="left")
    lines2 = compute_epipolar_lines(F, pts1, which="right")

    errors = []

    # For each pair of corresponding points
    for i in range(pts1.shape[0]):
        # Distance from point in image 1 to its epipolar line
        x1, y1, _ = pts1[i]
        a1, b1, c1 = lines1[i]
        dist1 = abs(a1 * x1 + b1 * y1 + c1)  # Distance formula for normalized line

        # Distance from point in image 2 to its epipolar line
        x2, y2, _ = pts2[i]
        a2, b2, c2 = lines2[i]
        dist2 = abs(a2 * x2 + b2 * y2 + c2)  # Distance formula for normalized line

        # Average the distances (symmetric measure)
        error = (dist1 + dist2) / 2
        errors.append(error)

    return np.array(errors)


def ransac_fundamental_matrix(
    pts1, pts2, iterations=1000, threshold=2.0, adaptive=True, final_percentile=0.7
):
    """
    Estimate the fundamental matrix robustly using RANSAC.

    Parameters:
    -----------
    pts1 : numpy.ndarray
        Points from the first image in homogeneous coordinates
    pts2 : numpy.ndarray
        Corresponding points from the second image in homogeneous coordinates
    iterations : int, optional
        Number of RANSAC iterations (default: 1000)
    threshold : float, optional
        Initial inlier threshold in pixels (default: 2.0)
    adaptive : bool, optional
        Whether to use adaptive thresholding (default: True)
    final_percentile : float, optional
        Percentile for adaptive threshold calculation (default: 0.7)

    Returns:
    --------
    best_F : numpy.ndarray
        Best fundamental matrix found
    best_inliers : numpy.ndarray
        Indices of inlier point correspondences
    """
    best_F = None
    best_inliers = []
    num_points = pts1.shape[0]

    # Standard RANSAC approach
    if not adaptive:
        for _ in range(iterations):
            # Randomly sample 8 points (minimum needed for the eight-point algorithm)
            indices = random.sample(range(num_points), 8)

            # Compute fundamental matrix from the sample
            F = compute_fundamental_matrix(pts1[indices], pts2[indices])

            # Calculate error for all points
            errors = compute_geometric_error(pts1, pts2, F)

            # Find inliers (points with error below threshold)
            inliers = np.where(errors < threshold)[0]

            # Keep the model with the most inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_F = F

    # Adaptive RANSAC approach - uses a two-stage process
    else:
        # Stage 1: Initial RANSAC with fixed threshold
        for _ in range(iterations // 2):
            indices = random.sample(range(num_points), 8)
            F = compute_fundamental_matrix(pts1[indices], pts2[indices])
            errors = compute_geometric_error(pts1, pts2, F)
            inliers = np.where(errors < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_F = F

        # If we found enough inliers, refine the threshold
        if len(best_inliers) >= 8:
            # Compute a refined F using all current inliers
            refined_F = compute_fundamental_matrix(
                pts1[best_inliers], pts2[best_inliers]
            )

            # Calculate errors for all points using the refined F
            all_errors = compute_geometric_error(pts1, pts2, refined_F)

            # Set adaptive threshold based on error distribution
            sorted_errors = np.sort(all_errors)
            adaptive_threshold = sorted_errors[
                int(final_percentile * len(sorted_errors))
            ]

            # Clamp threshold to reasonable values
            adaptive_threshold = max(0.5, min(adaptive_threshold, 3.0))

            print(f"Adaptive threshold: {adaptive_threshold:.4f} pixels")

            # Stage 2: Second round of RANSAC with adaptive threshold
            for _ in range(iterations // 2):
                indices = random.sample(range(num_points), 8)
                F = compute_fundamental_matrix(pts1[indices], pts2[indices])
                errors = compute_geometric_error(pts1, pts2, F)
                inliers = np.where(errors < adaptive_threshold)[0]

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_F = F

    # Final refinement using all inliers
    if len(best_inliers) >= 8:
        best_F = compute_fundamental_matrix(pts1[best_inliers], pts2[best_inliers])
    else:
        print(
            f"Warning: Only {len(best_inliers)} inliers found, which is less than the minimum 8 required"
        )

    return best_F, best_inliers


def eight_point_algorithm(img1_path, img2_path, use_ransac=True, num_matches=100):
    """
    Implement the complete eight-point algorithm workflow to estimate the fundamental matrix.

    Parameters:
    -----------
    img1_path : str
        Path to the first image
    img2_path : str
        Path to the second image
    use_ransac : bool, optional
        Whether to use RANSAC for robust estimation (default: True)
    num_matches : int, optional
        Number of feature matches to use (default: 100)

    Returns:
    --------
    F : numpy.ndarray
        Estimated fundamental matrix
    pts1 : numpy.ndarray
        Points from the first image used in the estimation
    pts2 : numpy.ndarray
        Corresponding points from the second image
    match_img : numpy.ndarray
        Visualization of the feature matches
    errors : numpy.ndarray
        Geometric errors for each point correspondence
    fig : matplotlib.figure.Figure
        Visualization of epipolar geometry
    """
    # Load the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Detect features and establish correspondences
    pts1, pts2, match_img = detect_and_match_features(img1, img2, num_matches)

    # Robust estimation with RANSAC
    if use_ransac:
        F, inliers = ransac_fundamental_matrix(
            pts1, pts2, iterations=1000, adaptive=True
        )

        # Keep only inlier points for subsequent operations
        pts1_inliers = pts1[inliers]
        pts2_inliers = pts2[inliers]

        # Compute geometric errors
        errors = compute_geometric_error(pts1_inliers, pts2_inliers, F)

        # Print statistics
        print(f"RANSAC found {len(inliers)} inliers out of {len(pts1)} points")
        print(f"Mean geometric error: {np.mean(errors):.4f}")
        print(f"Max geometric error: {np.max(errors):.4f}")

        # Visualize epipolar geometry
        fig = draw_epipolar_lines(img1, img2, pts1_inliers, pts2_inliers, F)

        return F, pts1_inliers, pts2_inliers, match_img, errors, fig

    # Standard (non-robust) estimation
    else:
        F = compute_fundamental_matrix(pts1, pts2)
        errors = compute_geometric_error(pts1, pts2, F)

        # Print statistics
        print(f"Mean geometric error: {np.mean(errors):.4f}")
        print(f"Max geometric error: {np.max(errors):.4f}")

        # Visualize epipolar geometry
        fig = draw_epipolar_lines(img1, img2, pts1, pts2, F)

        return F, pts1, pts2, match_img, errors, fig


# Example usage:
if __name__ == "__main__":
    # Estimate the fundamental matrix between two images
    F, pts1, pts2, match_img, errors, fig = eight_point_algorithm(
        "./img/giratina_1.jpeg", "./img/giratina_2.jpeg"
    )

    # Display the matched features
    plt.figure(figsize=(10, 5))
    plt.imshow(match_img)
    plt.title("SIFT Matches")
    plt.axis("off")
    plt.show()
