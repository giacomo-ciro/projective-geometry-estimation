import math
import os
import sys
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# TODO: look into why the SIFT retrieved correspondences are all squeezed on the left side of the picture

def generate_postfix(config):
    a =  (
        f"{config["img"]}"
        f"_{config["n_correspondences"]}"
        f"_{config["use_ransac"]}"
    )

    return a

def get_images(config):
    img_name = config["img"]
    img1_path = f"./img/{img_name}_1.jpeg"
    img2_path = f"./img/{img_name}_2.jpeg"
    
    # Check existence of required files
    if not os.path.exists(img1_path):
        print("Cannot find image path {img1_path}. Please, rename accordingly.")
        sys.exit()
    if not os.path.exists(img2_path):
        print("Cannot find image path {img2_path}. Please, rename accordingly.")
        sys.exit()
    
    # Load images
    return cv2.imread(img1_path), cv2.imread(img2_path)

def get_annotated_correspondences(correspondences_path):
    
    # Read correspondences
    pts1 = []
    pts2 = []

    with open(correspondences_path, "r") as f:
        for line in f:
            # Skip comment lines
            if line.startswith("#"):
                continue

            # Parse line
            try:
                values = line.strip().split()
                if len(values) == 4:
                    x1, y1, x2, y2 = map(float, values)
                    pts1.append((x1, y1, 1.0))
                    pts2.append((x2, y2, 1.0))
            except ValueError:
                continue

    return np.array(pts1), np.array(pts2)

def plot_correspondences(img1, img2, pts1, pts2):
    # Convert points to keypoints
    kp1 = [cv2.KeyPoint(x, y, 1) for x, y, u in pts1]
    kp2 = [cv2.KeyPoint(x, y, 1) for x, y, u in pts2]

    # Create match objects
    matches = [cv2.DMatch(i, i, 0) for i in range(len(pts1))]

    # Set random seed for reproducible colors
    np.random.seed(42)

    # Create the match visualization
    # flags:
    # cv2.DrawMatchesFlags_DEFAULT: draws lines and keypoints
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: draws keypoints in size with orientation
    # cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: only draws lines, no keypoints
    match_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return match_img

def normalize_points(points):

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

def eight_points_algo(pts1, pts2, normalized=True):

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
        y_1, y_2, _ = pts1_norm[i]  # Coordinates in image 1
        y_prime_1, y_prime_2, _ = pts2_norm[i]  # Corresponding coordinates in image 2

        # Each row has the form
        A[i] = [
            y_prime_1 * y_1,
            y_prime_1 * y_2,
            y_prime_1,
            y_prime_2 * y_1,
            y_prime_2 * y_2,
            y_prime_2,
            y_1,
            y_2,
            1
        ]

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

    if which == "left":
        # Lines in the left image: l = F^T * x2
        lines = np.dot(F.T, points.T).T
    else:
        # Lines in the right image: l' = F * x1
        lines = np.dot(F, points.T).T

    # Normalize lines so that a^2 + b^2 = 1
    # This makes the distance from a point to a line equal to |ax + by + c|
    # Recall in 2D geometry the distance of a point to a line is just:
    # distance = abs(ax₀ + by₀ + c) / sqrt(a² + b²)
    # hence if we normalize the coordinates in this 
    # way we simplify the distance computation
    # (we can do this because lines in the projective plane are
    # invariant under scalar multiplication)
    norm = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2).reshape(-1, 1)
    lines = lines / norm

    return lines

def draw_epipolar_lines(img1, img2, pts1, pts2, F, figsize=(15, 10)):

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Show the images in RGB (OpenCV loads as BGR)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Restrict graph to image dimension
    ax1.set_xlim(0, w1)
    ax1.set_ylim(h1, 0)
    ax2.set_xlim(0, w2)
    ax2.set_ylim(h2, 0)

    # Create a color map to assign each point/line pair a unique color
    colors = cm.rainbow(np.linspace(0, 1, len(pts1)))

    # RIGHT Epipolar Lines // LEFT Points
    lines2 = compute_epipolar_lines(F, pts1, which="right")

    # For each point in the left image and its corresponding epipolar line in the right image
    for i, (pt1, line, color) in enumerate(zip(pts1, lines2, colors)):

        x1, y1, _ = pt1
        ax1.scatter(x1, y1, c=color.reshape(1, -1), s=50)

        a, b, c = line
        if abs(b) > 1e-8:
            x_start, x_end = 0, w2
            y_start = (-c - a * x_start) / b
            y_end = (-c - a * x_end) / b
        else:
            y_start, y_end = 0, h2
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0

        ax2.plot([x_start, x_end], [y_start, y_end], c=color)

        x2, y2, _ = pts2[i]
        ax2.scatter(x2, y2, c=color.reshape(1, -1), s=50)

    # LEFT Epipolar Lines // RIGHT Points
    lines1 = compute_epipolar_lines(F, pts2, which="left")

    # For each point in the right image and its corresponding epipolar line in the left image
    for i, (pt2, line, color) in enumerate(zip(pts2, lines1, colors)):

        a, b, c = line
        if abs(b) > 1e-8:  # Non-vertical line
            x_start, x_end = 0, w1
            y_start = (-c - a * x_start) / b
            y_end = (-c - a * x_end) / b
        else: 
            y_start, y_end = 0, h1
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0

        ax1.plot([x_start, x_end], [y_start, y_end], c=color)

    ax1.axis("off")
    ax2.axis("off")

    plt.tight_layout()

    return fig

def compute_geometric_error(pts1, pts2, F):
    """
    Compute the geometric error (point-to-line distance) for point correspondences.
    
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
    
    # Calculate point-to-line distances
    errors = []
    for i in range(pts1.shape[0]):
        # Convert to inhomogeneous coordinates for proper distance calculation
        x1 = pts1[i, 0] / pts1[i, 2] if pts1[i, 2] != 0 else pts1[i, 0]
        y1 = pts1[i, 1] / pts1[i, 2] if pts1[i, 2] != 0 else pts1[i, 1]
        
        x2 = pts2[i, 0] / pts2[i, 2] if pts2[i, 2] != 0 else pts2[i, 0]
        y2 = pts2[i, 1] / pts2[i, 2] if pts2[i, 2] != 0 else pts2[i, 1]
        
        # Extract line parameters (already normalized in compute_epipolar_lines)
        a1, b1, c1 = lines1[i]
        a2, b2, c2 = lines2[i]
        
        # Calculate point-to-line distances
        # For normalized lines (a²+b²=1), the distance is |ax+by+c|
        dist1 = abs(a1 * x1 + b1 * y1 + c1)
        dist2 = abs(a2 * x2 + b2 * y2 + c2)
        
        # Average the distances (symmetric measure)
        error = (dist1 + dist2) / 2
        errors.append(error)
    
    return np.array(errors)

def eight_points_ransac(
    pts1,
    pts2,
    iterations,
    threshold,
):

    best_F = None
    best_inliers = np.array([], dtype=int)
    num_points = pts1.shape[0]

    # Check if we have enough points
    if num_points < 8:
        raise ValueError(
            f"Need at least 8 point correspondences, but only {num_points} provided"
        )

    # Normalize points flag should be True for better numerical stability
    normalize_points = True

    # RANSAC implementation
    for i in range(iterations):

        # Randomly sample 8 points (minimum needed for the eight-point algorithm)
        indices = np.random.choice(num_points, 8, replace=False)

        # Compute fundamental matrix from the sample
        F = eight_points_algo(pts1[indices], pts2[indices], normalized=normalize_points)

        # Calculate error for all points
        errors = compute_geometric_error(pts1, pts2, F)

        # Find inliers (points with error below threshold)
        current_inliers = np.where(errors < threshold)[0]

        # Keep the model with the most inliers
        if len(current_inliers) > len(best_inliers):
            best_inliers = current_inliers
            best_F = F

    # Final refinement using all inliers
    if len(best_inliers) >= 8:
        print(
            f"RANSAC found {len(best_inliers)} inliers, estimating the final F."
        )
        best_F = eight_points_algo(
            pts1[best_inliers], pts2[best_inliers], normalized=normalize_points
        )
        errors = compute_geometric_error(pts1, pts2, best_F)
        best_inliers = np.where(errors < threshold)[0]

    else:
        print(
            f"RANSAC only found {len(best_inliers)} inliers, falling back and using all points to estimate F."
        )
        best_F = eight_points_algo(pts1, pts2, normalized=normalize_points)
        best_inliers = np.arange(num_points)

    return best_F, best_inliers

def is_degenerate_configuration(pts1, pts2, min_distance=1e-3):

    # Check if points are too close to each other
    for i in range(len(pts1)):
        for j in range(i + 1, len(pts1)):
            if (
                np.linalg.norm(pts1[i, :2] - pts1[j, :2]) < min_distance
                or np.linalg.norm(pts2[i, :2] - pts2[j, :2]) < min_distance
            ):
                return True

    # Check if points are collinear in either image (requires at least 3 points)
    if len(pts1) >= 3:
        # Convert to homogeneous if needed
        if pts1.shape[1] == 2:
            pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
        else:
            pts1_hom = pts1.copy()
            pts2_hom = pts2.copy()

        # Check for collinearity in first image
        for i in range(len(pts1_hom) - 2):
            for j in range(i + 1, len(pts1_hom) - 1):
                for k in range(j + 1, len(pts1_hom)):
                    # Calculate cross product of vectors from i to j and i to k
                    cross = np.cross(
                        pts1_hom[j, :2] - pts1_hom[i, :2],
                        pts1_hom[k, :2] - pts1_hom[i, :2],
                    )
                    if abs(cross) < min_distance:
                        return True

        # Check for collinearity in second image
        for i in range(len(pts2_hom) - 2):
            for j in range(i + 1, len(pts2_hom) - 1):
                for k in range(j + 1, len(pts2_hom)):
                    cross = np.cross(
                        pts2_hom[j, :2] - pts2_hom[i, :2],
                        pts2_hom[k, :2] - pts2_hom[i, :2],
                    )
                    if abs(cross) < min_distance:
                        return True

    return False

def main(img1, img2, config):
        
    # Check existence of correspondences.txt
    correspondences_path=f"./correspondences_{config["img"]}.txt"
    if not os.path.exists(correspondences_path):
        print("Cannot find correspondences path {correspondences_path}. Please, rename accordingly.")
        sys.exit()
    
    # Parse
    pts1, pts2 = get_annotated_correspondences(correspondences_path)
    pts1, pts2 = pts1[:config["n_correspondences"], :], pts2[:config["n_correspondences"], :]

    # Plot the correspondences
    match_img = plot_correspondences(img1, img2, pts1, pts2)

    # Robust estimation with RANSAC
    if config["use_ransac"]:
        F, inliers = eight_points_ransac(
            pts1,
            pts2,
            threshold= config["ransac_threshold"],
            iterations = config["ransac_iterations"],
        )

        # Keep only inlier points for subsequent operations
        pts1_inliers = pts1[inliers]
        pts2_inliers = pts2[inliers]

        # Print statistics
        print(f"RANSAC found {len(inliers)} inliers out of {len(pts1)} points")

        return F, pts1_inliers, pts2_inliers, match_img

    # Standard (non-robust) estimation
    else:
        F = eight_points_algo(pts1, pts2, normalized=config["normalize"])

        return F, pts1, pts2, match_img


if __name__ == "__main__":
    
    # --------------------- PLOTTING -------------------
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    # Reproducibility
    if config["seed"]:
        np.random.seed(config["seed"])
        cv2.setRNGSeed(config["seed"])
    
    # Get images
    img1, img2 = get_images(config)

    # --------------- 8 POINTS ALGO ----------------
    F, pts1, pts2, match_img = main(
        img1, 
        img2,
        config,
    )

    # --------------------- ERRORS -------------------
    # Compute errors for each point (distance to epipolar line)
    errors = compute_geometric_error(pts1, pts2, F)

    # Get stats
    print(f"Mean geometric error: {np.mean(errors):}")
    print(f"Max geometric error: {np.max(errors):}")
    
    # --------------------- PLOTTING -------------------
    
    postfix = generate_postfix(config)

    # Epipolar Lines
    fig = draw_epipolar_lines(
        img1,
        img2,
        pts1,
        pts2,
        F,
    )
    fig.savefig(f"./save/epipolar_{postfix}.png")

    # Correspondences
    plt.figure(figsize=(10, 5))
    plt.imshow(match_img)
    plt.axis("off")
    plt.savefig(f"./save/correspondeces_{postfix}.png")