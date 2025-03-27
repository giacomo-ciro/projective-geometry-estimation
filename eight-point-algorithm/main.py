import json
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def generate_postfix(config: dict) -> str:
    """
    Generate a string postfix for filenames based on configuration parameters.

    Args:
        config (dict): Configuration dictionary containing image name, number of correspondences,
                      and whether RANSAC is used.

    Returns:
        str: Formatted string to use as a postfix for output filenames.
    """
    a = f"{config['img']}{"_ransac" if config['use_ransac'] else ""}"

    return a


def get_images(config: dict) -> tuple:
    """
    Load image pairs based on configuration.

    Args:
        config (dict): Configuration dictionary containing the image base name.

    Returns:
        tuple: A pair of loaded images (img1, img2) using OpenCV.

    Raises:
        SystemExit: If either image cannot be found.
    """
    img_name = config["img"]
    img1_path = f"./img/{img_name}_1.jpeg"
    img2_path = f"./img/{img_name}_2.jpeg"

    if not os.path.exists(img1_path):
        print(f"Cannot find image path {img1_path}. Please, rename accordingly.")
        sys.exit()
    if not os.path.exists(img2_path):
        print(f"Cannot find image path {img2_path}. Please, rename accordingly.")
        sys.exit()

    return cv2.imread(img1_path), cv2.imread(img2_path)


def get_annotated_correspondences(correspondences_path: str) -> tuple:
    """
    Parse a text file containing manually annotated point correspondences.

    Args:
        correspondences_path (str): Path to the file containing point correspondences.

    Returns:
        tuple: Two numpy arrays containing homogeneous coordinates (x, y, 1) for corresponding points
               in the first and second images.
    """
    pts1 = []
    pts2 = []

    with open(correspondences_path, "r") as f:
        for line in f:
            if line.startswith("#"):  # Skip comment lines
                continue
            try:
                values = line.strip().split()
                if len(values) == 4:
                    x1, y1, x2, y2 = map(float, values)
                    pts1.append((x1, y1, 1.0))  # Add homogeneous coordinate
                    pts2.append((x2, y2, 1.0))  # Add homogeneous coordinate
            except ValueError:
                continue

    return np.array(pts1), np.array(pts2)


def plot_correspondences(
    img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
) -> np.ndarray:
    """
    Visualize point correspondences between two images.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        pts1 (np.ndarray): Point coordinates in the first image (Nx3).
        pts2 (np.ndarray): Corresponding point coordinates in the second image (Nx3).

    Returns:
        np.ndarray: Image showing both input images with lines connecting corresponding points.
    """
    # Convert points to keypoints for OpenCV visualization
    kp1 = [cv2.KeyPoint(x, y, 1) for x, y, u in pts1]
    kp2 = [cv2.KeyPoint(x, y, 1) for x, y, u in pts2]

    # Create DMatch objects to define correspondences
    matches = [cv2.DMatch(i, i, 0) for i in range(len(pts1))]

    # Draw matches between the two images
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


def normalize_points(points: np.ndarray) -> tuple:
    """
    Normalize points using Hartley normalization to improve numerical stability.

    Transforms the points such that their centroid is at the origin and the average
    distance from the origin is sqrt(2).

    Args:
        points (np.ndarray): Array of points in homogeneous coordinates (Nx3).

    Returns:
        tuple: Normalized points and the transformation matrix T.
    """
    # Make mean 0
    centroid = np.mean(points[:, :2], axis=0)
    centered_points = points[:, :2] - centroid

    # Scale factor to make distance from the origin = sqrt(2) (Hartley Normalization)
    mean_dist = np.mean(np.sqrt(np.sum(centered_points**2, axis=1)))
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0

    # Create the transformation matrix (to later transform the estimated matrix F)
    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )

    # Apply transformation to points
    normalized_points = np.dot(T, points.T).T

    return normalized_points, T


def eight_points_algo(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Implementation of the eight-point algorithm for fundamental matrix estimation.

    This function first normalizes the points for numerical stability, builds the constraint
    matrix, solves for F using SVD, enforces the rank-2 constraint, and then denormalizes.

    Args:
        pts1 (np.ndarray): Points in the first image (Nx3 homogeneous coordinates).
        pts2 (np.ndarray): Corresponding points in the second image (Nx3 homogeneous coordinates).

    Returns:
        np.ndarray: The estimated 3x3 fundamental matrix F.
    """
    # Normalize points for better numerical stability
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # A is the constraint matrix with one row per point correspondence
    # Each row represents the equation x2^T * F * x1 = 0
    n_points = pts1_norm.shape[0]
    A = np.zeros((n_points, 9))

    # Build the constraint matrix
    for i in range(n_points):
        y_1, y_2, _ = pts1_norm[i]  # Coordinates in image 1
        y_prime_1, y_prime_2, _ = pts2_norm[i]  # Corresponding coordinates in image 2

        # Each row has the form [x'x, x'y, x', y'x, y'y, y', x, y, 1]
        A[i] = [
            y_prime_1 * y_1,
            y_prime_1 * y_2,
            y_prime_1,
            y_prime_2 * y_1,
            y_prime_2 * y_2,
            y_prime_2,
            y_1,
            y_2,
            1,
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
    F = T2.T @ F_norm @ T1

    # Normalize F to make F[2,2] = 1 (common convention)
    F = F / F[2, 2] if F[2, 2] != 0 else F

    return F


def compute_epipolar_lines(
    F: np.ndarray, points: np.ndarray, which: str = "left"
) -> np.ndarray:
    """
    Compute epipolar lines in one image from points in the other image.

    Args:
        F (np.ndarray): Fundamental matrix (3x3).
        points (np.ndarray): Points in homogeneous coordinates (Nx3).
        which (str): Direction of computation. 'left' computes lines in left image
                     from points in right image, and 'right' does the opposite.

    Returns:
        np.ndarray: Epipolar lines represented as [a, b, c] where ax + by + c = 0.
                   Lines are normalized such that a^2 + b^2 = 1.
    """
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


def draw_epipolar_lines(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    figsize: tuple = (15, 10),
) -> plt.Figure:
    """
    Visualize epipolar lines in both images based on the fundamental matrix.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        pts1 (np.ndarray): Points in the first image (Nx3 homogeneous coordinates).
        pts2 (np.ndarray): Corresponding points in the second image (Nx3 homogeneous coordinates).
        F (np.ndarray): Fundamental matrix (3x3).
        figsize (tuple): Figure size for the plot.

    Returns:
        plt.Figure: Matplotlib figure containing the visualization.
    """
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
    ax1.set_ylim(h1, 0)  # Inverted y-axis for image coordinates
    ax2.set_xlim(0, w2)
    ax2.set_ylim(h2, 0)  # Inverted y-axis for image coordinates

    # Create a color map to assign each point/line pair a unique color
    colors = cm.rainbow(np.linspace(0, 1, len(pts1)))

    # RIGHT Epipolar Lines // LEFT Points
    lines2 = compute_epipolar_lines(F, pts1, which="right")

    # For each point in the left image and its corresponding epipolar line in the right image
    for i, (pt1, line, color) in enumerate(zip(pts1, lines2, colors)):
        x1, y1, _ = pt1
        ax1.scatter(x1, y1, c=color.reshape(1, -1), s=50)

        # Calculate line endpoints
        a, b, c = line
        if abs(b) > 1e-8:  # Non-horizontal line
            x_start, x_end = 0, w2
            y_start = (-c - a * x_start) / b
            y_end = (-c - a * x_end) / b
        else:  # Horizontal line
            y_start, y_end = 0, h2
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0

        ax2.plot([x_start, x_end], [y_start, y_end], c=color)

        # Plot the corresponding point in the right image
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
        else:  # Vertical line
            y_start, y_end = 0, h1
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0

        ax1.plot([x_start, x_end], [y_start, y_end], c=color)

    ax1.axis("off")
    ax2.axis("off")

    plt.tight_layout()

    return fig


def compute_geometric_error(
    pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray
) -> np.ndarray:
    """
    Compute symmetric epipolar distance error for point correspondences.

    The error is the average of the distances from each point to its corresponding epipolar line
    in both directions.

    Args:
        pts1 (np.ndarray): Points in the first image (Nx3 homogeneous coordinates).
        pts2 (np.ndarray): Corresponding points in the second image (Nx3 homogeneous coordinates).
        F (np.ndarray): Fundamental matrix (3x3).

    Returns:
        np.ndarray: Array of geometric errors for each point correspondence.
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
    pts1: np.ndarray, pts2: np.ndarray, iterations: int, threshold: float
) -> tuple:
    """
    Robust fundamental matrix estimation using RANSAC with the eight-point algorithm.

    Randomly samples 8 point correspondences, estimates F, and keeps the model with
    the most inliers. Final F is computed using all inliers.

    Args:
        pts1 (np.ndarray): Points in the first image (Nx3 homogeneous coordinates).
        pts2 (np.ndarray): Corresponding points in the second image (Nx3 homogeneous coordinates).
        iterations (int): Number of RANSAC iterations.
        threshold (float): Error threshold for considering a point as an inlier.

    Returns:
        tuple: (F, inliers) where F is the fundamental matrix and inliers is an array
               of indices of inlier points.

    Raises:
        ValueError: If fewer than 8 point correspondences are provided.
    """
    best_F = None
    best_inliers = np.array([], dtype=int)
    num_points = pts1.shape[0]

    # Check if we have enough points
    if num_points < 8:
        raise ValueError(
            f"Need at least 8 point correspondences, but only {num_points} provided"
        )

    # RANSAC implementation
    for i in range(iterations):
        # Randomly sample 8 points (minimum needed for the eight-point algorithm)
        indices = np.random.choice(num_points, 8, replace=False)

        # Compute fundamental matrix from the sample
        F = eight_points_algo(pts1[indices], pts2[indices])

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
        print(f"RANSAC found {len(best_inliers)} inliers, estimating the final F.")
        best_F = eight_points_algo(pts1[best_inliers], pts2[best_inliers])
        errors = compute_geometric_error(pts1, pts2, best_F)
        best_inliers = np.where(errors < threshold)[0]

    else:
        print(
            f"RANSAC only found {len(best_inliers)} inliers, falling back and using all points to estimate F."
        )
        best_F = eight_points_algo(pts1, pts2)
        best_inliers = np.arange(num_points)

    return best_F, best_inliers


def main(img1: np.ndarray, img2: np.ndarray, config: dict) -> tuple:
    """
    Main function for fundamental matrix estimation and visualization.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        config (dict): Configuration dictionary with parameters for the algorithm.

    Returns:
        tuple: (F, pts1, pts2, match_img) where F is the fundamental matrix,
               pts1 and pts2 are the point correspondences used (possibly filtered
               by RANSAC), and match_img is an image showing correspondences.

    Raises:
        SystemExit: If correspondences file cannot be found.
    """
    # Check existence of correspondences.txt
    correspondences_path = f"./correspondences_{config['img']}.txt"
    if not os.path.exists(correspondences_path):
        print(
            f"Cannot find correspondences path {correspondences_path}. Please, rename accordingly."
        )
        sys.exit()

    # Parse correspondences file and limit to specified number
    pts1, pts2 = get_annotated_correspondences(correspondences_path)

    # Plot the correspondences
    match_img = plot_correspondences(img1, img2, pts1, pts2)

    # Robust estimation with RANSAC
    if config["use_ransac"]:
        F, inliers = eight_points_ransac(
            pts1,
            pts2,
            threshold=config["ransac_threshold"],
            iterations=config["ransac_iterations"],
        )

        # Keep only inlier points for subsequent operations
        pts1_inliers = pts1[inliers]
        pts2_inliers = pts2[inliers]

        # Print statistics
        # print(f"RANSAC found {len(inliers)} inliers out of {len(pts1)} points")

        return F, pts1_inliers, pts2_inliers, match_img

    # Standard (non-robust) estimation
    else:
        F = eight_points_algo(pts1, pts2)

        return F, pts1, pts2, match_img


def run_multiple_trials(config, num_trials=100):
    """
    Run multiple trials of the fundamental matrix estimation and collect error statistics.
    
    Args:
        config (dict): Configuration dictionary with parameters for the algorithm.
        num_trials (int): Number of trials to run.
        
    Returns:
        dict: Dictionary containing statistics of mean and max errors across trials.
    """
    mean_errors = []
    max_errors = []
    
    # Get images (only need to do this once)
    img1, img2 = get_images(config)
    
    # Get correspondences path
    correspondences_path = f"./correspondences_{config['img']}.txt"
    if not os.path.exists(correspondences_path):
        print(f"Cannot find correspondences path {correspondences_path}. Please, rename accordingly.")
        sys.exit()
    
    # Parse correspondences file
    pts1, pts2 = get_annotated_correspondences(correspondences_path)
    
    print(f"Running {num_trials} trials...")
    for i in range(num_trials):
        if i % 10 == 0:
            print(f"Trial {i}/{num_trials}")
            
        # Set a different random seed for each trial
        np.random.seed(config["seed"] + i if config["seed"] else None)
        cv2.setRNGSeed(config["seed"] + i if config["seed"] else i)
        
        # Run the fundamental matrix estimation
        if config["use_ransac"]:
            F, inliers = eight_points_ransac(
                pts1,
                pts2,
                threshold=config["ransac_threshold"],
                iterations=config["ransac_iterations"],
            )
            pts1_inliers = pts1[inliers]
            pts2_inliers = pts2[inliers]
        else:
            F = eight_points_algo(pts1, pts2)
            pts1_inliers, pts2_inliers = pts1, pts2
        
        # Compute errors
        errors = compute_geometric_error(pts1_inliers, pts2_inliers, F)
        
        # Store the statistics
        mean_errors.append(np.mean(errors))
        max_errors.append(np.max(errors))
    
    # Calculate statistics across all trials
    stats = {
        "mean_error": {
            "average": np.mean(mean_errors),
            "median": np.median(mean_errors),
            "std_dev": np.std(mean_errors)
        },
        "max_error": {
            "average": np.mean(max_errors),
            "median": np.median(max_errors),
            "std_dev": np.std(max_errors)
        },
        "raw_data": {
            "mean_errors": mean_errors,
            "max_errors": max_errors
        }
    }
    
    return stats


if __name__ == "__main__":
    """
    Main entry point of the script for epipolar geometry estimation.
    
    This loads configuration from a JSON file, runs the estimation algorithm multiple times,
    computes error statistics, and generates visualizations.
    """
    # --------------------- PLOTTING -------------------
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    # Set random seed for reproducibility if specified
    if config["seed"]:
        np.random.seed(config["seed"])
        cv2.setRNGSeed(config["seed"])

    # Get images for visualization
    img1, img2 = get_images(config)

    # Run a single trial for visualization purposes
    F, pts1, pts2, match_img = main(img1, img2, config)

    # Display single trial error statistics
    errors = compute_geometric_error(pts1, pts2, F)
    print(f"Single trial - Mean geometric error: {np.mean(errors)}")
    print(f"Single trial - Max geometric error: {np.max(errors)}")
    
    # Run multiple trials and get statistics
    stats = run_multiple_trials(config, num_trials=100)
    
    # Print statistics
    print("\nError Statistics across 100 trials:")
    print(f"Mean Error - Average: {stats['mean_error']['average']:.6f}, "
          f"Median: {stats['mean_error']['median']:.6f}, "
          f"StdDev: {stats['mean_error']['std_dev']:.6f}")
    print(f"Max Error - Average: {stats['max_error']['average']:.6f}, "
          f"Median: {stats['max_error']['median']:.6f}, "
          f"StdDev: {stats['max_error']['std_dev']:.6f}")

    # --------------------- PLOTTING -------------------
    # Generate output filename postfix based on configuration
    postfix = generate_postfix(config)

    # Save visualization of epipolar lines
    fig = draw_epipolar_lines(
        img1,
        img2,
        pts1,
        pts2,
        F,
    )
    fig.savefig(f"./save/epipolar_{postfix}.png")

    # Save visualization of point correspondences
    plt.figure(figsize=(10, 5))
    plt.imshow(match_img)
    plt.axis("off")
    plt.savefig(f"./save/correspondeces_{postfix}.png")
    
    # Optionally, save the statistics to a file
    with open(f"./save/error_stats_{postfix}.json", "w") as f:
        json.dump({k: v for k, v in stats.items() if k != "raw_data"}, f, indent=4)