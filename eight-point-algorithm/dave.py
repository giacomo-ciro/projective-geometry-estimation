import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse

def read_correspondences(file_path):
    """
    Read point correspondences from a file.
    """
    pts1 = []
    pts2 = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#'):
                continue
            
            # Parse the coordinates
            try:
                values = line.strip().split()
                if len(values) == 4:
                    x1, y1, x2, y2 = map(float, values)
                    pts1.append([x1, y1, 1.0])  # Homogeneous coordinates
                    pts2.append([x2, y2, 1.0])  # Homogeneous coordinates
            except ValueError:
                continue
    
    return np.array(pts1), np.array(pts2)

def detect_and_match_features(img1, img2, num_matches=10):
    """
    Extract correspondences between two images using SIFT.
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors using kNN
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test (Lowe's)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Sort matches by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    # Take only the specified number of matches
    good_matches = good_matches[:num_matches]
    
    # Extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Convert to homogeneous coordinates
    pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homogeneous = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    
    # Create an image with the matches drawn
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return pts1_homogeneous, pts2_homogeneous, match_img

def normalize_points(points):
    """
    Normalize points for improved numerical stability.
    """
    # Centroid of points (ignoring homogeneous coordinate)
    centroid = np.mean(points[:, :2], axis=0)
    
    # Center the points
    centered_points = points[:, :2] - centroid
    
    # Average distance from origin
    dist = np.sqrt(np.sum(centered_points**2, axis=1))
    mean_dist = np.mean(dist)
    
    # Scale factor
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    
    # Normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # Apply transformation
    normalized_points = (T @ points.T).T
    
    return normalized_points, T

def compute_fundamental_matrix(pts1, pts2, normalized=True):
    """
    Compute the fundamental matrix using the eight-point algorithm.
    """
    # Normalize points if requested
    if normalized:
        pts1_norm, T1 = normalize_points(pts1)
        pts2_norm, T2 = normalize_points(pts2)
    else:
        pts1_norm, pts2_norm = pts1, pts2
        T1, T2 = np.eye(3), np.eye(3)
    
    # Number of points
    n = pts1_norm.shape[0]
    
    # Build the constraint matrix A
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1, _ = pts1_norm[i]
        x2, y2, _ = pts2_norm[i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    
    # Solve for F using SVD
    _, _, Vt = np.linalg.svd(A)
    
    # The solution is the last row of Vt
    F_flat = Vt[-1]
    
    # Reshape into 3x3 matrix
    F_norm = F_flat.reshape(3, 3)
    
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0  # Set smallest singular value to 0
    F_norm = U @ np.diag(S) @ Vt
    
    # Denormalize
    if normalized:
        F = T2.T @ F_norm @ T1
    else:
        F = F_norm
    
    # Normalize F to make F[2,2] = 1
    F = F / F[2, 2] if F[2, 2] != 0 else F
    
    return F

def compute_epipolar_lines(F, points, which='left'):
    """
    Compute epipolar lines for points.
    """
    if which == 'left':
        # Lines in left image: l = F^T * x2
        lines = (F.T @ points.T).T
    else:
        # Lines in right image: l' = F * x1
        lines = (F @ points.T).T
    
    # Normalize lines to have a^2 + b^2 = 1
    norms = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2).reshape(-1, 1)
    lines = lines / norms
    
    return lines

def compute_geometric_error(pts1, pts2, F):
    """
    Compute the geometric error for point correspondences.
    """
    # Compute epipolar lines
    lines1 = compute_epipolar_lines(F, pts2, 'left')
    lines2 = compute_epipolar_lines(F, pts1, 'right')
    
    errors = []
    
    # For each pair of corresponding points
    for i in range(pts1.shape[0]):
        # Distance from point in image 1 to its epipolar line
        x1, y1, _ = pts1[i]
        a1, b1, c1 = lines1[i]
        dist1 = abs(a1*x1 + b1*y1 + c1)
        
        # Distance from point in image 2 to its epipolar line
        x2, y2, _ = pts2[i]
        a2, b2, c2 = lines2[i]
        dist2 = abs(a2*x2 + b2*y2 + c2)
        
        # Average the distances
        error = (dist1 + dist2) / 2
        errors.append(error)
    
    return np.array(errors)

def draw_epipolar_lines(img1, img2, pts1, pts2, F, figsize=(15, 10)):
    """
    Draw epipolar lines and points on images.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Display images
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    
    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Set limits
    ax1.set_xlim(0, w1)
    ax1.set_ylim(h1, 0)
    ax2.set_xlim(0, w2)
    ax2.set_ylim(h2, 0)
    
    # Compute epipolar lines in right image
    lines2 = compute_epipolar_lines(F, pts1, 'right')
    
    # Generate colors for points and lines
    colors = cm.rainbow(np.linspace(0, 1, len(pts1)))
    
    # Draw points and lines
    for i, (pt1, pt2, line, color) in enumerate(zip(pts1, pts2, lines2, colors)):
        # Draw point in left image
        ax1.scatter(pt1[0], pt1[1], c=color.reshape(1, -1), s=50)
        
        # Draw corresponding point in right image
        ax2.scatter(pt2[0], pt2[1], c=color.reshape(1, -1), s=50)
        
        # Draw epipolar line in right image
        a, b, c = line
        if abs(b) > 1e-8:
            # Line is not vertical
            x_start, x_end = 0, w2
            y_start = (-c - a * x_start) / b
            y_end = (-c - a * x_end) / b
        else:
            # Line is vertical
            y_start, y_end = 0, h2
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0
        
        ax2.plot([x_start, x_end], [y_start, y_end], c=color)
    
    # Compute epipolar lines in left image
    lines1 = compute_epipolar_lines(F, pts2, 'left')
    
    # Draw epipolar lines in left image
    for i, (pt2, line, color) in enumerate(zip(pts2, lines1, colors)):
        a, b, c = line
        if abs(b) > 1e-8:
            # Line is not vertical
            x_start, x_end = 0, w1
            y_start = (-c - a * x_start) / b
            y_end = (-c - a * x_end) / b
        else:
            # Line is vertical
            y_start, y_end = 0, h1
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0
        
        ax1.plot([x_start, x_end], [y_start, y_end], c=color)
    
    # Add titles
    ax1.set_title('Image 1 with Epipolar Lines')
    ax2.set_title('Image 2 with Epipolar Lines')
    
    # Turn off axes
    ax1.axis('off')
    ax2.axis('off')
    
    plt.tight_layout()
    
    return fig

def ransac_fundamental_matrix(pts1, pts2, iterations=1000, threshold=2.0, adaptive=True, final_percentile=0.7):
    """
    Estimate the fundamental matrix using RANSAC.
    """
    import random
    
    best_F = None
    best_inliers = []
    num_points = pts1.shape[0]
    
    for _ in range(iterations):
        # Randomly sample 8 points
        if num_points >= 8:
            indices = random.sample(range(num_points), 8)
        else:
            print(f"Not enough points: {num_points}")
            return compute_fundamental_matrix(pts1, pts2), np.arange(num_points)
        
        # Compute F from these points
        F = compute_fundamental_matrix(pts1[indices], pts2[indices])
        
        # Calculate error for all points
        errors = compute_geometric_error(pts1, pts2, F)
        
        # Find inliers
        inliers = np.where(errors < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F
    
    # If adaptive thresholding is enabled
    if adaptive and len(best_inliers) >= 8:
        # Recompute F using all inliers
        refined_F = compute_fundamental_matrix(pts1[best_inliers], pts2[best_inliers])
        
        # Calculate errors with the refined F
        all_errors = compute_geometric_error(pts1, pts2, refined_F)
        
        # Set adaptive threshold based on error distribution
        sorted_errors = np.sort(all_errors)
        adaptive_idx = min(int(final_percentile * len(sorted_errors)), len(sorted_errors) - 1)
        adaptive_threshold = sorted_errors[adaptive_idx]
        
        # Clamp threshold to reasonable values
        adaptive_threshold = max(0.5, min(adaptive_threshold, 3.0))
        
        print(f"Adaptive threshold: {adaptive_threshold:.4f} pixels")
        
        # Update inliers with new threshold
        best_inliers = np.where(all_errors < adaptive_threshold)[0]
        best_F = refined_F
    
    # Final refinement using all inliers
    if len(best_inliers) >= 8:
        best_F = compute_fundamental_matrix(pts1[best_inliers], pts2[best_inliers])
    
    return best_F, best_inliers

def dave(img1_path, img2_path, correspondences_path=None, use_sift=False, use_ransac=False, num_points=10):
    """
    Main function to run the 8-point algorithm.
    """
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not read images {img1_path} and/or {img2_path}")
        return
    
    # Get point correspondences
    if use_sift:
        print("Using SIFT to extract correspondences")
        pts1, pts2, match_img = detect_and_match_features(img1, img2, num_points)
    else:
        print(f"Reading correspondences from {correspondences_path}")
        pts1, pts2 = read_correspondences(correspondences_path)
        # Limit to specified number of points
        pts1 = pts1[:num_points]
        pts2 = pts2[:num_points]
        
        # Create match visualization
        kp1 = [cv2.KeyPoint(x, y, 1) for x, y, _ in pts1]
        kp2 = [cv2.KeyPoint(x, y, 1) for x, y, _ in pts2]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(pts1))]
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    
    print(f"Using {len(pts1)} point correspondences")
    
    # Compute fundamental matrix
    if use_ransac:
        print("Using RANSAC for robust estimation")
        F, inliers = ransac_fundamental_matrix(pts1, pts2)
        pts1_inliers = pts1[inliers]
        pts2_inliers = pts2[inliers]
        print(f"RANSAC found {len(inliers)} inliers out of {len(pts1)} points")
        
        # Compute errors
        errors = compute_geometric_error(pts1_inliers, pts2_inliers, F)
        
        # Draw epipolar lines
        fig = draw_epipolar_lines(img1, img2, pts1_inliers, pts2_inliers, F)
    else:
        print("Using standard 8-point algorithm")
        F = compute_fundamental_matrix(pts1, pts2)
        
        # Compute errors
        errors = compute_geometric_error(pts1, pts2, F)
        
        # Draw epipolar lines
        fig = draw_epipolar_lines(img1, img2, pts1, pts2, F)
    
    # Print matrix and error statistics
    print("\nEstimated fundamental matrix:")
    print(F)
    print(f"\nMean geometric error: {np.mean(errors):.4f}")
    print(f"Max geometric error: {np.max(errors):.4f}")
    
    # Draw matches
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title("Point Correspondences")
    plt.axis('off')
    plt.savefig("matches.png")
    
    # Save epipolar visualization
    fig.savefig("epipolar.png")
    
    # Show figures
    plt.show()
    
    return F, pts1, pts2, errors, fig

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Eight-point algorithm for fundamental matrix estimation')
    parser.add_argument('--img1', default='img/giratina_occhiali_1.jpeg', help='Path to first image')
    parser.add_argument('--img2', default='img/giratina_occhiali_2.jpeg', help='Path to second image')
    parser.add_argument('--correspondences', default='correspondences_occhiali.txt', help='Path to correspondences file')
    parser.add_argument('--use_sift', action='store_true', help='Use SIFT instead of manual correspondences')
    parser.add_argument('--use_ransac', action='store_true', help='Use RANSAC for robust estimation')
    parser.add_argument('--num_points', type=int, default=10, help='Number of correspondences to use')
    args = parser.parse_args()
    
    # Run the algorithm
    dave(args.img1, args.img2, args.correspondences, args.use_sift, args.use_ransac, args.num_points)