import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import random

def detect_and_match_features(img1, img2, num_matches = 100):

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    good_matches = good_matches[:num_matches]
    
    # extract location of matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # convert to homogeneous coordinates
    pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homogeneous = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    
    # create an image with the matches drawn
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return pts1_homogeneous, pts2_homogeneous, match_img

def normalize_points(points):

    centroid = np.mean(points[:, :2], axis=0)

    centered_points = points[:, :2] - centroid
    mean_dist = np.mean(np.sqrt(np.sum(centered_points**2, axis=1)))

    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    
    # transformation matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    normalized_points = np.dot(T, points.T).T
    
    return normalized_points, T

def compute_fundamental_matrix(pts1, pts2, normalized=True):

    if normalized:
        pts1_norm, T1 = normalize_points(pts1)
        pts2_norm, T2 = normalize_points(pts2)
    else:
        pts1_norm, pts2_norm = pts1, pts2
        T1, T2 = np.eye(3), np.eye(3)
    
    # A is the contraint matrix
    n_points = pts1_norm.shape[0]
    A = np.zeros((n_points, 9))
    
    for i in range(n_points):
        x1, y1, _ = pts1_norm[i]
        x2, y2, _ = pts2_norm[i]
        
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    
    # we compute the SVD
    _, _, Vt = np.linalg.svd(A)
    
    # retrieve last column 
    F_flat = Vt[-1]
    
    F_norm = F_flat.reshape(3, 3)
    
    # enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0  
    F_norm = U @ np.diag(S) @ Vt

    if normalized:
        F = T2.T @ F_norm @ T1
    else:
        F = F_norm

    # normalize F to make F[2,2] = 1
    F = F / F[2, 2] if F[2, 2] != 0 else F
    
    return F

def compute_epipolar_lines(F, points, which='left'):

    if which == 'left':
        # lines in the left image
        lines = np.dot(F.T, points.T).T
    else:
        # lines in the right image
        lines = np.dot(F, points.T).T
    
    # normalize to make the first two elements have unit norm
    norm = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2).reshape(-1, 1)
    lines = lines / norm
    
    return lines

def draw_epipolar_lines(img1, img2, pts1, pts2, F, figsize=(15, 10)):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    lines2 = compute_epipolar_lines(F, pts1, which='right')
    
    colors = cm.rainbow(np.linspace(0, 1, len(pts1)))
    
    for i, (pt1, line, color) in enumerate(zip(pts1, lines2, colors)):
        x1, y1, _ = pt1
        ax1.scatter(x1, y1, c=color.reshape(1, -1), s=50)
        
        a, b, c = line
        if abs(b) > 1e-8:  # non-vertical line
            x_start, x_end = 0, w2
            y_start = (-c - a * x_start) / b
            y_end = (-c - a * x_end) / b
        else:  # vertical line
            y_start, y_end = 0, h2
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0
        
        ax2.plot([x_start, x_end], [y_start, y_end], c=color)
        
        x2, y2, _ = pts2[i]
        ax2.scatter(x2, y2, c=color.reshape(1, -1), s=50)

    lines1 = compute_epipolar_lines(F, pts2, which='left')
    
    for i, (pt2, line, color) in enumerate(zip(pts2, lines1, colors)):

        a, b, c = line
        if abs(b) > 1e-8: 
            x_start, x_end = 0, w1
            y_start = (-c - a * x_start) / b
            y_end = (-c - a * x_end) / b
        else: 
            y_start, y_end = 0, h1
            x_start = x_end = -c / a if abs(a) > 1e-8 else 0
        
        ax1.plot([x_start, x_end], [y_start, y_end], c=color)
    
    ax1.set_title('Image 1 with Epipolar Lines')
    ax2.set_title('Image 2 with Epipolar Lines')

    ax1.axis('off')
    ax2.axis('off')
    
    plt.tight_layout()

    return fig

def compute_geometric_error(pts1, pts2, F):
    
    lines1 = compute_epipolar_lines(F, pts2, which='left')
    lines2 = compute_epipolar_lines(F, pts1, which='right')

    errors = []
    
    for i in range(pts1.shape[0]): 
        x1, y1, _ = pts1[i]
        a1, b1, c1 = lines1[i]
        dist1 = abs(a1*x1 + b1*y1 + c1)
        
        x2, y2, _ = pts2[i]
        a2, b2, c2 = lines2[i]
        dist2 = abs(a2*x2 + b2*y2 + c2)

        error = (dist1 + dist2) / 2
        errors.append(error)
    
    return np.array(errors)

def ransac_fundamental_matrix(pts1, pts2, iterations = 1000, threshold = 2.0, adaptive = True, final_percentile = 0.7):

    best_F = None
    best_inliers = []
    num_points = pts1.shape[0]
    
    if not adaptive:
        for _ in range(iterations):
            indices = random.sample(range(num_points), 8)
            F = compute_fundamental_matrix(pts1[indices], pts2[indices])
            errors = compute_geometric_error(pts1, pts2, F)
            inliers = np.where(errors < threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_F = F
    else:
        for _ in range(iterations // 2): 
            indices = random.sample(range(num_points), 8)
            F = compute_fundamental_matrix(pts1[indices], pts2[indices])
            errors = compute_geometric_error(pts1, pts2, F)
            inliers = np.where(errors < threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_F = F
        
        if len(best_inliers) >= 8:
            refined_F = compute_fundamental_matrix(pts1[best_inliers], pts2[best_inliers])
            
            all_errors = compute_geometric_error(pts1, pts2, refined_F)

            sorted_errors = np.sort(all_errors)
            adaptive_threshold = sorted_errors[int(final_percentile * len(sorted_errors))]
            
            adaptive_threshold = max(0.5, min(adaptive_threshold, 3.0))
            
            print(f"Adaptive threshold: {adaptive_threshold:.4f} pixels")
            
            for _ in range(iterations // 2):  
                indices = random.sample(range(num_points), 8)
                F = compute_fundamental_matrix(pts1[indices], pts2[indices])
                errors = compute_geometric_error(pts1, pts2, F)
                inliers = np.where(errors < adaptive_threshold)[0]
                
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_F = F
    
    if len(best_inliers) >= 8:
        best_F = compute_fundamental_matrix(pts1[best_inliers], pts2[best_inliers])
    else:
        print(len(best_inliers))    
    
    return best_F, best_inliers

def eight_point_algorithm(img1_path, img2_path, use_ransac = True, num_matches = 100):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    pts1, pts2, match_img = detect_and_match_features(img1, img2, num_matches)
    
    if use_ransac:
        F, inliers = ransac_fundamental_matrix(pts1, pts2, iterations = 1000, adaptive = True)
        pts1_inliers = pts1[inliers]
        pts2_inliers = pts2[inliers]
        errors = compute_geometric_error(pts1_inliers, pts2_inliers, F)
        
        print(f"RANSAC found {len(inliers)} inliers out of {len(pts1)} points")
        print(f"Mean geometric error: {np.mean(errors):.4f}")
        print(f"Max geometric error: {np.max(errors):.4f}")
        
        fig = draw_epipolar_lines(img1, img2, pts1_inliers, pts2_inliers, F)
        
        return F, pts1_inliers, pts2_inliers, match_img, errors, fig
    else:
        F = compute_fundamental_matrix(pts1, pts2)
        errors = compute_geometric_error(pts1, pts2, F)
        
        print(f"Mean geometric error: {np.mean(errors):.4f}")
        print(f"Max geometric error: {np.max(errors):.4f}")
        
        fig = draw_epipolar_lines(img1, img2, pts1, pts2, F)
        
        return F, pts1, pts2, match_img, errors, fig

# Example usage:
F, pts1, pts2, match_img, errors, fig = eight_point_algorithm('giratina_1.jpeg', 'giratina_2.jpeg')
plt.figure(figsize=(10, 5))
plt.imshow(match_img)
plt.title('SIFT Matches')
plt.axis('off')
plt.show()