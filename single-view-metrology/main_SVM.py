import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_arguments():
    """
    Parse and validate command line arguments.

    Returns:
        tuple: (image_path, annotations_path, save_path, reference_length)

    Exits with error message if arguments are incorrect.
    """
    if len(sys.argv) != 5:
        print(
            "\nUSAGE:\n\tpython3 main_SVM.py path/to/input_image.jpeg path/to/annotations.txt path/to/output_image.jpeg <reference_length_in_cm>"
        )
        print(
            "\nEXAMPLE:\n\tpython3 main_SVM.py img/img2.jpeg annotations/annotations_img2.txt outputs/output_img2.png 175\n"
        )
        sys.exit(1)

    return sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])


def load_annotations(annotations_path):
    """
    Load and parse point coordinates from the annotations file.

    Args:
        annotations_path (str): Path to the annotations file

    Returns:
        tuple: Arrays of x and y coordinates
    """
    with open(annotations_path, "r") as f:
        annotations = f.read()

        # Remove comments and whitespace
        annotations = re.sub(r"#.*", "", annotations)
        annotations = re.sub(r" ", "", annotations)

        # Parse coordinates
        valid_points = [
            xy.split(",")
            for xy in annotations.splitlines()
            if re.match(r"^\d+,\d+$", xy)
        ]

        xs = np.array([int(xy[0]) for xy in valid_points], dtype=np.float32)
        ys = np.array([int(xy[1]) for xy in valid_points], dtype=np.float32)

    return xs, ys


def compute_geometric_entities(xs, ys):
    """
    Compute various geometric entities needed for height estimation using the annotated points.

    Args:
        xs (np.ndarray): Array of x coordinates
        ys (np.ndarray): Array of y coordinates

    Returns:
        dict: Dictionary containing all computed geometric entities
    """
    # Dictionary to store homogeneous coordinates of points & lines
    obj = {}

    # Extract person reference points
    obj["person_1_top"] = np.array([xs[8], ys[8], 1.0])
    obj["person_1_bottom"] = np.array([xs[9], ys[9], 1.0])
    obj["person_2_top"] = np.array([xs[10], ys[10], 1.0])
    obj["person_2_bottom"] = np.array([xs[11], ys[11], 1.0])

    # Extract parallel lines from the first 8 points (4 pairs)
    for n, i in enumerate([0, 2, 4, 6]):
        obj[f"point_{n + 1}_1"] = np.array([xs[i], ys[i], 1.0])
        obj[f"point_{n + 1}_2"] = np.array([xs[i + 1], ys[i + 1], 1.0])
        obj[f"parallel_{n + 1}"] = np.cross(
            np.array([xs[i], ys[i], 1.0]),
            np.array([xs[i+1], ys[i+1], 1.0])
            )

    # Define the vertical lines for each person
    obj["person_1_line"] = np.cross(obj["person_1_top"], obj["person_1_bottom"])
    obj["person_2_line"] = np.cross(obj["person_2_top"], obj["person_2_bottom"])

    # Find vanishing points from parallel lines and compute the horizon
    obj["left_vp"] = np.cross(obj["parallel_1"], obj["parallel_2"])
    obj["right_vp"] = np.cross(obj["parallel_3"], obj["parallel_4"])
    obj["horizon"] = np.cross(obj["left_vp"], obj["right_vp"])

    # Compute the line through the feet of both persons
    obj["feet_line"] = np.cross(obj["person_1_bottom"], obj["person_2_bottom"])

    # Find the point at infinity (where feet line meets horizon)
    obj["p_inf"] = np.cross(obj["feet_line"], obj["horizon"])

    # Project person_2's height onto person_1's position
    obj["heads_line"] = np.cross(obj["p_inf"], obj["person_2_top"])
    obj["person_2_top_projected"] = np.cross(obj["heads_line"], obj["person_1_line"])
    # Normalize the homogeneous coordinates
    obj["person_2_top_projected"] = (
        obj["person_2_top_projected"] / obj["person_2_top_projected"][2]
    )

    return obj


def calculate_height(obj, reference_length):
    """
    Calculate the height of person_2 using the projected height and reference length.

    Args:
        obj (dict): Dictionary containing geometric entities
        reference_length (float): Known height of person_1 in cm

    Returns:
        float: Estimated height of person_2 in cm
    """
    # Compute the length of person_1 in image space
    person_1_length = np.linalg.norm(obj["person_1_top"] - obj["person_1_bottom"])

    # Compute the projected length of person_2 in image space
    person_2_length_projected = np.linalg.norm(
        obj["person_2_top_projected"] - obj["person_1_bottom"]
    )

    # Apply the ratio to calculate person_2's height
    return person_2_length_projected * reference_length / person_1_length


def visualize_results(img, obj, reference_length, estimated_height, true_length, save_path):
    """
    Visualize the results by plotting geometric entities on the image.

    Args:
        img (np.ndarray): Input image
        obj (dict): Dictionary containing geometric entities
        reference_length (float): Known height of person_1 in cm
        estimated_height (float): Estimated height of person_2 in cm
        true_length (float): True height of person_2 in cm (for comparison)
        save_path (str): Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Plot the parallel line points
    for i, c in enumerate(["red", "red", "blue", "blue"]):
        for j in [1, 2]:
            point_key = f"point_{i + 1}_{j}"
            plt.plot(
                obj[point_key][0],
                obj[point_key][1],
                marker="x",
                color=c,
                linestyle="",
            )
            plt.annotate(
                f"{i}-{j}",
                xy=(obj[point_key][0], obj[point_key][1]),
                xytext=(obj[point_key][0] - 15, obj[point_key][1] - 15),
                color=c,
                fontsize=15,
            )

    # Plot the geometric lines
    line_colors = {
        "heads_line": "pink",
        "feet_line": "pink",
        "horizon": "green",
        "parallel_1": "red",
        "parallel_2": "red",
        "parallel_3": "blue",
        "parallel_4": "blue",
    }

    for line, color in line_colors.items():
        # Calculate the line equation ax + by + c = 0
        a = obj[line]

        # Plot the line across the image width
        xs = np.array([0, img.shape[1]])
        slope = -a[0] / a[1]
        intercept = -a[2] / a[1]
        ys = slope * xs + intercept

        plt.plot(xs, ys, linestyle="-", linewidth=0.5, label=line, color=color)

    # Plot the person heights
    person_lines = {
        ("person_2_top", "person_2_bottom", "person_2_line"): "yellow",
        ("person_1_top", "person_1_bottom", "person_1_line"): "purple",
        ("person_2_top_projected","person_1_bottom","person_2_line_projected"): "yellow",
    }

    for points, color in person_lines.items():
        plt.plot(
            (obj[points[0]][0], obj[points[1]][0]),
            (obj[points[0]][1], obj[points[1]][1]),
            color=color,
            marker="x",
            label=points[2],
            linestyle=":" if "projected" in points[0] else "-",
        )

    # Add title with measurement information
    plt.title(
        f"Person_1 = {reference_length:.2f} cm\n"
        f"Person_2 = {estimated_height:.2f} cm (true is {true_length:.2f} cm)"
    )

    # Finalize the plot
    plt.axis("off")
    plt.legend(loc="center", bbox_to_anchor=(0.5, 0), ncol=4, fontsize=14)
    plt.xlim([0, img.shape[1]])
    plt.ylim([img.shape[0], 0])
    plt.tight_layout()

    # Save the result
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")


def main():
    """
    Main function to orchestrate the height estimation process.
    """
    # Constants
    TRUE_LENGTH = 184  # True height of person_2 in cm (for verification)

    # Parse command line arguments
    image_path, annotations_path, save_path, reference_length = parse_arguments()

    # Load the image
    img = Image.open(image_path)
    img = np.array(img)

    # Load and parse annotations
    global xs, ys  # Make these global for the cross function
    xs, ys = load_annotations(annotations_path)

    # Compute geometric entities
    obj = compute_geometric_entities(xs, ys)

    # Calculate the height
    estimated_height = calculate_height(obj, reference_length)

    # Visualize and save the results
    visualize_results(
        img, obj, reference_length, estimated_height, TRUE_LENGTH, save_path
    )


if __name__ == "__main__":
    main()