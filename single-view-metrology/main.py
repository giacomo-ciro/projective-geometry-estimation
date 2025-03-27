import re
import PIL
import numpy as np
import matplotlib.pyplot as plt
import sys

# Parse command line
if len(sys.argv) != 5:
    print('USAGE:\n\tpython3 main.py path/to/input_image.jpeg path/to/annotations.txt path/to/output_image.jpeg <reference_length_in_cm>')
    sys.exit()
else:
    IMAGE_PATH = sys.argv[1]
    ANNOTATIONS_PATH = sys.argv[2]
    SAVE_PATH = sys.argv[3]
    REFERENCE_LENGTH = float(sys.argv[4])
    TRUE_LENGTH = 184
    
# Read annotations
img = PIL.Image.open(IMAGE_PATH)
img = np.array(img)

with open(ANNOTATIONS_PATH, 'r') as f:
    annotations = f.read()
    
    # Remove comments
    annotations = re.sub(r"#.*", '', annotations)

    # Remove whitespace
    annotations = re.sub(r" ", '', annotations)

    # Retrieve coordinates
    annotations = [xy.split(',') for xy in annotations.splitlines() if re.match(r"^\d+,\d+$", xy)]
    xs = np.array([int(xy[0]) for xy in annotations], dtype = np.float32)
    ys = np.array([int(xy[1]) for xy in annotations], dtype = np.float32)

def cross(p1, p2):
    """
    Compute homogeneous cross prouduct between 2D points by appending the 1.0 coordinate.
    """
    ans = np.cross(
        np.array([xs[p1], ys[p1], 1.0]),
        np.array([xs[p2], ys[p2], 1.0])
    )

    return ans

# To store the name and homogeneous coordinates of points & lines        
obj = {}        # str:np.ndarray

obj["person_1_top"] = np.array([xs[8], ys[8], 1.0])
obj["person_1_bottom"] = np.array([xs[9], ys[9], 1.0])
obj["person_2_top"] = np.array([xs[10], ys[10], 1.0])
obj["person_2_bottom"] = np.array([xs[11], ys[11], 1.0])

# Retrieve parallel lines
n = 1
for i in [0, 2, 4, 6]:
    obj[f"point_{n}_1"] = np.array([xs[i], ys[i], 1.0])
    obj[f"point_{n}_2"] = np.array([xs[i+1], ys[i+1], 1.0])
    obj[f"parallel_{n}"] = cross(i, i+1)
    n += 1

# Retrieve reference and measure lines
obj["person_1_line"] = np.cross(
    obj["person_1_top"],
    obj["person_1_bottom"]
)
obj["person_2_line"] = np.cross(
    obj["person_2_top"],
    obj["person_2_bottom"]
)

# Find vanishing points and horizon
obj["left_vp"] = np.cross(
    obj["parallel_1"],
    obj["parallel_2"]
)

obj["right_vp"] = np.cross(
    obj["parallel_3"],
    obj["parallel_4"]
)

obj["horizon"] = np.cross(
    obj["left_vp"],
    obj["right_vp"]
)

# Intersect with line through the feet
obj["feet_line"] = np.cross(
    obj["person_1_bottom"],     
    obj["person_2_bottom"]      
)

obj["p_inf"] = np.cross(
    obj["feet_line"], 
    obj["horizon"]
)

# Project person_2's height onto person_1's
obj["heads_line"] = np.cross(
    obj["p_inf"],
    obj["person_2_top"] 
)

obj["person_2_top_projected"] = np.cross(
    obj["heads_line"], 
    obj["person_1_line"]
)
obj["person_2_top_projected"] = obj["person_2_top_projected"] / obj["person_2_top_projected"][2]

# Compute their ratio and estimate original height
person_1_length = np.linalg.norm(
    obj["person_1_top"] - obj["person_1_bottom"]
)

person_2_length_projected = np.linalg.norm(
    obj["person_2_top_projected"] - obj["person_1_bottom"]
)

# person_2 : person_1 = person_2_img : person_1_img
ans = person_2_length_projected * REFERENCE_LENGTH / person_1_length

# Plot image
plt.imshow(img)
print(obj.keys())
for i, c in enumerate(['red', 'red', 'blue', 'blue']):
    for j in [1, 2]:
        plt.plot(
            obj[f"point_{i+1}_{j}"][0],
            obj[f"point_{i+1}_{j}"][1],
            marker = 'x',
            color = c,
            linestyle = "",
        )
        plt.annotate(
            f"{i}-{j}",
            xy = (
                obj[f"point_{i+1}_{j}"][0],
                obj[f"point_{i+1}_{j}"][1],
            ),
            xytext=(
                obj[f"point_{i+1}_{j}"][0]-30,
                obj[f"point_{i+1}_{j}"][1]-30,
            ),
            color = c,
            fontsize = 8,
        )
# Plot lines
for i,c in {
    "heads_line":"pink",
    "feet_line":"pink",
    "horizon":"green",
    "parallel_1":"red",
    "parallel_2":"red",
    "parallel_3":"blue",
    "parallel_4":"blue",
}.items():
    
    xs = np.array([0, img.shape[1]])
    a = obj[i]
    slope = -a[0] / a[1]
    intercept = - a[2] / a[1]
    
    ys = slope * xs + intercept
    
    plt.plot(
        xs, 
        ys,
        marker = '',
        # linestyle = ':',
        linewidth = 0.5,
        label = i,
        c = c
    )

# Plot points
for i, c in {
    ("person_2_top", "person_2_bottom", "person_2_line"):"yellow",
    ("person_1_top", "person_1_bottom", "person_1_line"):"purple",
    ("person_2_top_projected", "person_1_bottom", "person_2_line_projected"):"yellow",
}.items():
    plt.plot(
        (obj[i[0]][0],obj[i[1]][0]),
        (obj[i[0]][1],obj[i[1]][1]),
        c = c,
        marker = 'x',
        label = i[2],
        linestyle=":" if "projected" in i[0] else "-"
    )
    
plt.title(
    f"Person_1 = {REFERENCE_LENGTH:.2f} cm\n"
    f"Person_2 = {ans:.2f} cm (true is {TRUE_LENGTH:.2f} cm)"
    )
plt.axis("off")
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
plt.xlim([0, img.shape[1]])
plt.ylim([img.shape[0], 0])
plt.tight_layout()

# Save
plt.savefig(SAVE_PATH)
print(f'Result saved to {SAVE_PATH}')