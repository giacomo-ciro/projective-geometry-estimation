import re
import PIL
import numpy as np
import matplotlib.pyplot as plt
import PIL
import sys

# Parse command line
if len(sys.argv) != 5:
    print(f'SYNOPSIS: python run main.py path/to/image.jpeg path/to/annotations.txt path/to/output.png <reference_length>')
    sys.exit()
else:
    IMAGE_PATH = sys.argv[1]
    ANNOTATIONS_PATH = sys.argv[2]
    SAVE_PATH = sys.argv[3]
    REFERENCE_LENGTH = float(sys.argv[4])
    

# Read annotations
img = PIL.Image.open(IMAGE_PATH)
img = np.array(img)

with open(ANNOTATIONS_PATH, 'r') as f:
    annotations = f.read()
    
    # remove comments
    annotations = re.sub(r"#.*", '', annotations)

    # remove whitespace
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

# to store the name and homogeneous coordinates of points & lines        
obj = {}        # str:np.ndarray

obj["dave_top"] = np.array([xs[8], ys[8], 1.0])
obj["dave_bottom"] = np.array([xs[9], ys[9], 1.0])
obj["jack_top"] = np.array([xs[10], ys[10], 1.0])
obj["jack_bottom"] = np.array([xs[11], ys[11], 1.0])

# Retrieve parallel lines
n = 1
for i in [0, 2, 3, 6]:
    obj[f"parallel_{n}"] = cross(i, i+1)
    n += 1

# Retrieve reference and measure lines
obj["dave_line"] = np.cross(
    obj["dave_top"],
    obj["dave_bottom"]
)
obj["jack_line"] = np.cross(
    obj["jack_top"],
    obj["jack_bottom"]
)

# Find Vanishing Points & Horizon
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
obj["bottom_line"] = np.cross(
    obj["dave_bottom"],     
    obj["jack_bottom"]      
)

obj["p_inf"] = np.cross(
    obj["bottom_line"], 
    obj["horizon"]
)

# Project jack's height onto dave's
obj["top_line"] = np.cross(
    obj["p_inf"],
    obj["jack_top"] 
)

obj["jack_top_projected"] = np.cross(
    obj["top_line"], 
    obj["dave_line"]
)
obj["jack_top_projected"] = obj["jack_top_projected"] / obj["jack_top_projected"][2]

# Compute their ratio and estimate original height
dave_length = np.linalg.norm(
    obj["dave_top"] - obj["dave_bottom"]
)

jack_length_projected = np.linalg.norm(
    obj["jack_top_projected"] - obj["dave_bottom"]
)

# jack : dave = jack_img : dave_img
ans = jack_length_projected * REFERENCE_LENGTH / dave_length

# ----------- Plot
plt.imshow(img)

# Plot lines
for i,c in {
    "top_line":"pink",
    "bottom_line":"pink",
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

# Plot points:
for i, c in {
    ("jack_top", "jack_bottom", "jack_line"):"yellow",
    ("dave_top", "dave_bottom", "dave_line"):"orange",
    ("jack_top_projected", "dave_bottom", "jack_line_projected"):"brown",
}.items():
    plt.plot(
        (obj[i[0]][0],obj[i[1]][0]),
        (obj[i[0]][1],obj[i[1]][1]),
        c = c,
        marker = 'x',
        label = i[2]
    )
    
plt.title(f"Estimated Length = {ans:.2f} cm")
plt.axis("off")
plt.legend(loc='upper center', bbox_to_anchor=(1,1), ncol=1)
plt.xlim([0, img.shape[1]])
plt.ylim([img.shape[0], 0])

# Save
plt.savefig(SAVE_PATH)
print(f'Result saved to {SAVE_PATH}')

