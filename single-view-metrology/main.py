import re
import PIL
import numpy as np
import matplotlib.pyplot as plt
import PIL
import sys

# Parse command line
if len(sys.argv) != 4:
    print(f'SYNOPSIS: python run main.py path/to/image.jpeg path/to/annotations.txt <reference_length>')
    sys.exit()
else:
    IMAGE_PATH = sys.argv[1]
    ANNOTATIONS_PATH = sys.argv[2]
    REFERENCE_LENGTH = float(sys.argv[3])

# Read annotations
img = PIL.Image.open(IMAGE_PATH)

with open(ANNOTATIONS_PATH, 'r') as f:
    annotations = f.read()
    
    annotations = re.sub(r"#.*", '', annotations)
    annotations = [xy.split(',') for xy in annotations.splitlines() if re.match(r"^\d+,\d+$", xy)]
    xs = np.array([int(xy[0]) for xy in annotations], dtype = np.float32)
    ys = np.array([int(xy[1]) for xy in annotations], dtype = np.float32)
    labels = np.arange(0, len(xs))

def cross(p1, p2):
    
    ans = np.cross(
        np.array([xs[p1], ys[p1], 1.0]),
        np.array([xs[p2], ys[p2], 1.0])
    )

    return ans

dave_top = np.array([xs[8], ys[8], 1.0])
dave_bottom = np.array([xs[9], ys[9], 1.0])

left_vp = np.cross(
    cross(0, 1),
    cross(2, 3)
)

right_vp = np.cross(
    cross(4, 5),
    cross(6, 7)
)

horizon = np.cross(
    left_vp,
    right_vp
)

bottom_line = cross(
    9,      # Dave's bottom point
    11      # Jack's bottom point
)

p_inf = np.cross(
    bottom_line, 
    horizon
)

top_line = np.cross(
    p_inf,
    np.array([xs[10], ys[10], 1.0])     # jack's top point
)

dave_line = np.cross(
    dave_top,
    dave_bottom
)

jack_top_projected = np.cross(
    top_line, 
    dave_line
)
jack_top_projected = jack_top_projected / jack_top_projected[2]

dave_length = np.linalg.norm(
    dave_top - dave_bottom
)

jack_length_projected = np.linalg.norm(
    jack_top_projected - dave_bottom
)

# jack : dave = jack_img : dave_img
ans = jack_length_projected * REFERENCE_LENGTH / dave_length

plt.imshow(img)
plt.plot(
    jack_top_projected[0],
    jack_top_projected[1],
    c = 'r',
    marker = 'x')
plt.plot(
    dave_bottom[0],
    dave_bottom[1],
    c = 'r',
    marker = 'x')
plt.plot(
    dave_top[0],
    dave_top[1],
    c = 'r',
    marker = 'x')
plt.title(f"Estimated Length = {ans:.2f} cm")
plt.axis("off")
plt.savefig('plot.png')

