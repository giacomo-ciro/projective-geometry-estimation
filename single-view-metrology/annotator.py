import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

class SimpleMetrologyAnnotator:
    def __init__(self, image_path, output_file='annotations_new.txt'):
        """Initialize with an image path and output file"""
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.annotations_filename = output_file
        
        # Store points and lines
        self.points = []
        self.lines = []
        self.current_line = []
        
        # Keep track of point types and line counters
        self.point_types = []  # 'parallel_1', 'parallel_2', 'dave', 'jack'
        self.current_type = 'parallel_1'
        self.line_counters = {
            'parallel_1': 0,
            'parallel_2': 0,
            'dave': 0,
            'jack': 0
        }
        
        # Colors for different point types
        self.colors = {
            'parallel_1': 'red',
            'parallel_2': 'green',
            'dave': 'blue',
            'jack': 'purple'
        }
        
        # Setup the figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.15)
        self.ax.imshow(self.img)
        self.ax.set_title(f"Annotation Mode: {self.current_type}")
        
        # Add buttons
        self.add_buttons()
        
        # Status message
        self.status_text = self.ax.text(
            0.5, 0.01, "Click to add points", 
            transform=self.fig.transFigure,
            horizontalalignment='center',
            color='black', fontsize=12
        )
        
        # Connect to mouse events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def add_buttons(self):
        """Add UI buttons"""
        # Type selection buttons
        ax_parallel_1 = plt.axes([0.05, 0.05, 0.15, 0.05])
        self.btn_parallel_1 = Button(ax_parallel_1, 'Parallel Set 1')
        self.btn_parallel_1.on_clicked(lambda event: self.set_type('parallel_1'))
        
        ax_parallel_2 = plt.axes([0.25, 0.05, 0.15, 0.05])
        self.btn_parallel_2 = Button(ax_parallel_2, 'Parallel Set 2')
        self.btn_parallel_2.on_clicked(lambda event: self.set_type('parallel_2'))
        
        ax_dave = plt.axes([0.45, 0.05, 0.15, 0.05])
        self.btn_dave = Button(ax_dave, 'Dave')
        self.btn_dave.on_clicked(lambda event: self.set_type('dave'))
        
        ax_jack = plt.axes([0.65, 0.05, 0.15, 0.05])
        self.btn_jack = Button(ax_jack, 'Jack')
        self.btn_jack.on_clicked(lambda event: self.set_type('jack'))
        
        # Save button
        ax_save = plt.axes([0.85, 0.05, 0.1, 0.05])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self.save_annotations)
    
    def set_type(self, point_type):
        """Set the current point type"""
        self.current_type = point_type
        self.ax.set_title(f"Annotation Mode: {self.current_type}")
        self.status_text.set_text(f"Mode: {self.current_type}. Click to add points.")
        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return
        
        x, y = int(round(event.xdata)), int(round(event.ydata))
        
        # Add point to the list
        self.points.append((x, y))
        self.point_types.append(self.current_type)
        
        # Plot the point
        self.ax.plot(x, y, 'o', color=self.colors[self.current_type], markersize=8)
        
        # Add label
        point_num = len(self.points) - 1
        self.ax.text(x + 10, y + 10, f"{point_num}", 
                   fontsize=12, color=self.colors[self.current_type])
        
        # Add to current line
        self.current_line.append((x, y, point_num))
        
        # If we have 2 points for a line type, we have a complete line
        # Dave and Jack are special types where each has top and bottom points
        if len(self.current_line) == 2:
            x1, y1, p1_idx = self.current_line[0]
            x2, y2, p2_idx = self.current_line[1]
            
            # Draw the line
            self.ax.plot([x1, x2], [y1, y2], '-', color=self.colors[self.current_type], linewidth=2)
            
            # Increment line counter for this type
            self.line_counters[self.current_type] += 1
            line_num = self.line_counters[self.current_type]
            
            # Save the line with its number and type
            self.lines.append({
                'type': self.current_type,
                'line_num': line_num,
                'point1': (x1, y1, p1_idx),
                'point2': (x2, y2, p2_idx)
            })
            
            # Reset current line
            self.current_line = []
        
        self.fig.canvas.draw_idle()
    
    def on_key_press(self, event):
        """Handle key presses"""
        if event.key == 'escape':
            # Cancel current line if escape is pressed
            self.current_line = []
            self.status_text.set_text("Current line canceled. Click to start a new line.")
            self.fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(self.fig)
    
    def save_annotations(self, event):
        """Save annotations to file in the specified format"""
        # Sort lines by type and line number
        parallel1_lines = [line for line in self.lines if line['type'] == 'parallel_1']
        parallel2_lines = [line for line in self.lines if line['type'] == 'parallel_2']
        dave_points = [line for line in self.lines if line['type'] == 'dave']
        jack_points = [line for line in self.lines if line['type'] == 'jack']
        
        # Write to file in the requested format
        with open(self.annotations_filename, 'w') as f:
            # Parallel Set 1
            f.write("# Parallel Set 1 (left vanishing point)\n")
            for i, line in enumerate(parallel1_lines, 1):
                f.write(f"# Line {i}\n")
                f.write(f"{line['point1'][0]},{line['point1'][1]}        # {i}_1\n")
                f.write(f"{line['point2'][0]},{line['point2'][1]}         # {i}_2\n")
            
            # Parallel Set 2
            f.write("# Parallel Set 2 (right vanishing point)\n")
            for i, line in enumerate(parallel2_lines, len(parallel1_lines) + 1):
                f.write(f"# Line {i}\n")
                f.write(f"{line['point1'][0]},{line['point1'][1]}         # {i}_1\n")
                point2_str = f"{line['point2'][0]},{line['point2'][1]}"
                
                # Check if this point is shared with any other line
                shared_points = []
                for other_line in self.lines:
                    if other_line != line:
                        if (line['point2'][0], line['point2'][1]) == (other_line['point1'][0], other_line['point1'][1]):
                            shared_points.append(f"{other_line['line_num']}_1")
                        elif (line['point2'][0], line['point2'][1]) == (other_line['point2'][0], other_line['point2'][1]):
                            shared_points.append(f"{other_line['line_num']}_2")
                
                if shared_points:
                    point2_str += f"         # {i}_2 (note: same as {shared_points[0]})"
                else:
                    point2_str += f"         # {i}_2"
                
                f.write(point2_str + "\n")
            
            # Dave
            if dave_points:
                f.write("# Dave\n")
                for line in dave_points:
                    f.write(f"{line['point1'][0]},{line['point1'][1]}         # top\n")
                    f.write(f"{line['point2'][0]},{line['point2'][1]}         # bottom\n")
            
            # Jack
            if jack_points:
                f.write("# Jack\n")
                for line in jack_points:
                    f.write(f"{line['point1'][0]},{line['point1'][1]}         # top\n")
                    f.write(f"{line['point2'][0]},{line['point2'][1]}         # bottom\n")
        
        self.status_text.set_text(f"Annotations saved to {self.annotations_filename}")
        self.fig.canvas.draw_idle()
        print(f"Annotations saved to {self.annotations_filename}")
    
    def run(self):
        """Run the annotation tool"""
        plt.show()
        
        return {
            'points': self.points,
            'point_types': self.point_types,
            'lines': self.lines
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Simple Metrology Annotation Tool')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('output_file', nargs='?', default='annotations/annotations_new.txt', 
                      help='Output file for annotations (default: annotations/annotations_new.txt)')
    args = parser.parse_args()
    
    annotator = SimpleMetrologyAnnotator(args.image_path, args.output_file)
    annotations = annotator.run()
    
    print("Annotation complete!")
    return annotations

if __name__ == "__main__":
    annotations = main()