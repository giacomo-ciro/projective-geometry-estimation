import cv2
import numpy as np
import argparse
import os


class CorrespondenceAnnotator:
    def __init__(self, image1_path, image2_path, output_path):
        # Load images
        self.img1 = cv2.imread(image1_path)
        self.img2 = cv2.imread(image2_path)
        
        if self.img1 is None or self.img2 is None:
            raise ValueError("Could not load one or both images")
            
        # Store original images for display
        self.original_img1 = self.img1.copy()
        self.original_img2 = self.img2.copy()
        
        # Resize if images are too large
        self.resize_images()
        
        # Output file path
        self.output_path = output_path
        
        # Store point correspondences
        self.points1 = []
        self.points2 = []
        
        # Keep track of which image we're selecting points from
        self.selecting_img1 = True
        
        # Current temporary point (for display purposes)
        self.temp_point = None
        
        # Window name
        self.window_name = "Image Correspondence Annotator"
        
        # Colors for markers
        self.colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]
        
    def resize_images(self):
        """Resize images if they're too large for display"""
        max_height = 2000
        max_width = 2000
        
        h1, w1 = self.img1.shape[:2]
        h2, w2 = self.img2.shape[:2]
        
        # Resize image 1 if needed
        if h1 > max_height or w1 > max_width:
            scale = min(max_height / h1, max_width / w1)
            self.img1 = cv2.resize(self.img1, None, fx=scale, fy=scale)
            
        # Resize image 2 if needed
        if h2 > max_height or w2 > max_width:
            scale = min(max_height / h2, max_width / w2)
            self.img2 = cv2.resize(self.img2, None, fx=scale, fy=scale)
            
        # Store resize scales for saving original coordinates
        self.scale1 = self.original_img1.shape[1] / self.img1.shape[1]
        self.scale2 = self.original_img2.shape[1] / self.img2.shape[1]
            
        # Get dimensions after resize
        self.h1, self.w1 = self.img1.shape[:2]
        self.h2, self.w2 = self.img2.shape[:2]
        
        # Height of the combined display (max of two images)
        self.display_height = max(self.h1, self.h2)
        
        # Width of the combined display
        self.display_width = self.w1 + self.w2 + 20  # 20px separator
        
    def create_display_image(self):
        """Create the combined display image with annotations"""
        # Create blank canvas
        display = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Place first image on the left
        display[0:self.h1, 0:self.w1] = self.img1
        
        # Place second image on the right
        display[0:self.h2, self.w1+20:self.w1+20+self.w2] = self.img2
        
        # Draw gray separator
        cv2.line(display, (self.w1+10, 0), (self.w1+10, self.display_height), (100, 100, 100), 1)
        
        # Draw all point pairs (with matching colors)
        for i, (p1, p2) in enumerate(zip(self.points1, self.points2)):
            color = self.colors[i % len(self.colors)]
            # Draw point on first image (1 pixel for precision)
            cv2.circle(display, p1, 1, color, -1)
            cv2.putText(display, str(i+1), (p1[0]+5, p1[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw point on second image (1 pixel for precision)
            adjusted_p2 = (p2[0] + self.w1 + 20, p2[1])
            cv2.circle(display, adjusted_p2, 1, color, -1)
            cv2.putText(display, str(i+1), (adjusted_p2[0]+5, adjusted_p2[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw line connecting points
            cv2.line(display, p1, adjusted_p2, color, 1)
            
        # Draw temporary point if it exists (1 pixel for precision)
        if self.temp_point is not None:
            if self.selecting_img1:
                cv2.circle(display, self.temp_point, 1, (255, 255, 255), -1)
            else:
                adjusted_temp = (self.temp_point[0] + self.w1 + 20, self.temp_point[1])
                cv2.circle(display, adjusted_temp, 1, (255, 255, 255), -1)
        
        # Add instructions
        cv2.putText(display, "ESC: Quit, S: Save, D: Delete last, R: Reset", 
                   (10, self.display_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
                   
        if self.selecting_img1:
            cv2.putText(display, "Selecting point in LEFT image", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(display, "Selecting point in RIGHT image", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return display
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_MOUSEMOVE:
            # Update temporary point location based on which image the mouse is over
            if 0 <= x < self.w1:  # First image
                if self.selecting_img1:
                    self.temp_point = (x, y)
                else:
                    self.temp_point = None
            elif self.w1+20 <= x < self.display_width:  # Second image
                if not self.selecting_img1:
                    self.temp_point = (x - (self.w1+20), y)
                else:
                    self.temp_point = None
            else:
                self.temp_point = None
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Handle left click for point selection
            if self.selecting_img1 and 0 <= x < self.w1:
                # Selected point in first image
                self.points1.append((x, y))
                self.selecting_img1 = False
                print(f"Selected point in image 1: ({x}, {y})")
                
            elif not self.selecting_img1 and self.w1+20 <= x < self.display_width:
                # Selected point in second image
                adjusted_x = x - (self.w1+20)
                self.points2.append((adjusted_x, y))
                self.selecting_img1 = True
                print(f"Selected point in image 2: ({adjusted_x}, {y})")
                print(f"Pair {len(self.points1)} added successfully")
                
        self.update_display()
                
    def update_display(self):
        """Update the display window"""
        display = self.create_display_image()
        cv2.imshow(self.window_name, display)
        
    def save_correspondences(self):
        """Save point correspondences to a text file"""
        with open(self.output_path, 'w') as f:
            f.write("# Point correspondences between two images\n")
            f.write("# Format: x1 y1 x2 y2\n")
            f.write(f"# Total pairs: {len(self.points1)}\n")
            
            for i, (p1, p2) in enumerate(zip(self.points1, self.points2)):
                # Convert back to original image coordinates
                orig_x1 = int(p1[0] * self.scale1)
                orig_y1 = int(p1[1] * self.scale1)
                orig_x2 = int(p2[0] * self.scale2)
                orig_y2 = int(p2[1] * self.scale2)
                
                f.write(f"{orig_x1} {orig_y1} {orig_x2} {orig_y2}\n")
                
        print(f"Saved {len(self.points1)} point pairs to {self.output_path}")
        
    def delete_last_pair(self):
        """Delete the last point pair"""
        if self.points1 and not self.selecting_img1:
            # If we're selecting the second point, remove the first point
            self.points1.pop()
            self.selecting_img1 = True
            print("Removed last point from image 1")
        elif self.points1 and self.points2 and self.selecting_img1:
            # If we're selecting the first point, remove the last pair
            self.points1.pop()
            self.points2.pop()
            print("Removed last point pair")
        self.update_display()
        
    def reset(self):
        """Reset all point selections"""
        self.points1 = []
        self.points2 = []
        self.selecting_img1 = True
        print("Reset all points")
        self.update_display()
        
    def run(self):
        """Run the annotation tool"""
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Initial display
        self.update_display()
        
        print("\nImage Correspondence Annotator")
        print("------------------------------")
        print("Instructions:")
        print("  - Click to select corresponding points in the two images")
        print("  - Press 'S' to save correspondences to file")
        print("  - Press 'D' to delete the last point pair")
        print("  - Press 'R' to reset all points")
        print("  - Press 'ESC' to quit\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC key
                break
            elif key == ord('s'):  # Save
                if len(self.points1) == len(self.points2) and len(self.points1) > 0:
                    self.save_correspondences()
                else:
                    print("Nothing to save or incomplete pair")
            elif key == ord('d'):  # Delete last
                self.delete_last_pair()
            elif key == ord('r'):  # Reset
                self.reset()
                
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Annotate corresponding points in two images')
    parser.add_argument('image1', help='Path to first image')
    parser.add_argument('image2', help='Path to second image')
    parser.add_argument('--output', '-o', default='annotations/correspondences.txt', 
                        help='Output file path (default: annotations/correspondences.txt)')
    
    args = parser.parse_args()
    
    # Check if images exist
    if not os.path.exists(args.image1):
        print(f"Error: Image file not found: {args.image1}")
        return
    if not os.path.exists(args.image2):
        print(f"Error: Image file not found: {args.image2}")
        return
        
    # Create and run the annotator
    try:
        annotator = CorrespondenceAnnotator(args.image1, args.image2, args.output)
        annotator.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()