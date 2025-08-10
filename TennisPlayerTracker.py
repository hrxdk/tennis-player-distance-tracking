import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import random

from matplotlib.patches import Circle
import json



video_path = "/Users/hrxdk/Documents/Machine Learning/Tennis Player Movement Calculator/tennis_video_assignment.mp4"

class WeightedTennisCourtDetector:

    def __init__(self, threshold=0.75):
        self.threshold = threshold
        print(f"🎾 Weighted Tennis Court Detector initialized (Threshold: {self.threshold})")

    def is_court_visible(self, frame):
        """
        Returns (is_valid, details) with confidence scoring.
        """
        score_details = {}

        color_score = self._score_court_color_area(frame)
        shape_score = self._score_rectangular_structure(frame)
        line_score = self._score_horizontal_lines(frame)

        total_score = 0.25 * color_score + 0.40 * shape_score + 0.35 * line_score
        is_valid = total_score >= self.threshold

        score_details.update({
            'color_score': round(color_score, 2),
            'shape_score': round(shape_score, 2),
            'line_score': round(line_score, 2),
            'final_score': round(total_score, 2)
        })

        return is_valid, score_details

    def _score_court_color_area(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        green_lower = np.array([35, 50, 50])
        green_upper = np.array([90, 255, 255])
        blue_lower = np.array([90, 20, 30])
        blue_upper = np.array([140, 255, 255])

        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        court_mask = cv2.bitwise_or(blue_mask, green_mask)

        court_pixels = np.sum(court_mask > 0)
        total_pixels = frame.shape[0] * frame.shape[1]
        court_ratio = court_pixels / total_pixels


        return min(court_ratio / 0.30, 1.0) 

    def _score_rectangular_structure(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150) 
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = 0
        frame_area = frame.shape[0] * frame.shape[1]  
        min_area = frame_area * 0.0015  

        for cnt in contours:
            epsilon = 0.04 * cv2.arcLength(cnt, True) 
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            area = cv2.contourArea(cnt)
            if len(approx) == 4 and area > min_area: 
                rectangles += 1

        return min(rectangles / 4, 1.0)  

    def _score_horizontal_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=5)

        if lines is None:
            return 0.0

        horizontal_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 3 or angle > 177: 
                horizontal_lines += 1

        
        return min(horizontal_lines / 10, 1.0)




class TennisCourtCornerSelector:
    def __init__(self):
        self.corners = []
        self.corner_names = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
        self.colors = ['red', 'blue', 'green', 'orange']
        
    def select_corners_interactive(self, frame):
        """
        Interactive corner selection using matplotlib
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display the frame
        if len(frame.shape) == 3:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(frame, cmap='gray')
            
        ax.set_title(f"Click to select: {self.corner_names[len(self.corners)]}")
        
        def onclick(event):
            if event.inaxes != ax:
                return
                
            x, y = int(event.xdata), int(event.ydata)
            self.corners.append([x, y])
            
            color = self.colors[len(self.corners)-1]
            circle = Circle((x, y), 8, color=color, fill=True)
            ax.add_patch(circle)
            ax.text(x+15, y, self.corner_names[len(self.corners)-1], 
                   color=color, fontsize=10, fontweight='bold')
            
            fig.canvas.draw()
            

            if len(self.corners) < 4:
                ax.set_title(f"Click to select: {self.corner_names[len(self.corners)]}")
            else:
                ax.set_title("All corners selected! Close the window.")
                

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        plt.show()
        
        if len(self.corners) == 4:
            return np.array(self.corners, dtype=np.float32)
        else:
            print(f"Only {len(self.corners)} corners selected. Need 4 corners.")
            return None


class PredefinedCourtHomography:
    def __init__(self):
        self.court_width = 23.77
        self.court_length = 10.97
      
        self.reference_court = np.array([
            [0, 0],                               # Bottom-left
            [self.court_width, 0],                # Bottom-right  
            [self.court_width, self.court_length], # Top-right
            [0, self.court_length]                # Top-left
        ], dtype=np.float32)
        
        self.homography_matrix = None
        self.inverse_homography = None
        
    def set_court_corners(self, detected_corners):
        """
        Set the predefined court corners and calculate homography
        """
        if detected_corners is None or len(detected_corners) != 4:
            print("❌ Need exactly 4 corners")
            return False
            
        self.detected_corners = detected_corners
        
        
        self.homography_matrix, mask = cv2.findHomography(
            detected_corners, 
            self.reference_court,
            cv2.RANSAC,
            5.0
        )
        
    
        self.inverse_homography, _ = cv2.findHomography(
            self.reference_court,
            detected_corners,
            cv2.RANSAC,
            5.0
        )
        
        if self.homography_matrix is not None:
            print("✅ Homography matrix calculated successfully!")
            return True
        else:
            print("❌ Failed to calculate homography matrix")
            return False
    
    def pixel_to_meters(self, pixel_coords):
       
        if self.homography_matrix is None:
            print("❌ Homography matrix not calculated yet")
            return None
            
        
        if len(np.array(pixel_coords).shape) == 1:
            pixel_coords = [pixel_coords]
            
       
        points = np.array(pixel_coords, dtype=np.float32)
        
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
            
       
        real_world_points = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2), 
            self.homography_matrix
        )
        
        return real_world_points.reshape(-1, 2)
    
    def meters_to_pixel(self, meter_coords):
        
        if self.inverse_homography is None:
            print("❌ Inverse homography matrix not calculated yet")
            return None
            
       
        if len(np.array(meter_coords).shape) == 1:
            meter_coords = [meter_coords]
            
        
        points = np.array(meter_coords, dtype=np.float32)
        
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
            
       
        pixel_points = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2), 
            self.inverse_homography
        )
        
        return pixel_points.reshape(-1, 2)
    
    def calculate_distance_meters(self, point1_pixels, point2_pixels):
        
        point1_meters = self.pixel_to_meters(point1_pixels)[0]
        point2_meters = self.pixel_to_meters(point2_pixels)[0]
        
       
        distance = np.sqrt((point2_meters[0] - point1_meters[0])**2 + 
                          (point2_meters[1] - point1_meters[1])**2)
        
        return distance
    
    def get_perspective_corrected_frame(self, frame, output_width=800, output_height=600):
     
        if self.inverse_homography is None:
            return frame
            
       
        court_aspect_ratio = self.court_width / self.court_length
        
        if output_width / output_height > court_aspect_ratio:
            # Width is too large, adjust it
            output_width = int(output_height * court_aspect_ratio)
        else:
           
            output_height = int(output_width / court_aspect_ratio)
        
      
        dst_points = np.array([
            [0, output_height],           # Bottom-left
            [output_width, output_height], # Bottom-right
            [output_width, 0],            # Top-right  
            [0, 0]                        # Top-left
        ], dtype=np.float32)
        
        
        correction_matrix = cv2.getPerspectiveTransform(self.detected_corners, dst_points)
        
        
        corrected = cv2.warpPerspective(frame, correction_matrix, (output_width, output_height))
        
        return corrected
    
    def visualize_court_setup(self, frame):
       
        vis_frame = frame.copy()
        
        if hasattr(self, 'detected_corners'):
            # Draw corners
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 165, 0)]  # BGR format
            corner_names = ["BL", "BR", "TR", "TL"]
            
            for i, corner in enumerate(self.detected_corners):
                # Draw circle
                cv2.circle(vis_frame, (int(corner[0]), int(corner[1])), 8, colors[i], -1)
                # Draw text
                cv2.putText(vis_frame, corner_names[i], 
                           (int(corner[0])+15, int(corner[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
            
            # Draw court outline
            court_corners = self.detected_corners.astype(int)
            cv2.polylines(vis_frame, [court_corners], True, (255, 255, 0), 3)
        
        return vis_frame
    
    def save_corners(self, filename="court_corners.json"):
       
        if hasattr(self, 'detected_corners'):
            corners_data = {
                'corners': self.detected_corners.tolist(),
                'court_width': self.court_width,
                'court_length': self.court_length
            }
            
            with open(filename, 'w') as f:
                json.dump(corners_data, f, indent=2)
            print(f"✅ Corners saved to {filename}")
        else:
            print("❌ No corners to save")
    
    def load_corners(self, filename="court_corners.json"):
       
        try:
            with open(filename, 'r') as f:
                corners_data = json.load(f)
            
            corners = np.array(corners_data['corners'], dtype=np.float32)
            success = self.set_court_corners(corners)
            
            if success:
                print(f"✅ Corners loaded from {filename}")
            return success
            
        except FileNotFoundError:
            print(f"❌ File {filename} not found")
            return False
        except Exception as e:
            print(f"❌ Error loading corners: {e}")
            return False


def setup_court_homography(video_path, frame_number=100):
    
    print("🎾 Setting up Tennis Court Homography")
    print("=" * 50)
    
  
    cap = cv2.VideoCapture(video_path)
    
   
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_number}")
        return None, None
    
   
    frame = cv2.resize(frame, (960, 540))
    
    print(f"📷 Loaded frame {frame_number} for corner selection")
    print("\n📍 Instructions:")
    print("1. Click on the tennis court corners in this order:")
    print("   - Bottom-Left (baseline left)")
    print("   - Bottom-Right (baseline right)")  
    print("   - Top-Right (far baseline right)")
    print("   - Top-Left (far baseline left)")
    print("2. Close the matplotlib window when done")
    print("\n👆 Click on the frame to select corners...")
    
    
    selector = TennisCourtCornerSelector()
    homography = PredefinedCourtHomography()
    
   
    corners = selector.select_corners_interactive(frame)
    
    if corners is not None:
        
        success = homography.set_court_corners(corners)
        
        if success:
            print("\n✅ Setup completed successfully!")
            
        
            vis_frame = homography.visualize_court_setup(frame)
            
            plt.figure(figsize=(15, 5))
            
            # Original frame with corners
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            plt.title("Original Frame with Corners")
            plt.axis('off')
            
            # Perspective corrected view
            corrected = homography.get_perspective_corrected_frame(frame)
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            plt.title("Top-Down Court View")
            plt.axis('off')
            
            # Test distance calculation visualization
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            plt.title("Ready for Player Tracking!")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Save corners for future use
            homography.save_corners("tennis_court_corners.json")
            
            return frame, homography
        else:
            print("❌ Failed to calculate homography")
            return frame, None
    else:
        print("❌ Corner selection failed")
        return frame, None


def test_distance_calculation(homography, test_frame):

    if homography is None:
        print("❌ No valid homography matrix")
        return
        
    print("\n🧮 Testing Distance Calculations")
    print("=" * 40)
    

    print("📏 Testing with sample court points...")
    
   
    bottom_center = homography.meters_to_pixel([11.885, 0])[0]  # Court center bottom
    top_center = homography.meters_to_pixel([11.885, 10.97])[0]  # Court center top
    
    court_length_pixels = homography.calculate_distance_meters(bottom_center, top_center)
    print(f"Court length: {court_length_pixels:.2f}m (should be ~10.97m)")
    
   
    left_center = homography.meters_to_pixel([0, 5.485])[0]  # Court center left
    right_center = homography.meters_to_pixel([23.77, 5.485])[0]  # Court center right
    
    court_width_pixels = homography.calculate_distance_meters(left_center, right_center)
    print(f"Court width: {court_width_pixels:.2f}m (should be ~23.77m)")
    
    print("\n✅ Homography is ready for player tracking!")
    print("💡 You can now use homography.calculate_distance_meters(point1, point2) for any two pixel coordinates")


print("🎾 Starting Tennis Court Homography Setup...")
frame, homography = setup_court_homography(video_path, frame_number=150)

if homography is not None:
    test_distance_calculation(homography, frame)
    print("\n🎯 Next steps:")
    print("1. Use YOLO to detect players in frames")
    print("2. Track player movements across frames") 
    print("3. Calculate distances using homography.calculate_distance_meters()")
else:
    print("❌ Setup failed. Try a different frame or reselect corners.")



from ultralytics import YOLO
from collections import deque
import time

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Player:
    id: int
    name: str
    color: Tuple[int, int, int]  # BGR color for visualization
    positions: deque = field(default_factory=lambda: deque(maxlen=10))
    total_distance: float = 0.0
    last_position: Optional[Tuple[float, float]] = None
    track_id: Optional[int] = None  # For YOLO tracking
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=5))


class TennisPlayerTracker:
    def __init__(self, homography, court_detector):
        self.homography = homography
        self.court_detector = court_detector
        
        # Initialize YOLO model
        print("🤖 Loading YOLO model...")
        self.yolo_model = YOLO('yolov8n.pt')  # Using nano version for speed
        
        # Initialize players
        self.players = {
            1: Player(
                id=1,
                name="Wozniacki",
                color=(255, 0, 0)  # Blue in BGR
            ),
            2: Player(
                id=2, 
                name="Sharapova",
                color=(0, 0, 255)  # Red in BGR
            )
        }
        
        # Tracking parameters
        self.min_confidence = 0.3
        self.position_smoothing = True
        self.max_assignment_distance = 100  # pixels
        
        print("✅ Tennis Player Tracker initialized")
    
    def detect_players(self, frame):
     
        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Only keep person detections (class 0 in COCO dataset)
                    if int(box.cls) == 0 and float(box.conf) > self.min_confidence:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf)
                        
                        # Calculate head position (top-center of bounding box)
                        head_x = (x1 + x2) / 2
                        head_y = y1 + (y2 - y1) * 0.1  # 10% from top for head position
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'head_position': (head_x, head_y),
                            'confidence': confidence
                        })
        
        return detections
    
    def assign_detections_to_players(self, detections):
        
        if len(detections) == 0:
            return {}
        
       
        detections_sorted = sorted(detections, key=lambda d: d['head_position'][1])
        
        assignments = {}
      
        if len(detections_sorted) >= 2:
           
            detections_by_conf = sorted(detections, key=lambda d: d['confidence'], reverse=True)
            top_detections = detections_by_conf[:2]
            
            
            top_detections.sort(key=lambda d: d['head_position'][1])
            
          
            assignments[1] = top_detections[1]  # Bottom (closer to camera)
            assignments[2] = top_detections[0]  # Top (farther from camera)
            
        elif len(detections_sorted) == 1:
          
            detection = detections_sorted[0]
            head_pos = detection['head_position']
            
            
            min_distance = float('inf')
            best_player = 1
            
            for player_id, player in self.players.items():
                if player.last_position is not None:
                    last_pixel_pos = self.homography.meters_to_pixel([player.last_position])[0]
                    distance = np.sqrt((head_pos[0] - last_pixel_pos[0])**2 + 
                                     (head_pos[1] - last_pixel_pos[1])**2)
                    if distance < min_distance and distance < self.max_assignment_distance:
                        min_distance = distance
                        best_player = player_id
            
            assignments[best_player] = detection
        
        return assignments
    
    def smooth_position(self, player_id, new_position):
        
        player = self.players[player_id]
        
        if not self.position_smoothing or len(player.positions) == 0:
            return new_position
        
        # Simple moving average
        recent_positions = list(player.positions)
        recent_positions.append(new_position)
        
        # Calculate weighted average (more weight to recent positions)
        weights = np.exp(np.linspace(-1, 0, len(recent_positions)))
        weights /= weights.sum()
        
        smoothed_x = np.average([pos[0] for pos in recent_positions], weights=weights)
        smoothed_y = np.average([pos[1] for pos in recent_positions], weights=weights)
        
        return (smoothed_x, smoothed_y)
    
    def update_player_distances(self, assignments):
       
        for player_id, detection in assignments.items():
            player = self.players[player_id]
            head_pixel_pos = detection['head_position']
            
           
            head_meters = self.homography.pixel_to_meters([head_pixel_pos])[0]
            
           
            smoothed_meters = self.smooth_position(player_id, tuple(head_meters))
            
           
            player.positions.append(smoothed_meters)
            player.confidence_history.append(detection['confidence'])
            
            
            if player.last_position is not None:
                distance_moved = np.sqrt(
                    (smoothed_meters[0] - player.last_position[0])**2 + 
                    (smoothed_meters[1] - player.last_position[1])**2
                )
                
                # Player-specific thresholds
                if player_id == 1:  # Near player
                    min_move = 0.2
                    max_move = 2.0
                    scale_factor = 1.0
                else:  # Far player
                    min_move = 0.1  # Lower minimum threshold for far player
                    max_move = 1.5  # Lower maximum threshold for far player
                    scale_factor = 0.7  # Scale down far player movements
                
                
                if min_move < distance_moved < max_move:
                    player.total_distance += (distance_moved * scale_factor)
            
            player.last_position = smoothed_meters
    
    def draw_player_tracking(self, frame, assignments):
       
        vis_frame = frame.copy()
        
        for player_id, detection in assignments.items():
            player = self.players[player_id]
            bbox = detection['bbox']
            head_pos = detection['head_position']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         player.color, 2)
            
            # Draw head tracking point
            cv2.circle(vis_frame, (int(head_pos[0]), int(head_pos[1])), 
                      8, player.color, -1)
            
            
            label = f"{player.name} (P{player_id})"
            confidence_text = f"Conf: {confidence:.2f}"
            distance_text = f"Dist: {player.total_distance:.1f}m"
            
            # Calculate text position (above bounding box)
            text_x = bbox[0]
            text_y = bbox[1] - 60
            
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_frame, (text_x, text_y - 45), 
                         (text_x + max(text_size[0], 120), text_y + 5), 
                         player.color, -1)
            
            # Draw text
            cv2.putText(vis_frame, label, (text_x + 5, text_y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, confidence_text, (text_x + 5, text_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis_frame, distance_text, (text_x + 5, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw movement trail
            if len(player.positions) > 1:
                trail_points = []
                for pos_meters in list(player.positions)[-5:]:  # Last 5 positions
                    pixel_pos = self.homography.meters_to_pixel([pos_meters])[0]
                    trail_points.append(tuple(map(int, pixel_pos)))
                
                # Draw trail lines
                for i in range(1, len(trail_points)):
                    cv2.line(vis_frame, trail_points[i-1], trail_points[i], 
                            player.color, 2)
        
        return vis_frame
    
    def draw_scorecard(self, frame):
       
        vis_frame = frame.copy()
        
        # Scorecard background
        scorecard_width = 300
        scorecard_height = 120
        
        # Draw semi-transparent background
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + scorecard_width, 10 + scorecard_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        # Draw border
        cv2.rectangle(vis_frame, (10, 10), (10 + scorecard_width, 10 + scorecard_height), 
                     (255, 255, 255), 2)
        
        # Title
        cv2.putText(vis_frame, "DISTANCE TRACKER", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Player distances
        y_offset = 60
        for player_id, player in self.players.items():
            player_text = f"{player.name}: {player.total_distance:.1f}m"
            cv2.putText(vis_frame, player_text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, player.color, 2)
            y_offset += 25
        
        # Total distance
        total_distance = sum(player.total_distance for player in self.players.values())
        cv2.putText(vis_frame, f"Total: {total_distance:.1f}m", (20, y_offset + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_frame

def process_tennis_video_with_tracking(video_path, homography, court_detector, 
                                     duration_seconds=5, output_path="tracked_tennis.mp4"):
    """
    Process tennis video with player tracking and distance calculation
    """
    print(f"🎾 Processing tennis video with tracking...")

    # Initialize tracker
    tracker = TennisPlayerTracker(homography, court_detector)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = min(int(cap.get(cv2.CAP_PROP_FPS)), 30)  # Cap FPS to 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    process_width = 640
    process_height = int(process_width * orig_height / orig_width)
    
    print(f"📐 Original: {orig_width}x{orig_height}, Processing: {process_width}x{process_height}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (process_width, process_height))
    
    frame_count = 0
    valid_frames = 0
    processing_times = []
    
    print(f"🎬 Processing {total_frames} frames...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            frame = cv2.resize(frame, (process_width, process_height))
            
           
            is_valid, score_details = court_detector.is_court_visible(frame)

            score_text = (
                f"C:{score_details['color_score']} "
                f"R:{score_details['shape_score']} "
                f"L:{score_details['line_score']} "
                f"S:{score_details['final_score']}"
            )
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, process_height - 60), (300, process_height - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            cv2.putText(
                frame, f"Scores: {score_text}",
                (10, process_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            info_text = f"Frame {frame_count+1}/{total_frames} | Valid: {is_valid}"
            cv2.putText(
                frame, info_text,
                (10, process_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            if is_valid:
                detections = tracker.detect_players(frame)
                assignments = tracker.assign_detections_to_players(detections)
                tracker.update_player_distances(assignments)
                frame = tracker.draw_player_tracking(frame, assignments)
                valid_frames += 1
            
            frame = tracker.draw_scorecard(frame)
            out.write(frame)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            if frame_count % 30 == 0:
                avg_time = np.mean(processing_times[-30:]) if len(processing_times) >= 30 else np.mean(processing_times)
                remaining_time = (total_frames - frame_count) * avg_time
                print(f"⏳ Frame {frame_count}/{total_frames} | "
                      f"Avg: {avg_time:.3f}s/frame | "
                      f"ETA: {remaining_time:.1f}s | "
                      f"Valid frames: {valid_frames}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    
    finally:
        cap.release()
        out.release()
        
        print(f"\n📊 Processing Complete!")
        print(f"═" * 50)
        print(f"Total frames processed: {frame_count}")
        print(f"Valid court frames: {valid_frames}")
        print(f"Success rate: {valid_frames/frame_count*100:.1f}%")
        print(f"Average processing time: {np.mean(processing_times):.3f}s/frame")
        print(f"Output saved to: {output_path}")
        
        print(f"\n🏃‍♀️ Final Distance Summary:")
        for player_id, player in tracker.players.items():
            print(f"{player.name}: {player.total_distance:.2f} meters")
        
        total_distance = sum(player.total_distance for player in tracker.players.values())
        print(f"Combined total: {total_distance:.2f} meters")
        
        return tracker


if 'homography' in locals() and homography is not None:
    print("🎾 Starting player tracking...")
    
    
    court_detector = WeightedTennisCourtDetector(threshold=0.60)
    

    tracker = process_tennis_video_with_tracking(
        video_path=video_path,
        homography=homography,
        court_detector=court_detector,
        duration_seconds=30,
        output_path="tennis_tracking_output.mp4"
    )
    
    print("\n✅ Tracking complete! Check 'tennis_tracking_output.mp4' for results.")
    
else:
    print("❌ Homography not found! Please run the corner selection setup first.")


def show_tracking_samples(tracker, video_path, sample_frames=[100, 200, 300]):
   
    cap = cv2.VideoCapture(video_path)
    
    fig, axes = plt.subplots(1, len(sample_frames), figsize=(15, 5))
    if len(sample_frames) == 1:
        axes = [axes]
    
    for i, frame_num in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.resize(frame, (640, 480))
            
            # Process frame
            detections = tracker.detect_players(frame)
            assignments = tracker.assign_detections_to_players(detections)
            vis_frame = tracker.draw_player_tracking(frame, assignments)
            vis_frame = tracker.draw_scorecard(vis_frame)
            
            axes[i].imshow(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Frame {frame_num}")
            axes[i].axis('off')
    
    cap.release()
    plt.tight_layout()
    plt.show()
