import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Obstacle:
    """Represents a polygonal obstacle."""
    vertices: np.ndarray  # Shape (N, 2) where N is number of vertices
    
    def __post_init__(self):
        """Ensure vertices are in the correct format."""
        self.vertices = np.array(self.vertices, dtype=np.float32)
        if len(self.vertices.shape) != 2 or self.vertices.shape[1] != 2:
            raise ValueError("Vertices must be Nx2 array")
    
    @property
    def center(self) -> np.ndarray:
        """Calculate the centroid of the polygon."""
        return np.mean(self.vertices, axis=0)
    
    @property
    def bounding_radius(self) -> float:
        """Calculate the radius of the smallest circle containing the polygon."""
        center = self.center
        distances = np.linalg.norm(self.vertices - center, axis=1)
        return np.max(distances)
    
    @classmethod
    def create_circle(cls, center: Tuple[float, float], radius: float, num_vertices: int = 12):
        """Create a circular obstacle approximated by a polygon."""
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        vertices = np.array([
            [center[0] + radius * np.cos(angle), 
             center[1] + radius * np.sin(angle)]
            for angle in angles
        ])
        return cls(vertices)
    
    @classmethod
    def create_rectangle(cls, center: Tuple[float, float], width: float, height: float):
        """Create a rectangular obstacle."""
        half_w, half_h = width / 2, height / 2
        vertices = np.array([
            [center[0] - half_w, center[1] - half_h],
            [center[0] + half_w, center[1] - half_h],
            [center[0] + half_w, center[1] + half_h],
            [center[0] - half_w, center[1] + half_h]
        ])
        return cls(vertices)


class ObstacleManager:
    """Manages obstacles in the environment."""
    
    def __init__(self, lidar_range: float = 3.0):
        self.obstacles: List[Obstacle] = []
        self.lidar_range = lidar_range
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the manager."""
        self.obstacles.append(obstacle)
    
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles.clear()
    
    def generate_random_obstacles(self, num_obstacles: int, bounds: dict, 
                                   robot_pos: np.ndarray, goal_pos: np.ndarray,
                                   min_clearance: float = 0.5, np_random=None):
        """
        Generate random obstacles within bounds, avoiding robot and goal positions.
        
        Args:
            num_obstacles: Number of obstacles to generate
            bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
            robot_pos: Current robot position [x, y]
            goal_pos: Goal position [x, y]
            min_clearance: Minimum distance from robot/goal
            np_random: Random number generator
        """
        if np_random is None:
            np_random = np.random
        
        self.clear_obstacles()
        
        for _ in range(num_obstacles):
            # Try to place obstacle with clearance
            max_attempts = 50
            for _ in range(max_attempts):
                # Random center position
                center_x = np_random.uniform(bounds['x_min'], bounds['x_max'])
                center_y = np_random.uniform(bounds['y_min'], bounds['y_max'])
                center = np.array([center_x, center_y])
                
                # Check clearance from robot and goal
                dist_robot = np.linalg.norm(center - robot_pos[:2])
                dist_goal = np.linalg.norm(center - goal_pos[:2])
                
                if dist_robot > min_clearance and dist_goal > min_clearance:
                    # Randomly choose obstacle type
                    obs_type = np_random.choice(['circle', 'rectangle'])
                    
                    if obs_type == 'circle':
                        radius = np_random.uniform(0.15, 0.3)
                        obstacle = Obstacle.create_circle((center_x, center_y), radius)
                    else:
                        width = np_random.uniform(0.2, 0.5)
                        height = np_random.uniform(0.2, 0.5)
                        obstacle = Obstacle.create_rectangle((center_x, center_y), width, height)
                    
                    self.add_obstacle(obstacle)
                    break
    
    def check_collision(self, robot_pos: np.ndarray, robot_radius: float) -> bool:
        """
        Check if robot collides with any obstacle using circle-polygon collision.
        
        Args:
            robot_pos: Robot position [x, y]
            robot_radius: Robot radius
            
        Returns:
            True if collision detected
        """
        for obstacle in self.obstacles:
            if self._circle_polygon_collision(robot_pos[:2], robot_radius, obstacle.vertices):
                return True
        return False
    
    def _circle_polygon_collision(self, circle_center: np.ndarray, circle_radius: float, 
                                   polygon_vertices: np.ndarray) -> bool:
        """
        Check collision between a circle and a polygon.
        Uses separation axis theorem (SAT) approach.
        """
        # Quick check: if circle center is far from polygon center
        poly_center = np.mean(polygon_vertices, axis=0)
        poly_radius = np.max(np.linalg.norm(polygon_vertices - poly_center, axis=1))
        
        if np.linalg.norm(circle_center - poly_center) > (circle_radius + poly_radius):
            return False
        
        # Check if circle center is inside polygon
        if self._point_in_polygon(circle_center, polygon_vertices):
            return True
        
        # Check distance from circle center to each edge
        n_vertices = len(polygon_vertices)
        for i in range(n_vertices):
            p1 = polygon_vertices[i]
            p2 = polygon_vertices[(i + 1) % n_vertices]
            
            dist = self._point_to_segment_distance(circle_center, p1, p2)
            if dist < circle_radius:
                return True
        
        return False
    
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _point_to_segment_distance(self, point: np.ndarray, seg_start: np.ndarray, 
                                     seg_end: np.ndarray) -> float:
        """Calculate minimum distance from a point to a line segment."""
        # Vector from start to end
        seg_vec = seg_end - seg_start
        seg_length_sq = np.dot(seg_vec, seg_vec)
        
        if seg_length_sq == 0:
            # Segment is a point
            return np.linalg.norm(point - seg_start)
        
        # Project point onto line segment
        t = max(0, min(1, np.dot(point - seg_start, seg_vec) / seg_length_sq))
        projection = seg_start + t * seg_vec
        
        return np.linalg.norm(point - projection)
    
    def get_closest_obstacles(self, robot_pos: np.ndarray, n_obstacles: int) -> List[Tuple[float, float]]:
        """
        Get the N closest obstacles to the robot.
        
        Args:
            robot_pos: Robot position [x, y, theta]
            n_obstacles: Number of closest obstacles to return
            
        Returns:
            List of tuples (distance, angle) in robot frame
        """
        theta = robot_pos[2]
        obs_data = []
        
        for obstacle in self.obstacles:
            center = obstacle.center
            d_vec = center - robot_pos[:2]
            d_dist = np.linalg.norm(d_vec)
            
            if d_dist < self.lidar_range:
                # Angle relative to robot heading
                obs_angle = np.arctan2(d_vec[1], d_vec[0]) - theta
                obs_angle = (obs_angle + np.pi) % (2 * np.pi) - np.pi
                obs_data.append((d_dist, obs_angle))
            else:
                obs_data.append((self.lidar_range, 0.0))
        
        # Sort by distance and return closest N
        obs_data.sort(key=lambda k: k[0])
        
        # Pad if fewer obstacles than requested
        while len(obs_data) < n_obstacles:
            obs_data.append((self.lidar_range, 0.0))
        
        return obs_data[:n_obstacles]
    
    def get_all_vertices(self) -> List[np.ndarray]:
        """Get all obstacle vertices for rendering."""
        return [obs.vertices for obs in self.obstacles]