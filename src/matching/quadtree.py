"""
Quad-Tree Spatial Uniformity Enforcer.
Recursively divides the image swath to ensure Ground Control Points (GCPs)
are distributed evenly, preventing geometric distortion in featureless lunar maria.
"""

import numpy as np

class QuadTreeNode:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.bounds = (x_min, y_min, x_max, y_max)
        self.points = []
        self.children = []

    def insert(self, pt: np.ndarray, confidence: float):
        self.points.append((pt, confidence))

    def subdivide(self):
        x_min, y_min, x_max, y_max = self.bounds
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2

        self.children = [
            QuadTreeNode(x_min, y_min, mid_x, mid_y), # TL
            QuadTreeNode(mid_x, y_min, x_max, mid_y), # TR
            QuadTreeNode(x_min, mid_y, mid_x, y_max), # BL
            QuadTreeNode(mid_x, mid_y, x_max, y_max)  # BR
        ]

class UniformDistributor:
    @staticmethod
    def filter_points(points: np.ndarray, confidences: np.ndarray, 
                      img_width: int, img_height: int, 
                      max_points_per_cell: int = 5, max_depth: int = 4) -> np.ndarray:
        """
        Enforces spatial uniformity using a Quad-Tree decomposition.
        """
        root = QuadTreeNode(0, 0, img_width, img_height)
        
        # Load all points into root
        for pt, conf in zip(points, confidences):
            root.insert(pt, conf)

        def build_tree(node: QuadTreeNode, depth: int):
            if depth >= max_depth or len(node.points) <= max_points_per_cell:
                return
                
            node.subdivide()
            for pt_data in node.points:
                x, y = pt_data[0]
                # Route point to correct child quadrant
                for child in node.children:
                    bx_min, by_min, bx_max, by_max = child.bounds
                    if bx_min <= x <= bx_max and by_min <= y <= by_max:
                        child.insert(*pt_data)
                        break
            node.points = [] # Clear parent points
            for child in node.children:
                build_tree(child, depth + 1)

        build_tree(root, 0)

        # Collect the top N points from each leaf node
        uniform_points = []
        
        def collect_points(node: QuadTreeNode):
            if not node.children:
                if not node.points: return
                # Sort points in this cell by ML confidence score
                node.points.sort(key=lambda x: x[1], reverse=True)
                # Keep only the strongest points up to the limit
                for pt, _ in node.points[:max_points_per_cell]:
                    uniform_points.append(pt)
            else:
                for child in node.children:
                    collect_points(child)

        collect_points(root)
        return np.array(uniform_points)
