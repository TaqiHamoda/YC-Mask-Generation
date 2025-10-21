import numpy as np, cv2
from scipy.spatial import ConvexHull, KDTree, qhull
from shapely.geometry import Polygon, Point
from skimage.draw import polygon as draw_polygon
from segment_anything import sam_model_registry, SamPredictor
from scipy.spatial import KDTree
from typing import List, Tuple, Optional

from .colormap import Colormap


class SAMInference:
    """
    A class to perform image segmentation using the Segment Anything Model (SAM).

    This class handles the entire workflow:
    1. Loading an image and corresponding point annotations from a CSV file.
    2. Grouping points by their labels.
    3. For each group, generating a convex hull to define the region of interest.
    4. Strategically generating prompt points within this region to guide SAM.
    5. Running SAM prediction to get a segmentation mask.
    6. Post-processing the mask (filtering by size, applying colors).
    7. Saving the final colored mask and an overlayed image.
    """

    def __init__(
        self,
        checkpoint_path: str,
        colormap: Colormap,
        model_type: str = 'vit_b',
        device: str = 'cuda',
        point_generation_candidates: int = 2000,
    ):
        """
        Initializes the SAMInference instance.

        Args:
            checkpoint_path (str): Path to the SAM model checkpoint file.
            model_type (str): The type of SAM model (e.g., 'vit_b', 'vit_l', 'vit_h').
            output_path (str): Directory to save the output masks and images.
            colormap (Colormap): Color map from class to BGR.
            device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        """
        self.colormap = colormap
        self.point_generation_candidates = point_generation_candidates

        print(f"Initializing SAM model ({model_type}) on device '{device}'...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def generate_sam_prompts(
        self,
        polygon: Polygon,
        class_points: Optional[np.ndarray] = None,
        mode: str = 'p'
    ) -> List[Tuple[float, float]]:
        """
        Generates optimal prompt points for SAM using a circle-packing approach.

        This method aims to find a set of N circle centers that maximally cover
        the area of the polygon ('l' mode) or the provided class_points ('p' mode).

        Args:
            polygon (Polygon): The geometric area (convex hull) to generate points in.
            class_points (np.ndarray, optional): The original points to be covered.
                                                Required for 'p' (point) mode.
            mode (str): 'p' for point-based coverage, 'l' for area-based coverage.

        Returns:
            A list of (x, y) coordinates for the selected prompt points.
        """
        area = polygon.area
        if area > 1_000_000:
            n_circles, radius = 8, 150
        elif area > 500_000:
            n_circles, radius = 5, 150
        elif area > 100_000:
            n_circles, radius = 3, 100
        else:
            n_circles, radius = 2, 90
        
        candidates = []
        min_x, min_y, max_x, max_y = polygon.bounds
        for _ in range(self.point_generation_candidates):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = Point(x, y)
            if polygon.contains(point):
                candidates.append(point)

        if len(candidates) == 0:
            return []

        if mode == 'p' and class_points is not None and len(class_points) > 0:
            # Score candidates based on how many class_points they cover
            kdtree = KDTree(class_points)
            scored_candidates = []
            for point in candidates:
                indices = kdtree.query_ball_point([point.x, point.y], radius)
                scored_candidates.append((len(indices), point))
        else:
            # Score candidates based on how much of their circle area is inside the polygon
            scored_candidates = []
            for point in candidates:
                circle = point.buffer(radius)
                intersection_area = polygon.intersection(circle).area
                scored_candidates.append((intersection_area, point))
        
        # Sort candidates by score (descending)
        scored_candidates.sort(reverse=True, key=lambda x: x[0])

        # Select N centers that are well-separated
        selected_centers = []
        min_dist_between_centers = 2 * radius

        for _, point in scored_candidates:
            if not selected_centers:
                selected_centers.append(point)
                continue
            
            # Check distance to already selected centers
            is_far_enough = True
            for center in selected_centers:
                if point.distance(center) < min_dist_between_centers:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                selected_centers.append(point)

            if len(selected_centers) >= n_circles:
                break
        
        return [center.coords[0] for center in selected_centers]

    def process_object_group(
        self,
        image: np.ndarray,
        points_group: np.ndarray,
        label: str,
        image_shape: Tuple[int, int]
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int]]]:
        """
        Processes a single group of points belonging to the same object class.

        This involves creating a convex hull, generating SAM prompts, running
        prediction, and returning the resulting mask and color.

        Returns:
            A tuple of (mask, color) or None if processing fails.
        """
        self.predictor.set_image(image)

        unique_points = np.unique(points_group, axis=0)
        if len(unique_points) < 3:
            return None # Not enough points to form a polygon

        try:
            hull = ConvexHull(unique_points)
            hull_vertices = unique_points[hull.vertices]
            polygon = Polygon(hull_vertices)

            # Create a binary mask from the convex hull for IOU calculation
            binary_hull_mask = np.zeros(image_shape, dtype=np.uint8)
            poly_coords = np.array(polygon.exterior.coords)
            rr, cc = draw_polygon(poly_coords[:, 1], poly_coords[:, 0], shape=image_shape)
            binary_hull_mask[rr, cc] = 1

            # Determine mode for point generation
            mode = 'p' if 'chain' in label or 'scale' in label else 'l'
            prompt_points = self.generate_sam_prompts(polygon, unique_points, mode=mode)
            if not prompt_points:
                return None

            # Run SAM prediction
            input_points = np.array(prompt_points)
            input_labels = np.ones(len(prompt_points), dtype=int)
            
            sam_mask, _, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )

            # Check Intersection over Union (IoU) between SAM mask and hull mask
            intersection = np.logical_and(sam_mask, binary_hull_mask)
            union = np.logical_or(sam_mask, binary_hull_mask)
            iou = np.sum(intersection) / np.sum(union)
            
            # If IoU is high, refine the SAM mask by constraining it to a dilated hull area
            if iou > 0.2:
                kernel = np.ones((5, 5), np.uint8)
                dilated_hull = cv2.dilate(binary_hull_mask, kernel, iterations=50)
                sam_mask = np.logical_and(sam_mask, dilated_hull)

            return sam_mask, self.colormap.getColor(label)
        except (qhull.QhullError, ValueError) as e:
            print(f"  - Could not process group '{label}': {e}")
            return None
