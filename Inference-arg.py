import numpy as np

from scipy.spatial import ConvexHull, KDTree, qhull
from shapely.geometry import Polygon, Point
from skimage.draw import polygon as draw_polygon
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
from typing import List, Tuple, Dict, Optional

import cv2, os, csv, glob, tqdm, argparse


# --- Constants ---
# It's good practice to define constants for magic numbers.
MIN_MASK_AREA_DEFAULT = 2000
POINT_GENERATION_CANDIDATES = 2000
DEFAULT_DEVICE = "cuda"

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
        model_type: str = "vit_b",
        output_path: str = "output/masks",
        color_map_path: str = "color_mapping.csv",
        device: str = DEFAULT_DEVICE,
    ):
        """
        Initializes the SAMInference instance.

        Args:
            checkpoint_path (str): Path to the SAM model checkpoint file.
            model_type (str): The type of SAM model (e.g., 'vit_b', 'vit_l', 'vit_h').
            output_path (str): Directory to save the output masks and images.
            color_map_path (str): Path to the CSV file for class-color mapping.
            device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        """
        self.output_path = output_path
        self.color_map_path = color_map_path
        self.min_mask_area = MIN_MASK_AREA_DEFAULT
        os.makedirs(self.output_path, exist_ok=True)

        self.predictor = self._initialize_sam_model(checkpoint_path, model_type, device)
        self.color_map = self._load_color_map()

    def _initialize_sam_model(self, checkpoint: str, model_type: str, device: str) -> SamPredictor:
        """Loads and prepares the SAM model and predictor."""
        print(f"Initializing SAM model ({model_type}) on device '{device}'...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        return SamPredictor(sam)

    def _load_color_map(self) -> Dict[str, Tuple[int, int, int]]:
        """Loads or creates the color mapping from a CSV file."""
        color_map = {}
        if not os.path.exists(self.color_map_path):
            return color_map
            
        with open(self.color_map_path, 'r') as map_file:
            reader = csv.reader(map_file)
            for row in reader:
                if not row: continue
                class_name, hex_color = row
                color = mcolors.hex2color(hex_color)
                rgb_color = tuple(int(c * 255) for c in color)
                color_map[class_name] = rgb_color
        return color_map

    def _get_or_create_color(self, label: str) -> Tuple[int, int, int]:
        """
        Retrieves a color for a given label from the color map.
        If the label is not found, a new random color is generated and saved.
        """
        clean_label = label.lstrip('0')
        if clean_label in self.color_map:
            return self.color_map[clean_label]

        # Generate a new random color if not found
        color_array = np.random.rand(1, 3)
        hex_color = mcolors.to_hex(color_array.ravel())
        rgb_color = tuple(int(c * 255) for c in color_array.ravel())
        
        self.color_map[clean_label] = rgb_color

        # Append the new color to the CSV file for future runs
        with open(self.color_map_path, 'a+', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([clean_label, hex_color])
            
        return rgb_color

    @staticmethod
    def _read_annotation_file(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads point coordinates and labels from a CSV file.

        Args:
            csv_path (str): Path to the annotation CSV file.

        Returns:
            A tuple containing:
            - np.ndarray: An array of (x, y) point coordinates.
            - np.ndarray: An array of corresponding string labels.
        """
        points, class_list = [], []
        with open(csv_path, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if not row or "png" in row[0]: # Skip header or irrelevant rows
                    continue
                label, x, y = row
                points.append([int(float(x)), int(float(y))])
                class_list.append(label)
        return np.array(points), np.array(class_list)

    @staticmethod
    def _generate_candidate_points(polygon: Polygon, num_points: int) -> List[Point]:
        """Generates random candidate points within the bounding box of a polygon."""
        candidates = []
        min_x, min_y, max_x, max_y = polygon.bounds
        for _ in range(num_points):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = Point(x, y)
            if polygon.contains(point):
                candidates.append(point)
        return candidates

    def _generate_sam_prompts(
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
        
        candidates = self._generate_candidate_points(polygon, POINT_GENERATION_CANDIDATES)
        if not candidates:
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

    def _process_object_group(
        self,
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
            prompt_points = self._generate_sam_prompts(polygon, unique_points, mode=mode)
            
            if not prompt_points:
                return None

            # Run SAM prediction
            input_points = np.array(prompt_points)
            input_labels = np.ones(len(prompt_points), dtype=int)
            
            sam_mask, scores, _ = self.predictor.predict(
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

            color = self._get_or_create_color(label)
            return sam_mask, color

        except (qhull.QhullError, ValueError) as e:
            print(f"  - Could not process group '{label}': {e}")
            return None

    def _filter_and_color_mask(
        self, 
        mask: np.ndarray, 
        color: Tuple[int, int, int], 
        image_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Filters a binary mask to remove small components and applies color.
        
        Args:
            mask (np.ndarray): The input binary mask.
            color (tuple): The RGB color to apply.
            image_shape (tuple): The shape of the final colored mask (H, W, C).
        
        Returns:
            np.ndarray: The colored and filtered mask.
        """
        mask_uint8 = mask.squeeze().astype(np.uint8)
        
        # Find connected components and filter by area
        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
        filtered_mask = np.zeros_like(mask_uint8)
        for i in range(1, num_labels): # Start from 1 to skip background
            if stats[i, cv2.CC_STAT_AREA] >= self.min_mask_area:
                filtered_mask[labels_map == i] = 1 # Use 1 for logical operations
        
        # Apply morphological closing to fill holes
        kernel = np.ones((15, 15), np.uint8)
        filled_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply color
        colored_mask_part = np.zeros(image_shape, dtype=np.uint8)
        for c in range(3):
            colored_mask_part[:, :, c] = filled_mask * color[c]
            
        return colored_mask_part

    def process_image(self, annotation_path: str):
        """
        Main processing function for a single image and its annotation file.

        Args:
            annotation_path (str): Path to the annotation CSV file.
        """
        image_path = annotation_path.replace('.csv', '.JPG').replace('annotations', 'images')
        print(f"\nProcessing: {os.path.basename(image_path)}")

        if not os.path.exists(image_path):
            print(f"  - Image not found at: {image_path}")
            return
            
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  - Failed to read image: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        image_points, image_labels = self._read_annotation_file(annotation_path)
        if image_points.size == 0:
            print("  - No points found in annotation file.")
            return

        all_masks, all_colors = [], []
        unique_labels = np.unique(image_labels)

        for label in unique_labels:
            class_points = image_points[image_labels == label]
            result = self._process_object_group(class_points, label, image_rgb.shape[:2])
            if result:
                mask, color = result
                all_masks.append(mask)
                all_colors.append(color)

        if not all_masks:
            print("  - No valid masks were generated for this image.")
            return

        # Combine all processed masks into a single colored image
        final_colored_mask = np.zeros_like(image_rgb)
        for mask, color in zip(all_masks, all_colors):
            colored_part = self._filter_and_color_mask(mask, color, image_rgb.shape)
            final_colored_mask = cv2.add(final_colored_mask, colored_part)

        # --- Save Outputs ---
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. Save the final colored mask (PNG with transparency)
        output_mask_path = os.path.join(self.output_path, f"{base_name}.png")
        # To save with transparency, we need an alpha channel
        mask_alpha = (final_colored_mask.max(axis=2) > 0).astype(np.uint8) * 255
        mask_bgra = cv2.cvtColor(final_colored_mask, cv2.COLOR_RGB2BGRA)
        mask_bgra[:, :, 3] = mask_alpha
        cv2.imwrite(output_mask_path, mask_bgra)

        # 2. Save the overlay image
        output_overlay_path = os.path.join(self.output_path, f"{base_name}_overlay.jpg")
        overlay = cv2.addWeighted(image_rgb, 1, final_colored_mask, 0.6, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_overlay_path, overlay_bgr)
        
        print(f"  - Successfully saved outputs to {self.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run SAM inference on a folder of images with point annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_folder', 
        type=str, 
        required=True,
        help='Path to the root folder containing "images" and "annotations" subdirectories.'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default="../sam_vit_b_01ec64.pth",
        help='Path to the SAM model checkpoint file.'
    )
    parser.add_argument(
        '--model_type', 
        type=str, 
        default="vit_b",
        help='SAM model type (e.g., "vit_b", "vit_l", "vit_h").'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default=None,
        help='Path to the output folder. Defaults to "masks" inside the input folder.'
    )

    args = parser.parse_args()

    # Define paths
    output_path = args.output_folder or os.path.join(args.input_folder, 'masks')
    annotations_folder = os.path.join(args.input_folder, 'annotations')
    color_map_path = os.path.join(output_path, 'color_mapping.csv')

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint not found at '{args.checkpoint}'")
        exit()

    # Initialize the inference engine
    infer = SAMInference(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        output_path=output_path,
        color_map_path=color_map_path
    )

    # Find all annotation files and process them
    annotation_files = glob.glob(os.path.join(annotations_folder, '*.csv'))
    print(f'Found {len(annotation_files)} annotation files to process.')
    
    if not annotation_files:
        print(f"Warning: No .csv files found in '{annotations_folder}'")
        exit()

    for ann_file in tqdm(annotation_files, desc="Processing images"):
        infer.process_image(ann_file)