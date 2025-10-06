import numpy as np
import cv2, os, tqdm, argparse
from typing import Dict, List, Tuple, Optional

from .annotations import Annotations
from .SAMInference import SAMInference

DEFAULT_DEVICE = "cuda"
MIN_MASK_AREA_DEFAULT = 2000
POINT_GENERATION_CANDIDATES = 2000


def filter_and_color_mask(
        mask: np.ndarray,
        color: Tuple[int, int, int],
        image_shape: Tuple[int, int, int],
        min_mask_area: int
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
            if stats[i, cv2.CC_STAT_AREA] >= min_mask_area:
                filtered_mask[labels_map == i] = 1 # Use 1 for logical operations

        # Apply morphological closing to fill holes
        kernel = np.ones((15, 15), np.uint8)
        filled_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

        # Apply color
        colored_mask_part = np.zeros(image_shape, dtype=np.uint8)
        for c in range(3):
            colored_mask_part[:, :, c] = filled_mask * color[c]
            
        return colored_mask_part


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run SAM inference on a folder of images with point annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--annotations_path', 
        type=str, 
        required=True,
        help='Path to the root folder containing "images" and "annotations" subdirectories.'
    )
    parser.add_argument(
        '--images_path', 
        type=str, 
        required=True,
        help='Path to the root folder containing "images" and "annotations" subdirectories.'
    )
    parser.add_argument(
        '--checkpoint_path', 
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
        '--output_path',
        type=str,
        default=None,
        help='Path to the output folder. Defaults to "masks" inside the input folder.'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint not found at '{args.checkpoint}'")
        exit()

    # Initialize the inference engine
    sam_model = SAMInference(
        checkpoint_path=args.checkpoint,
        color_map=color_map,
        model_type=args.model_type,
        device=DEFAULT_DEVICE,
        point_generation_candidates=POINT_GENERATION_CANDIDATES
    )

    annotations = Annotations(annotation_path)
    for image_name in tqdm(tuple(annotations.data.values())):
        if not os.path.exists(f"{images_dir}/{image_name}") or not os.path.isfile(f"{images_dir}/{image_name}"):
            print(f"Image {image_name} not found.")
            continue

        image = cv2.imread(f"{images_dir}/{image_name}", cv2.IMREAD_COLOR_RGB)

        image_points, image_labels = annotations.data[image_name]
        if image_points.size == 0:
            print("  - No points found in annotation file.")
            exit()

        all_masks, all_colors = [], []
        unique_labels = np.unique(image_labels)

        for label in unique_labels:
            class_points = image_points[image_labels == label]
            result = sam_model.process_object_group(image, class_points, label, image.shape[:2])
            if result:
                mask, color = result
                all_masks.append(mask)
                all_colors.append(color)

        if not all_masks:
            print("  - No valid masks were generated for this image.")
            exit()

        # Combine all processed masks into a single colored image
        final_colored_mask = np.zeros_like(image)
        for mask, color in zip(all_masks, all_colors):
            colored_part = filter_and_color_mask(mask, color, image.shape, MIN_MASK_AREA_DEFAULT)
            final_colored_mask = cv2.add(final_colored_mask, colored_part)

        # --- Save Outputs ---
        
        # 1. Save the final colored mask (PNG with transparency)
        output_mask_path = os.path.join(output_path, f"{image_name}_color.png")
        # To save with transparency, we need an alpha channel
        mask_alpha = (final_colored_mask.max(axis=2) > 0).astype(np.uint8) * 255
        mask_bgra = cv2.cvtColor(final_colored_mask, cv2.COLOR_RGB2BGRA)
        mask_bgra[:, :, 3] = mask_alpha
        cv2.imwrite(output_mask_path, mask_bgra)

        # 2. Save the overlay image
        output_overlay_path = os.path.join(output_path, f"{image_name}_overlay.jpg")
        overlay = cv2.addWeighted(image, 1, final_colored_mask, 0.6, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_overlay_path, overlay_bgr)
        
        print(f"  - Successfully saved outputs to {output_path}")