import numpy as np
import cv2, os, tqdm, gc, yaml
from typing import Tuple

from .annotations import Annotations
from .colormap import Colormap
from .SAMInference import SAMInference


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
    # 💡 Load Configuration from YAML
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file 'config.yaml': {e}")
        exit()

    # Get values from config (or use sensible defaults if not in config)
    default_device = config.get('default_device', 'cuda')
    model_type = config.get('model_type', 'vit_b')
    min_mask_area_default = config.get('min_mask_area_default', 2000)
    point_generation_candidates = config.get('point_generation_candidates', 2000)
    annotations_file = config.get('annotations_file', 'annotations.csv')
    colormap_file = config.get('colormap_file', 'colormap.csv')
    images_dir = config.get('images_dir', 'images/')
    checkpoint_file = config.get('checkpoint_file', 'checkpoint.pth')
    output_dir = config.get('output_dir', 'masks/')

    dirs = config.get('dirs', [])

    for d in dirs:
        # 💡 Create output directory if it doesn't exist
        os.makedirs(f"{d}/{output_dir}", exist_ok=True)

        # Initialize the inference engine
        sam_model = SAMInference(
            checkpoint_path=f"{d}/{checkpoint_file}",
            color_map=Colormap(f"{d}/{colormap_file}"),
            model_type=model_type,
            device=default_device,
            point_generation_candidates=point_generation_candidates
        )

        # 💡 Initialize Annotations using the config path
        annotations = Annotations(annotations_file)
        
        # 💡 Fix the iteration: use keys (image_name) instead of values
        for image_name in tqdm.tqdm(annotations.data.keys()):
            image_path = f"{d}/{images_dir}/{image_name}"
            if not os.path.exists(image_path) or not os.path.isfile(image_path):
                print(f"Image {image_name} not found.")
                continue

            image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_labels, image_points = annotations.data[image_name]
            if image_points.size == 0:
                print("  - No points found in annotation file.")
                continue

            mask = np.zeros_like(image)
            for label in np.unique(image_labels):
                class_points = image_points[image_labels == label]
                result = sam_model.process_object_group(image, class_points, label, image.shape[:2])
                if result is None:
                    continue

                m, c = result
                colored_part = filter_and_color_mask(m, c, image.shape, min_mask_area_default)
                mask = cv2.add(mask, colored_part)

                del m, c, result

            if np.all(mask == 0):
                print(f"No valid masks were generated for image {image_name}.")
                continue

            cv2.imwrite(
                f"{d}/{output_dir}/{image_name.split('.')[:-1]}_color.png",
                cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            )

            cv2.imwrite(
                f"{d}/{output_dir}/{image_name.split('.')[:-1]}_overlay.png",
                cv2.cvtColor(cv2.addWeighted(image, 1, mask, 0.6, 0), cv2.COLOR_RGB2BGR)
            )

        del annotations, sam_model, image
        gc.collect()