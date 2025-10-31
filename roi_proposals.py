from selective_search import SelectiveSearch
import numpy as np
from typing import List, Tuple, Dict, Any
import cv2
import os
import glob
import json
from multiprocessing import Pool, cpu_count
from functools import partial

class RoiProposals:
    def __init__(self):
        pass

    @staticmethod
    def process_folder(input_folder: str, output_folder: str, num_workers: int = None) -> None:
        """
        Process all images in a folder and save proposals as JSON.
        
        Args:
            input_folder (str): Path to folder containing images
            output_folder (str): Path to save output JSON file
            num_workers (int): Number of parallel processes (default: CPU count)
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        # Determine number of workers
        if num_workers is None:
            num_workers = cpu_count()

        image_paths = glob.glob(os.path.join(input_folder, '*.*'))
        
        if not image_paths:
            print(f"No images found in {input_folder}")
            return

        print(f"Processing {len(image_paths)} images with {num_workers} workers...")
        
        # Use multiprocessing to process images in parallel
        with Pool(num_workers) as pool:
            process_func = partial(RoiProposals._process_single_image)
            results = pool.map(process_func, image_paths)
        
        # Combine all results into single JSON
        all_proposals = {os.path.basename(path): proposals for path, proposals in results}
        
        # Save to JSON (compact format - one bbox per line)
        output_file = os.path.join(output_folder, 'proposals.json')
        with open(output_file, 'w') as f:
            f.write('{\n')
            for idx, (filename, proposals) in enumerate(all_proposals.items()):
                # Format bboxes with each on its own line
                f.write(f'  "{filename}": [\n')
                for bbox_idx, bbox in enumerate(proposals):
                    bbox_str = json.dumps(bbox)
                    trailing_comma = ',' if bbox_idx < len(proposals) - 1 else ''
                    f.write(f'    {bbox_str}{trailing_comma}\n')
                f.write(f'  ]')
                
                # Add trailing comma for all but last entry
                trailing_comma = ',' if idx < len(all_proposals) - 1 else ''
                f.write(f'{trailing_comma}\n')
            f.write('}\n')
        
        print(f"✓ Saved proposals to {output_file}")
        return all_proposals

    @staticmethod
    def _process_single_image(image_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a single image and return proposals.
        Static method for multiprocessing compatibility.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Tuple of (image_path, proposals_list)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read {image_path}")
                return image_path, []
            
            roi = RoiProposals()
            proposals = roi.get_proposals(image)
            
            # Convert to list of dicts for JSON serialization
            proposals_list = [
                {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
                for x, y, w, h in proposals
            ]
            
            return image_path, proposals_list
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return image_path, []

    def process_image(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Process image through preprocessing, selective search, and postprocessing.
        
        Args:
            image (np.ndarray): RGB image
            
        Returns:
            List of (x, y, w, h) proposals
        """
        resized_image, scaling_factor = self.preprocess_image(image)
        ss = SelectiveSearch(resized_image)
        # hierarchical_grouping returns (bbox, indices) tuples
        # Extract only bbox, discard indices (useful for future segmentation tasks)
        candidates = ss.hierarchical_grouping()
        bboxes = [bbox for bbox, _ in candidates]
        
        # Convert from (y_min, y_max, x_min, x_max) to (x, y, w, h)
        rects = self._convert_bbox_format(bboxes)
        return self.resize_rects(self.filter_proposals(rects), scaling_factor)

    def _convert_bbox_format(self, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Convert from (y_min, y_max, x_min, x_max) to (x, y, w, h) format.
        
        Args:
            bboxes (List): List of bounding boxes in format (y_min, y_max, x_min, x_max)
            
        Returns:
            List of bounding boxes in format (x, y, w, h)
        """
        converted = []
        for y_min, y_max, x_min, x_max in bboxes:
            w = x_max - x_min
            h = y_max - y_min
            converted.append((x_min, y_min, w, h))
        return converted

    
    def get_proposals(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Extract region proposals from a single image.
        
        Args:
            image (np.ndarray): Input image (BGR from cv2)
            
        Returns:
            List of (x, y, w, h) tuples
        """
        # Convert BGR to RGB for SelectiveSearch
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.process_image(image_rgb)
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image for faster processing.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple of (resized_image, scaling_factor)
        """
        # Resize image to smaller dimensions for faster processing
        height, width = image.shape[:2]
        max_dim = 600

        if max(height, width) > max_dim:
            scaling_factor = max_dim / float(max(height, width))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            scaling_factor = 1.0
        return image, scaling_factor
    
    def resize_rects(self, rects: List[Tuple[int, int, int, int]], scaling_factor: float) -> List[Tuple[int, int, int, int]]:
        """
        Scale rectangle coordinates back to original image size.
        
        Args:
            rects (List): List of (x, y, w, h) rectangles
            scaling_factor (float): Scale factor from preprocessing
            
        Returns:
            List of scaled rectangles
        """
        if scaling_factor == 1.0:
            return rects
        resized_rects = []
        for (x, y, w, h) in rects:
            new_x = int(x / scaling_factor)
            new_y = int(y / scaling_factor)
            new_w = int(w / scaling_factor)
            new_h = int(h / scaling_factor)
            resized_rects.append((new_x, new_y, new_w, new_h))
        return resized_rects

    
    def filter_proposals(self, rects: List[Tuple[int, int, int, int]], min_size: int = 20) -> List[Tuple[int, int, int, int]]:
        """
        Filter out too-small proposals.
        
        Args:
            rects (List): List of (x, y, w, h) rectangles
            min_size (int): Minimum width and height in pixels
            
        Returns:
            Filtered list of rectangles
        """
        filtered_rects = []
        for (x, y, w, h) in rects:
            if w >= min_size and h >= min_size:
                filtered_rects.append((x, y, w, h))
        return filtered_rects