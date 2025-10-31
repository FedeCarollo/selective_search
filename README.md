# Selective Search for Object Recognition

A Python implementation of the **Selective Search** algorithm for generating region proposals in object detection. Based on the paper:

> **"Selective Search for Object Recognition"** by J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, and A.W.M. Smeulders  
> Reference: http://www.huppelen.nl/publications/selectiveSearchDraft.pdf

## Overview

Selective Search generates high-quality object detection proposals by hierarchically merging image regions based on multiple complementary similarity metrics. Unlike sliding-window approaches, it produces region proposals of varying sizes and scales, making it ideal for object detection pipelines.

### Key Features

- **Hierarchical merging**: Bottom-up region grouping for multi-scale proposals
- **Multi-modal similarity**: Combines color, texture, size, and spatial cues
- **Efficient candidate generation**: Hundreds to thousands of proposals per image
- **Deterministic**: Produces consistent results across runs
- **Multiprocessing support**: Parallel batch processing with `RoiProposals`
- **JSON output**: Easy integration with downstream tasks

### Algorithm Pipeline

```
Input Image
    ↓
[1] Initial Over-segmentation (Felzenszwalb)
    ↓ Creates fine-grained segments
[2] Region Feature Extraction
    ↓ Color histograms, texture, bounding boxes, neighbors
[3] Hierarchical Grouping
    ↓ Priority-queue based merging by similarity
    ↓ Collect proposals at each merge step
Output: Region Proposals (bbox + optional pixel indices)
```

## Installation

### Prerequisites

- **Python 3.7+** (3.9+ recommended)
- **pip** package manager

### Quick Install

```bash
pip install -r requirements.txt
```

This installs core dependencies:
- `numpy` - Numerical computing
- `scikit-image` - Image segmentation (Felzenszwalb)
- `opencv-python` - Image I/O (used in `RoiProposals`)

### Verification

```bash
python -c "from selective_search import SelectiveSearch; print('✓ Installation successful')"
```

## Usage Example

### Basic Selective Search

```python
import numpy as np
from selective_search import SelectiveSearch

# Load an image (RGB, values in [0, 255] or [0, 1])
img = np.array(...)  # Shape: (H, W, 3)

# Initialize Selective Search
ss = SelectiveSearch(img)

# Generate region proposals with pixel indices
candidates = ss.hierarchical_grouping()
print(f"Generated {len(candidates)} proposals")

# Each candidate is (bbox, pixel_indices)
for bbox, indices in candidates:
    y_min, y_max, x_min, x_max = bbox
    print(f"Proposal: ({x_min}, {y_min})-({x_max}, {y_max}), pixels={len(indices)}")
```

### Batch Processing with RoiProposals (JSON + Multiprocessing)

```python
from roi_proposals import RoiProposals

# Process all images in a folder with multiprocessing
roi = RoiProposals()
proposals = roi.process_folder(
    input_folder='img/',
    output_folder='bboxes/',
    num_workers=4  # Auto-detects CPU count if not specified
)

# Output: bboxes/proposals.json
# Format: {"image1.jpg": [{"x": 10, "y": 20, "w": 100, "h": 80}, ...], ...}
```

### Single Image Processing

```python
from roi_proposals import RoiProposals
import cv2

# Load image (BGR format from cv2)
image = cv2.imread('test.jpg')

# Extract proposals
roi = RoiProposals()
proposals = roi.get_proposals(image)

# proposals: List of (x, y, w, h) tuples
for x, y, w, h in proposals:
    print(f"Region: x={x}, y={y}, width={w}, height={h}")
```

### Visualizing Proposals

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(img)

# Draw bounding boxes for proposals with area > 2500 pixels
for bbox, indices in candidates:
    y_min, y_max, x_min, x_max = bbox
    h, w = x_max - x_min, y_max - y_min
    
    if h * w < 2500:  # Skip very small proposals
        continue
    
    rect = patches.Rectangle(
        (x_min, y_min), w, h,
        linewidth=1, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

ax.set_title(f"Selective Search Proposals ({len(candidates)} total)")
ax.axis('off')
plt.show()
```

## API Reference

### `SelectiveSearch` Class

#### Constructor

```python
SelectiveSearch(img)
```

**Parameters:**
- `img` (np.ndarray): RGB image with shape (H, W, 3), values in [0, 255] or [0, 1]

**Attributes:**
- `img`: Original image
- `segments`: Initial segmentation map (H, W)
- `regions`: Dictionary of region properties {label: {properties}}
- `candidates`: Generated proposals (populated after calling `hierarchical_grouping()`)

#### Methods

##### `hierarchical_grouping()`

Executes the core hierarchical merging algorithm.

```python
candidates = ss.hierarchical_grouping()
```

**Returns:** `list` of tuples `(bbox, indices)`
- `bbox`: (y_min, y_max, x_min, x_max) - pixel coordinates
- `indices`: np.ndarray of pixel indices in flattened image (useful for segmentation tasks)

**Time Complexity:** O(n log n) where n = number of region pairs
**Space Complexity:** O(n)

#### Similarity Metrics

##### `color_similarity(r1, r2)`

Computes color histogram intersection between two regions.

```python
sim = ss.color_similarity(region_id_1, region_id_2)  # Returns [0, 1]
```

**Range:** [0, 1] (0 = different colors, 1 = identical colors)

##### `texture_similarity(r1, r2)`

Computes texture (gradient distribution) similarity.

```python
sim = ss.texture_similarity(region_id_1, region_id_2)  # Returns [0, 1]
```

**Range:** [0, 1]

##### `size_similarity(r1, r2)`

Computes size contrast regularization.

```python
sim = ss.size_similarity(region_id_1, region_id_2)  # Returns [0, 1]
```

**Range:** [0, 1] (1 = similar sizes, 0 = very different sizes)

##### `fill_similarity(r1, r2)`

Computes spatial compactness (fill) metric.

```python
sim = ss.fill_similarity(region_id_1, region_id_2)  # Returns [0, 1]
```

**Range:** [0, 1] (1 = compact/adjacent, 0 = far apart)

##### `similarity(r1, r2)`

Computes combined similarity score (used for merging decisions).

```python
combined_sim = ss.similarity(region_id_1, region_id_2)  # Returns [0, 4]
```

**Formula:** S = color_sim + fill_sim + texture_sim + size_sim

**Range:** [0, 4]

### `RoiProposals` Class

#### Methods

##### `process_folder(input_folder, output_folder, num_workers=None)`

Process all images in a folder with multiprocessing and save to JSON.

```python
roi = RoiProposals()
roi.process_folder('input/', 'output/', num_workers=4)
```

**Parameters:**
- `input_folder` (str): Path to folder containing images
- `output_folder` (str): Path to save output JSON file
- `num_workers` (int): Number of parallel processes (default: auto-detect CPU count)

**Output:**
- `proposals.json`: Single JSON file with all results
  ```json
  {
    "image1.jpg": [
      {"x": 10, "y": 20, "w": 100, "h": 80},
      {"x": 50, "y": 30, "w": 120, "h": 90}
    ],
    "image2.jpg": [...]
  }
  ```

##### `get_proposals(image)`

Extract region proposals from a single image (BGR from cv2).

```python
image = cv2.imread('test.jpg')
proposals = roi.get_proposals(image)  # List of (x, y, w, h)
```

**Parameters:**
- `image` (np.ndarray): BGR image from cv2.imread()

**Returns:**
- `list`: List of (x, y, w, h) tuples

##### `process_image(image)`

Process RGB image through preprocessing, selective search, and postprocessing.

```python
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
proposals = roi.process_image(image_rgb)
```

**Parameters:**
- `image` (np.ndarray): RGB image

**Returns:**
- `list`: List of (x, y, w, h) tuples (scaled to original image size)

## Data Structures

### Region Dictionary (SelectiveSearch)

Each region in `ss.regions` has the following structure:

```python
{
    'label': int,                          # Unique region ID
    'indices': np.ndarray,                 # Pixel indices (shape: (n_pixels,))
    'size': int,                           # Number of pixels
    'color_hist': np.ndarray,              # Shape (3, 25) - RGB histograms
    'texture': np.ndarray,                 # Shape (10,) - gradient histogram
    'bounding_box': tuple,                 # (y_min, y_max, x_min, x_max)
    'neighbors': set,                      # Set of neighbor region IDs
}
```

### Bounding Box Formats

**SelectiveSearch format:**
```
(y_min, y_max, x_min, x_max)
```

**RoiProposals JSON format:**
```json
{"x": int, "y": int, "w": int, "h": int}
```

**Conversion:**
```python
# From SelectiveSearch to RoiProposals
y_min, y_max, x_min, x_max = bbox
x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
```

## Performance

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Initialization | O(H×W) | Segmentation + region extraction |
| `hierarchical_grouping()` | O(n log n) | n = number of neighbors |
| `process_folder()` | O(N*n log n / workers) | N = number of images, parallel processing |
| Feature extraction | O(H×W) | One-time cost per image |
| Similarity computation | O(1) | Constant-time per pair |

### Space Complexity

- Region storage: O(n) where n = number of segments
- Heap: O(neighbor pairs) ≈ O(n)
- Total: O(H×W) for pixel storage + O(n) for region metadata

### Typical Performance (CPU)

| Image Size | Proposals | Time | Memory |
|-----------|-----------|------|--------|
| 480×640 | ~1000 | 1-2s | ~100MB |
| 480×640 (scaled 0.15x) | ~1000 | 0.2s | ~10MB |
| 1024×768 | ~2000 | 5-10s | ~300MB |

**Multiprocessing speedup (batch):**
- 1 worker (sequential): N images × ~2s = 2N seconds
- 4 workers (parallel): N images × ~2s / 4 ≈ 0.5N seconds
- 8 workers (parallel): N images × ~2s / 8 ≈ 0.25N seconds

### Optimization Strategies

For large images or real-time requirements:

1. **Downscale image before processing:**
   ```python
   import cv2
   scale = 0.5
   img_small = cv2.resize(img, None, fx=scale, fy=scale)
   ```

2. **Filter proposals by size:**
   ```python
   large_proposals = [p for p in proposals if p[2] * p[3] > 500]  # w*h threshold
   ```

3. **Increase initial segmentation scale (fewer initial regions):**
   ```python
   # In selective_search.py get_initial_segments():
   segmentation.felzenszwalb(img, scale=150, ...)  # Default: 100
   ```

4. **Use multiprocessing for batch processing:**
   ```python
   roi.process_folder(input_dir, output_dir, num_workers=8)
   ```

## Parameter Tuning

### Initial Segmentation Parameters

In `SelectiveSearch.get_initial_segments()`:

```python
segmentation.felzenszwalb(self.img, scale=100, sigma=0.5, min_size=50)
```

| Parameter | Effect | Suggested Range |
|-----------|--------|-----------------|
| `scale` | Region size (higher → larger) | 50-200 |
| `sigma` | Blur bandwidth | 0.3-1.0 |
| `min_size` | Minimum region pixels | 30-100 |

**Effects:**
- **Larger `scale`**: Fewer initial regions, coarser segmentation, faster processing
- **Larger `min_size`**: Removes tiny segments, speeds up processing

### RoiProposals Filter Parameters

In `RoiProposals.filter_proposals()`:

```python
def filter_proposals(self, rects, min_size: int = 20):
    # Filters out proposals smaller than min_size × min_size
```

Adjust `min_size` to filter unwanted small proposals.

## Common Issues

### Issue: Too Few Proposals

**Cause:** Initial segmentation too coarse or similarity threshold too high

**Solution:**
```python
# Decrease scale for finer initial segmentation
segmentation.felzenszwalb(img, scale=50, ...)  # Default: 100
```

### Issue: Too Many Proposals

**Cause:** Initial segmentation too fine or all merges are accepted

**Solution:**
```python
# Increase scale for coarser segmentation
segmentation.felzenszwalb(img, scale=150, ...)

# Or filter by size
candidates = [(b, idx) for b, idx in candidates if len(idx) > 100]
```

### Issue: Slow Performance on Large Batches

**Cause:** Sequential processing

**Solution:**
```python
# Use multiprocessing
roi.process_folder(input_dir, output_dir, num_workers=8)
```

### Issue: "ModuleNotFoundError: No module named 'cv2'"

```bash
pip install opencv-python
```

## Comparison with Alternatives

| Method | Advantages | Disadvantages |
|--------|-----------|----------------|
| **Selective Search** | Deterministic, multi-scale, good recall | CPU-based, slower than deep learning |
| **Sliding Windows** | Simple, exhaustive | Computationally expensive, poor aspect ratios |
| **R-CNN** | End-to-end learning, faster | Requires labeled data, black-box |
| **Edge Boxes** | Fast, good precision | Requires edge detection, less flexible |

## References

1. **Original Paper**: [Selective Search for Object Recognition (Uijlings et al., 2013)](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)

2. **Felzenszwalb Segmentation**: [Efficient Graph-Based Image Segmentation (Felzenszwalb & Huttenlocher, 2004)](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)

3. **Related Work**:
   - R-CNN (Girshick et al., 2014) - Uses Selective Search for proposals
   - Faster R-CNN (Ren et al., 2015) - Learns proposals end-to-end

## License

This implementation is provided for educational and research purposes.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{uijlings2013selective,
  title={Selective search for object recognition},
  author={Uijlings, JRR and Van De Sande, KE and Gevers, T and Smeulders, AWM},
  journal={International journal of computer vision},
  volume={104},
  number={2},
  pages={154--171},
  year={2013},
  publisher={Springer}
}
```

## FAQ

### How many proposals do I typically get?

On a 480×640 image, expect 500-2000 proposals depending on content complexity.

### Can I control the number of proposals?

Yes, via:
- `scale` parameter (higher → fewer)
- Size filtering: `filter_proposals(min_size=X)`
- Similarity threshold (in `hierarchical_grouping()`)

### What's the difference between SelectiveSearch and RoiProposals?

- **SelectiveSearch**: Core algorithm, returns bboxes + pixel indices
- **RoiProposals**: Wrapper with preprocessing, multiprocessing, and JSON output for batch processing

### How do I use pixel indices for segmentation?

```python
ss = SelectiveSearch(img)
candidates = ss.hierarchical_grouping()

for bbox, indices in candidates:
    # Create binary mask
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask.flat[indices] = True
    
    # Extract region pixels
    region = img[mask]
```

## API Reference

### `SelectiveSearch` Class

#### Constructor

```python
SelectiveSearch(img)
```

**Parameters:**
- `img` (np.ndarray): RGB image with shape (H, W, 3), values in [0, 255] or [0, 1]

**Attributes:**
- `img`: Original image
- `segments`: Initial segmentation map (H, W)
- `regions`: Dictionary of region properties {label: {properties}}
- `candidates`: Generated proposals (populated after calling `hierarchical_grouping()`)

#### Methods

##### `hierarchical_grouping()`

Executes the core hierarchical merging algorithm.

```python
candidates = ss.hierarchical_grouping()
```

**Returns:** `list` of tuples `(bbox, indices)`
- `bbox`: (y_min, y_max, x_min, x_max) - pixel coordinates
- `indices`: np.ndarray of pixel indices in flattened image

**Time Complexity:** O(n log n) where n = number of region pairs
**Space Complexity:** O(n)

#### Similarity Metrics

##### `color_similarity(r1, r2)`

Computes color histogram intersection between two regions.

```python
sim = ss.color_similarity(region_id_1, region_id_2)  # Returns [0, 1]
```

**Range:** [0, 1] (0 = different colors, 1 = identical colors)

##### `texture_similarity(r1, r2)`

Computes texture (gradient distribution) similarity.

```python
sim = ss.texture_similarity(region_id_1, region_id_2)  # Returns [0, 1]
```

**Range:** [0, 1]

##### `size_similarity(r1, r2)`

Computes size contrast regularization.

```python
sim = ss.size_similarity(region_id_1, region_id_2)  # Returns [0, 1]
```

**Range:** [0, 1] (1 = similar sizes, 0 = very different sizes)

##### `fill_similarity(r1, r2)`

Computes spatial compactness (fill) metric.

```python
sim = ss.fill_similarity(region_id_1, region_id_2)  # Returns [0, 1]
```

**Range:** [0, 1] (1 = compact/adjacent, 0 = far apart)

##### `similarity(r1, r2)`

Computes combined similarity score (used for merging decisions).

```python
combined_sim = ss.similarity(region_id_1, region_id_2)  # Returns [0, 4]
```

**Formula:** S = color_sim + fill_sim + texture_sim + size_sim

**Range:** [0, 4]

## Data Structures

### Region Dictionary

Each region in `ss.regions` has the following structure:

```python
{
    'label': int,                          # Unique region ID
    'indices': np.ndarray,                 # Pixel indices (shape: (n_pixels,))
    'size': int,                           # Number of pixels
    'color_hist': np.ndarray,              # Shape (3, 25) - RGB histograms
    'texture': np.ndarray,                 # Shape (10,) - gradient histogram
    'bounding_box': tuple,                 # (y_min, y_max, x_min, x_max)
    'neighbors': set,                      # Set of neighbor region IDs
}
```

### Bounding Box Format

```
(y_min, y_max, x_min, x_max)
```

**Coordinate System:**
- `y`: Vertical axis (rows), increases downward
- `x`: Horizontal axis (columns), increases rightward
- All values are pixel indices (integers)

**Conversion to drawing coordinates:**
```python
height = y_max - y_min
width = x_max - x_min
# For matplotlib Rectangle: Rectangle((x_min, y_min), width, height)
```

## Parameter Tuning

### Initial Segmentation Parameters

In `get_initial_segments()`:

```python
segmentation.felzenszwalb(self.img, scale=100, sigma=0.5, min_size=50)
```

| Parameter | Effect | Suggested Range |
|-----------|--------|-----------------|
| `scale` | Region size (higher → larger) | 50-200 |
| `sigma` | Blur bandwidth | 0.3-1.0 |
| `min_size` | Minimum region pixels | 30-100 |

**Effects:**
- **Larger `scale`**: Fewer initial regions, coarser segmentation
- **Larger `min_size`**: Removes tiny segments, speeds up processing

### Similarity Metric Weights

Current weights (in `similarity()` method):

```python
return s1 + s2 + s4  # color + fill + size (texture excluded)
```

**Current behavior:**
- Color: Weight 1.0 (appearance most important)
- Fill: Weight 1.0 (spatial coherence)
- Texture: Weight 0.0 (not used)
- Size: Weight 1.0 (regularization)

**To adjust:**
```python
# Example: emphasize spatial coherence
return s1 + 1.5*s2 + s3 + s4  # Increase fill weight

# Example: include texture
return s1 + s2 + 0.5*s3 + s4  # Add texture with lower weight
```

### Quality Filtering

Optional threshold in `hierarchical_grouping()`:

```python
# Uncomment to only keep high-confidence merges
if neg_sim < -0.25:  # Similarity threshold
    candidates.append((merged_region['bounding_box'], merged_region['indices']))
```

## Performance Notes

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Initialization | O(H×W) | Segmentation + region extraction |
| `hierarchical_grouping()` | O(n log n) | n = number of neighbors |
| Feature extraction | O(H×W) | One-time cost |
| Similarity computation | O(1) | Constant-time per pair |

### Space Complexity

- Region storage: O(n) where n = number of segments
- Heap: O(neighbor pairs) ≈ O(n)
- Total: O(H×W) for pixel storage + O(n) for region metadata

### Typical Performance (CPU)

| Image Size | Proposals | Time | Memory |
|-----------|-----------|------|--------|
| 480×640 | ~1000 | 1-2s | ~100MB |
| 480×640 (scaled 0.15x) | ~1000 | 0.2s | ~10MB |
| 1024×768 | ~2000 | 5-10s | ~300MB |

### Optimization Strategies

For large images or real-time requirements:

1. **Downscale image**: 
   ```python
   import cv2
   scale = 0.5
   img_small = cv2.resize(img, None, fx=scale, fy=scale)
   ```

2. **Use Numba JIT compilation** for `construct_neighboring_segments()` and `get_bounding_boxes()`:
   ```python
   from numba import jit
   
   @jit(nopython=True)
   def construct_neighbors_fast(...):
       # Fast compiled version
   ```

3. **Parallelize** with Rust extension (see Performance Analysis)

4. **Filter candidates** by similarity threshold or size constraints

## Common Issues

### Issue: Too Few Proposals

**Cause:** Initial segmentation too coarse or similarity threshold too high

**Solution:**
```python
# Decrease scale for finer initial segmentation
segmentation.felzenszwalb(img, scale=50, ...)  # Default: 100

# Reduce quality threshold (comment out filtering)
```

### Issue: Too Many Proposals

**Cause:** Initial segmentation too fine or all merges are accepted

**Solution:**
```python
# Increase scale for coarser segmentation
segmentation.felzenszwalb(img, scale=150, ...)

# Filter by area
candidates = [(b, idx) for b, idx in candidates if len(idx) > 100]
```

### Issue: Slow Performance

**Cause:** Large image or complex initial segmentation

**Solution:**
1. Downscale image before processing
2. Increase `min_size` parameter
3. Increase initial `scale` (fewer segments to merge)
4. Use Numba/Rust optimization

## Comparison with Alternatives

| Method | Advantages | Disadvantages |
|--------|-----------|----------------|
| **Selective Search** | Deterministic, multi-scale, good recall | CPU-based, slower than deep learning |
| **Sliding Windows** | Simple, exhaustive | Computationally expensive, poor aspect ratios |
| **Region CNN (R-CNN)** | End-to-end learning, faster | Requires labeled data, black-box |
| **Edge Boxes** | Fast, good precision | Requires edge detection, less flexible |

## Extensions & Future Work

### Potential Improvements

1. **Multiple strategies**: Implement color-only, texture-only variants
2. **Machine learning**: Learn optimal similarity weights from labeled data
3. **Parallelization**: Use Rust/Numba for bottleneck operations
4. **Advanced features**: Add HOG, SIFT, or learned CNN features
5. **GPU acceleration**: Port similarity computations to CUDA/OpenCL

### Adding Custom Similarity Metrics

```python
class SelectiveSearchCustom(SelectiveSearch):
    def custom_similarity(self, r1, r2):
        """Your custom metric here"""
        # Return [0, 1] similarity score
        return ...
    
    def similarity(self, r1, r2):
        s1 = self.color_similarity(r1, r2)
        s2 = self.fill_similarity(r1, r2)
        s3 = self.custom_similarity(r1, r2)
        return s1 + s2 + s3
```

## References

1. **Original Paper**: [Selective Search for Object Recognition (Uijlings et al., 2013)](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)

2. **Felzenszwalb Segmentation**: [Efficient Graph-Based Image Segmentation (Felzenszwalb & Huttenlocher, 2004)](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)

3. **Related Work**:
   - R-CNN (Girshick et al., 2014) - Uses Selective Search for proposals
   - Edge Boxes (Zitnick & Dollár, 2014) - Alternative proposal method
   - Faster R-CNN (Ren et al., 2015) - Learns proposals end-to-end

## License

This implementation is provided for educational and research purposes.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{uijlings2013selective,
  title={Selective search for object recognition},
  author={Uijlings, JRR and Van De Sande, KE and Gevers, T and Smeulders, AWM},
  journal={International journal of computer vision},
  volume={104},
  number={2},
  pages={154--171},
  year={2013},
  publisher={Springer}
}
```

## Questions & Troubleshooting

### How many proposals do I typically get?

On a 480×640 image, expect 500-2000 proposals depending on content complexity. Use `len(candidates)` to check.

### Can I control the number of proposals?

Yes, via:
- `scale` parameter (higher → fewer)
- Size filtering: `[c for c in candidates if len(c[1]) > threshold]`
- Similarity threshold (uncomment in `hierarchical_grouping()`)

### What's the best way to filter proposals?

Common approaches:
```python
# By area
large_candidates = [c for c in candidates if len(c[1]) > 500]

# By aspect ratio
filtered = [c for c in candidates 
            if 0.2 < (h/(w+1)) < 5.0  # h, w from bbox]

# By consensus (if using multiple runs)
```

### How does this compare to anchor boxes in Faster R-CNN?

- **Selective Search**: Detects arbitrary-shaped regions, non-grid
- **Anchor boxes**: Pre-defined shapes at each grid point
- Selective Search has better recall but slower; anchors are faster but require careful tuning
