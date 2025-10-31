# Installation & Setup Guide

## Quick Start

### 1. Prerequisites

- **Python 3.7+** (3.9+ recommended)
- **pip** package manager

### 2. Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

This installs:
- `numpy` - Numerical computing
- `scikit-image` - Image segmentation (Felzenszwalb)
- `opencv-python` - Image I/O (for `RoiProposals`)

### 3. Verification

Test the installation:

```bash
python -c "from selective_search import SelectiveSearch; print('✓ SelectiveSearch OK')"
python -c "from roi_proposals import RoiProposals; print('✓ RoiProposals OK')"
```

Or run the test:

```bash
python test.py
```

---

## Virtual Environment Setup (Recommended)

### Using venv (Built-in)

```bash
# Create environment
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Using conda

```bash
# Create environment
conda create -n selective-search python=3.9

# Activate
conda activate selective-search

# Install packages
conda install numpy scikit-image opencv
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.19.0 | Numerical operations, arrays |
| `scikit-image` | ≥0.18.0 | Felzenszwalb segmentation |
| `opencv-python` | ≥4.5.0 | Image I/O (cv2) |

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'skimage'"

```bash
pip install scikit-image
```

### Issue: "ModuleNotFoundError: No module named 'cv2'"

```bash
pip install opencv-python
```

### Issue: "NumPy version conflict"

```bash
pip install --upgrade numpy
```

### Issue: Installation hangs on Windows

Try using conda instead:

```bash
conda install numpy scikit-image opencv
```

### Issue: "Permission denied" during installation

```bash
pip install --user -r requirements.txt
```

Or use a virtual environment (recommended).

---

## Environment Verification

Test your installation:

```python
# test_installation.py
import numpy as np
import cv2
from selective_search import SelectiveSearch
from roi_proposals import RoiProposals

print("✓ All imports successful!")

# Test SelectiveSearch with sample image
img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
ss = SelectiveSearch(img)
candidates = ss.hierarchical_grouping()
print(f"✓ SelectiveSearch: Generated {len(candidates)} proposals")

# Test RoiProposals
roi = RoiProposals()
proposals = roi.get_proposals(img)
print(f"✓ RoiProposals: Generated {len(proposals)} proposals")
print(f"✓ Installation verified successfully!")
```

Run it:

```bash
python test_installation.py
```

---

## Usage Example

### Batch Processing (JSON + Multiprocessing)

```python
from roi_proposals import RoiProposals

roi = RoiProposals()
roi.process_folder(
    input_folder='img/',
    output_folder='bboxes/',
    num_workers=4
)

# Output: bboxes/proposals.json
# {
#   "image1.jpg": [
#     {"x": 10, "y": 20, "w": 100, "h": 80},
#     ...
#   ]
# }
```

### Single Image

```python
from roi_proposals import RoiProposals
import cv2

roi = RoiProposals()
image = cv2.imread('test.jpg')
proposals = roi.get_proposals(image)

for x, y, w, h in proposals:
    print(f"Proposal: ({x}, {y}, {w}, {h})")
```

### Core Algorithm

```python
from selective_search import SelectiveSearch
import numpy as np

img = np.array(...)  # RGB image (H, W, 3)
ss = SelectiveSearch(img)

# Get proposals with pixel indices (for segmentation tasks)
candidates = ss.hierarchical_grouping()

for bbox, indices in candidates:
    y_min, y_max, x_min, x_max = bbox
    # Use indices for segmentation if needed
    print(f"Proposal: ({x_min}, {y_min})-({x_max}, {y_max})")
```

---

## Next Steps

1. **Read the documentation:** See `README.md`
2. **Explore examples:** Check `selective_search.ipynb`
3. **Review code:** Study `selective_search.py` docstrings
4. **Batch processing:** Use `RoiProposals.process_folder()` for images
