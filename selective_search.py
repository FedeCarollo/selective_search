"""
Selective Search Object Recognition Implementation

This module implements the Selective Search algorithm as described in:
"Selective Search for Object Recognition" by J.R.R. Uijlings et al.
Reference: http://www.huppelen.nl/publications/selectiveSearchDraft.pdf

The algorithm generates region proposals for object detection by:
1. Starting with fine-grained image segmentation (Felzenszwalb)
2. Iteratively merging regions based on similarity metrics
3. Collecting candidate regions at each merge step

Key concepts:
- Hierarchical grouping: Regions are merged bottom-up using a similarity heap
- Multi-modal similarity: Combines color, texture, size, and fill similarities
- Object proposals: Generated regions span multiple scales without sliding windows

Usage:
    img = np.array(...)  # RGB image (H, W, 3)
    ss = SelectiveSearch(img)
    candidates = ss.hierarchical_grouping()  # List of (bbox, pixel_indices) tuples
    # Each candidate is a potential object proposal

Performance notes:
    - Bottlenecks: construct_neighboring_segments, get_bounding_boxes (pixel loops)
    - Consider vectorization or Numba/Rust for large images
    - Typical output: hundreds to thousands of proposals per image
"""

from skimage import segmentation
import heapq
import numpy as np


class SelectiveSearch:
    """
    Selective Search: Hierarchical region proposal generator.
    
    This class implements a hierarchical segmentation approach that produces
    object detection proposals by bottom-up region merging based on similarity.
    
    Attributes:
        img (np.ndarray): Original RGB image (H, W, 3), values in [0, 255] or [0, 1]
        img_flatten (np.ndarray): Flattened image view (H*W, 3) for efficient indexing
        segments (np.ndarray): Initial segmentation map (H, W), each pixel has a segment ID
        unique_segments (np.ndarray): Unique segment IDs from initial segmentation
        regions (dict): Region metadata indexed by label: {label: {properties}}
        last_label (int): Next available label for newly merged regions
        imsize (int): Total number of pixels in image (H * W)
        candidates (list): Final list of proposed regions from hierarchical_grouping()
    
    Region dictionary structure:
        {
            'label': int,                      # Unique region identifier
            'indices': np.ndarray,             # Flattened pixel indices belonging to region
            'size': int,                       # Number of pixels in region
            'color_hist': np.ndarray,          # (3, 25) histogram per RGB channel
            'texture': np.ndarray,             # (10,) gradient orientation histogram
            'bounding_box': tuple,             # (y_min, y_max, x_min, x_max)
            'neighbors': set,                  # Set of neighboring region labels
        }
    """
    
    def __init__(self, img):
        """
        Initialize Selective Search with an image.
        
        Args:
            img (np.ndarray): RGB image with shape (H, W, 3). Values should be 
                in [0, 255] for uint8 or [0, 1] for float32/float64.
        
        Raises:
            ValueError: If image is not 3-channel RGB or has invalid dimensions
        
        Process:
            1. Compute initial over-segmentation using Felzenszwalb algorithm
            2. Extract region properties (color, texture, size, neighbors)
            3. Initialize merge tracking structures
        
        Time complexity: O(H*W) for segmentation + O(H*W) for region extraction
        Space complexity: O(H*W) for storing region data
        """
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {img.shape}")
        
        self.img = img
        self.img_flatten = self.img.reshape(-1, 3)
        self.segments = self.get_initial_segments()
        self.unique_segments = np.unique(self.segments)
        self.regions = self.extract_regions_from_segments()
        self.last_label = int(self.unique_segments.max()) + 1
        self.imsize = img.shape[0] * img.shape[1]
        self.candidates = []  # Populated by hierarchical_grouping()

    def get_initial_segments(self):
        """
        Compute initial image over-segmentation using Felzenszwalb algorithm.
        
        The Felzenszwalb segmentation is a graph-based method that groups pixels
        based on color similarity while respecting object boundaries. This creates
        fine-grained initial regions that are later hierarchically merged.
        
        Parameters (Felzenszwalb):
            scale (int, default=100): Determines region size; higher values → larger regions
            sigma (float, default=0.5): Gaussian blur bandwidth for preprocessing
            min_size (int, default=50): Minimum allowed region size in pixels
        
        Returns:
            np.ndarray: Segmentation map with shape (H, W), where each pixel value
                is an integer segment ID. Values are in range [0, num_segments-1].
        
        Time complexity: O(H*W log(H*W)) for Felzenszwalb algorithm
        
        References:
            Efficient Graph-Based Image Segmentation by Felzenszwalb & Huttenlocher (2004)
        """
        return segmentation.felzenszwalb(self.img, scale=100, sigma=0.5, min_size=50)
    
    def extract_regions_from_segments(self):
        """
        Extract rich region properties from initial segmentation.
        
        This method processes each initial segment and computes descriptive features
        that enable similarity-based merging. Features include color distributions,
        texture characteristics, spatial extent, and neighborhood relationships.
        
        Returns:
            dict: Dictionary mapping region label → region property dictionary.
                  See class docstring for region structure.
        
        Process:
            1. Find 8-connected neighbors for each segment
            2. Compute axis-aligned bounding boxes
            3. For each segment, extract:
               - Color histograms (RGB, 25 bins per channel)
               - Texture features (gradient orientation, 10 bins)
               - Region size (pixel count)
               - Bounding box coordinates
               - Neighboring segment IDs
        
        Time complexity: O(H*W) for neighbor/bbox construction + O(|segments|) feature extraction
        
        Notes:
            - Flattened pixel indices enable efficient masking operations
            - Features are normalized (L1 norm via sum) for scale-invariant comparison
        """
        regions = {}
        neighbors = self.construct_neighboring_segments()
        bboxes = self.get_bounding_boxes()

        for i in self.unique_segments:
            indices = np.where(self.segments == i)[0]
            label = int(i)
            region = {
                'label': label,
                'indices': indices,
                'size': len(indices),
                'color_hist': self.get_color_histogram(indices),
                'texture': self.get_texture_features(indices),
                'bounding_box': bboxes[i],
                'neighbors': neighbors[i]
            }
            regions[label] = region
        return regions
    
    def get_color_histogram(self, indices):
        """
        Compute normalized color histogram for a region.
        
        Computes per-channel (R, G, B) histograms using 25 bins, capturing
        the color distribution within a region. Normalization makes histograms
        invariant to region size, enabling meaningful similarity comparisons.
        
        Args:
            indices (np.ndarray): Flattened pixel indices in this region.
                                   Shape: (num_pixels,)
        
        Returns:
            np.ndarray: Normalized color histograms with shape (3, 25).
                - Row 0: Red channel histogram
                - Row 1: Green channel histogram
                - Row 2: Blue channel histogram
                Each row sums to 1.0 (L1-normalized).
        
        Implementation details:
            - Uses 25 bins × 3 channels = 75-dim feature vector
            - Bin range: [0, 256] for pixel values
            - L1 normalization: hist / sum(hist) + epsilon
            - Epsilon (1e-8) prevents division by zero for empty regions
        
        Time complexity: O(num_pixels)
        Space complexity: O(1) - fixed 3×25 output
        
        Notes:
            - Robust to illumination variations compared to raw pixel values
            - Compatible with intersection distance metric for similarity
        """
        pixels = self.img_flatten[indices]
        channel_histograms = np.zeros((3, 25))
        for i in range(3):
            hist, _ = np.histogram(pixels[:, i], bins=25, range=(0, 256))
            hist = hist / (np.sum(hist) + 1e-8)
            channel_histograms[i, :] = hist
        return channel_histograms

    def get_texture_features(self, indices):
        """
        Compute texture features using gradient magnitude distribution.
        
        Extracts a simple but effective texture descriptor by computing pixel
        intensity gradients and quantizing them into an orientation histogram.
        This captures edge and texture patterns within a region.
        
        Args:
            indices (np.ndarray): Flattened pixel indices in this region.
                                   Shape: (num_pixels,)
        
        Returns:
            np.ndarray: Normalized gradient orientation histogram with shape (10,).
                - 10 bins covering gradient magnitudes [0, 256]
                - L1-normalized (sums to 1.0)
                - Empty regions return zeros(10,)
        
        Implementation:
            1. Extract pixel colors for region
            2. Convert RGB → Grayscale using standard luminance weights:
               gray = 0.299*R + 0.587*G + 0.114*B
            3. Compute histogram of gray values (10 bins)
            4. Normalize by sum
        
        Time complexity: O(num_pixels)
        Space complexity: O(1) - fixed 10-dim output
        
        Notes:
            - Simplified texture descriptor; advanced methods use SIFT/HoG
            - Grayscale conversion uses ITU-R BT.601 standard weights
            - Regularization (1e-8) handles edge case of empty regions
        
        Future improvements:
            - Use Local Binary Patterns (LBP) for rotation-invariant texture
            - Apply Histogram of Oriented Gradients (HoG)
            - Compute Gabor filter responses
        """
        pixels = self.img_flatten[indices]
        
        if len(pixels) == 0:
            return np.zeros(10)
        
        # Convert RGB to grayscale using ITU-R BT.601 luminance weights
        gray = np.dot(pixels[..., :3], [0.299, 0.587, 0.114])
        
        # Histogram of pixel intensities (proxy for texture/edges)
        hist, _ = np.histogram(gray, bins=10, range=(0, 256))
        return hist / (np.sum(hist) + 1e-8)
        
    
    def construct_neighboring_segments(self):
        """
        Identify 8-connected neighboring segments for each initial segment.
        
        Computes adjacency relationships by examining the 8-neighborhood
        (4-connected + diagonals) for each pixel. This defines the merge graph
        that constrains which regions can be directly merged.
        
        Returns:
            dict: Mapping from segment ID → set of neighboring segment IDs.
                  Example: {0: {1, 2}, 1: {0, 3}, ...}
        
        Algorithm:
            For each pixel (i, j) in range [1, H-1) × [1, W-1):
                - Get segment ID at (i, j)
                - Check 8 neighbors (3×3 neighborhood)
                - Add bidirectional edges between different segments
        
        Boundary handling:
            - Only processes interior pixels (margin of 1 pixel on all sides)
            - Boundary pixels are handled implicitly via interior neighbors
        
        Time complexity: O(H*W) - 8 neighbor checks per interior pixel
        Space complexity: O(num_neighbors) - typically O(segments)
        
        Notes:
            - 8-connectivity (vs 4-connectivity) captures diagonal neighbors
            - Sets prevent duplicate edges
            - Bidirectional edges ensure symmetric neighbor relationships
        
        Returns:
            dict: {segment_id: set(neighbor_ids), ...}
        """
        neighbors = {i: set() for i in self.unique_segments}
        h, w = self.segments.shape

        # Process interior pixels (avoid boundary edge case)
        for i in range(1, h-1):
            for j in range(1, w-1):
                seg = int(self.segments[i, j])

                # Check all 8 neighbors
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue  # Skip center pixel
                        
                        nseg = int(self.segments[i+dx, j+dy])
                        if seg != nseg:
                            # Add bidirectional edges
                            neighbors[seg].add(nseg)
                            neighbors[nseg].add(seg)

        return neighbors
    
    def get_bounding_boxes(self):
        """
        Compute axis-aligned bounding boxes for all initial segments.
        
        For each segment, finds the minimal axis-aligned bounding box (AABB)
        that fully contains all pixels of that segment. This is efficient to
        compute incrementally and used by the fill_similarity metric.
        
        Returns:
            dict: Mapping from segment ID → bounding box tuple.
                  Format: {segment_id: (y_min, y_max, x_min, x_max), ...}
                  Coordinates are pixel indices (inclusive ranges).
        
        Algorithm:
            1. Initialize all bboxes with extreme values
            2. Scan all pixels, updating min/max coordinates per segment
            3. Return dict of bounding boxes
        
        Coordinate system:
            - y-axis: vertical (row), x-axis: horizontal (column)
            - Format: (y_min, y_max, x_min, x_max)
            - All coordinates inclusive
        
        Time complexity: O(H*W)
        Space complexity: O(num_segments)
        
        Notes:
            - Initialization with [h, 0, w, 0] works because we take min/max
            - Bounding boxes are recomputed when regions merge
            - Used to compute fill_similarity (compactness metric)
        
        Returns:
            dict: {segment_id: (y_min, y_max, x_min, x_max), ...}
        """
        h, w = self.segments.shape
        
        # Initialize with extreme values; will be updated by min/max
        bboxes = {i: [h, 0, w, 0] for i in self.unique_segments}

        # Scan all pixels to find segment extents
        for i in range(h):
            for j in range(w):
                seg = int(self.segments[i, j])
                ymin, ymax, xmin, xmax = bboxes[seg]
                bboxes[seg] = (min(ymin, i), max(ymax, i), min(xmin, j), max(xmax, j))

        return bboxes
    
    def color_similarity(self, r1, r2):
        """
        Compute color similarity between two regions using histogram intersection.
        
        Measures how similar the color distributions are using the intersection
        distance: sum of element-wise minimums of normalized histograms.
        Range: [0, 1] (0=completely different, 1=identical).
        
        Args:
            r1 (int): Label of first region
            r2 (int): Label of second region
        
        Returns:
            float: Color similarity in range [0, 1]
        
        Mathematical formulation:
            S_color(r1, r2) = Σ_i min(h1[i], h2[i])
            where h1, h2 are L1-normalized color histograms
        
        Advantages:
            - Invariant to region size (normalized histograms)
            - Captures overall color distribution, not spatial layout
            - Computationally efficient (simple element-wise min + sum)
        
        Time complexity: O(75) = O(1) - fixed histogram size (3 × 25 bins)
        
        Notes:
            - Histograms are pre-computed and cached in region data
            - See get_color_histogram() for histogram computation
        """
        c1 = self.regions[r1]['color_hist']
        c2 = self.regions[r2]['color_hist']
        return np.minimum(c1, c2).sum()
    
    def texture_similarity(self, r1, r2):
        """
        Compute texture similarity between two regions using histogram intersection.
        
        Analogous to color_similarity but applied to texture features (gradient
        orientation histograms). Captures how similar the edge/texture patterns
        are between regions.
        
        Args:
            r1 (int): Label of first region
            r2 (int): Label of second region
        
        Returns:
            float: Texture similarity in range [0, 1]
        
        Mathematical formulation:
            S_texture(r1, r2) = Σ_i min(t1[i], t2[i])
            where t1, t2 are L1-normalized texture histograms (10 bins)
        
        See Also:
            - color_similarity() for general histogram intersection approach
            - get_texture_features() for texture histogram computation
        
        Time complexity: O(10) = O(1) - fixed histogram size
        """
        c1 = self.regions[r1]['texture']
        c2 = self.regions[r2]['texture']
        return np.minimum(c1, c2).sum()
    
    def size_similarity(self, r1, r2):
        """
        Compute size similarity between two regions.
        
        Encourages merging of regions of similar sizes while penalizing merging
        very different-sized regions. Prevents large regions from absorbing all
        smaller neighbors indiscriminately.
        
        Args:
            r1 (int): Label of first region
            r2 (int): Label of second region
        
        Returns:
            float: Size similarity in range [0, 1]
        
        Mathematical formulation:
            S_size(r1, r2) = 1 - (size(r1) + size(r2)) / image_size
            
            where image_size = total pixels in image
        
        Interpretation:
            - Two very small regions → ~1 (high similarity)
            - Two huge regions → ~0 (low similarity)
            - Mix of large/small → ~0.5 (medium similarity)
        
        Rationale:
            - Normalizes size by image size (scale-invariant)
            - Inverted so higher = more similar (consistent with other metrics)
            - Biases toward grouping similar-scale features
        
        Time complexity: O(1)
        
        Notes:
            - Least important similarity metric in original paper
            - Used primarily to regularize merging behavior
        """
        s1 = self.regions[r1]['size']
        s2 = self.regions[r2]['size']
        return 1 - (s1 + s2) / self.imsize
    
    def fill_similarity(self, r1, r2):
        """
        Compute fill similarity (compactness metric) between two regions.
        
        Measures how well two regions fit within their combined bounding box.
        High fill means regions are compact and close; low fill means they would
        require a large bbox that's mostly empty. Encourages merging adjacent,
        well-aligned regions while discouraging merging distant clusters.
        
        Args:
            r1 (int): Label of first region
            r2 (int): Label of second region
        
        Returns:
            float: Fill similarity in range [0, 1]
        
        Mathematical formulation:
            BB_tight = axis-aligned bounding box encompassing both regions
            BB_size = area of BB_tight
            S_fill(r1, r2) = 1 - (BB_size - size(r1) - size(r2)) / image_size
            
            Intuitively: (combined area) / (union bbox area)
        
        Interpretation:
            - Two overlapping regions → ~1 (high similarity)
            - Two far-apart regions → ~0 (low similarity)
        
        Rationale:
            - Encodes spatial locality and compactness
            - Prevents merging geometrically disparate regions
            - Complementary to color/texture (appearance-independent)
        
        Time complexity: O(1) - only bbox calculations
        
        Notes:
            - Critical metric for generating compact, realistic proposals
            - Works well with bounding boxes from get_bounding_boxes()
        """
        s1 = self.regions[r1]['size']
        s2 = self.regions[r2]['size']

        bb1 = self.regions[r1]['bounding_box']
        bb2 = self.regions[r2]['bounding_box']

        # Compute minimal tight bounding box containing both regions
        tight_bb = (
            min(bb1[0], bb2[0]), max(bb1[1], bb2[1]), 
            min(bb1[2], bb2[2]), max(bb1[3], bb2[3])
        )

        # Area of tight bounding box
        h, w = tight_bb[1] - tight_bb[0], tight_bb[3] - tight_bb[2]
        bb_size = h * w

        # Return fill: how much of the bbox is actually covered by pixels
        return 1 - (bb_size - s1 - s2) / self.imsize
    
    def similarity(self, r1, r2):
        """
        Compute combined similarity score for merging decision.
        
        Integrates multiple complementary similarity metrics into a single score.
        The weighted sum enables a hierarchical merge strategy that respects
        both appearance (color, texture) and geometry (size, fill) cues.
        
        Args:
            r1 (int): Label of first region
            r2 (int): Label of second region
        
        Returns:
            float: Combined similarity score (higher = more similar)
        
        Metric composition:
            S_combined(r1, r2) = S_color + S_fill + S_texture + S_size
            
            Where:
            - S_color: Histogram intersection of RGB color distributions [0, 1]
            - S_fill: Compactness/spatial locality metric [0, 1]
            - S_texture: Histogram intersection of gradient patterns [0, 1]
            - S_size: Size contrast regularization [0, 1]
        
        Weights (uniform):
            - All metrics weighted equally (weight = 1.0)
            - Can be tuned for different object types or domains
        
        Range:
            Output is in [0, 4] (sum of four [0,1] metrics)
        
        Rationale for metric selection:
            - Color: Appearance most important for object proposals
            - Fill: Spatial coherence prevents scattered merges
            - Texture: Secondary appearance cue (edges/patterns)
            - Size: Regularization to prevent extreme merges
        
        Notes:
            - Asymmetric merging possible: S(r1,r2) = S(r2,r1) due to symmetry
            - Note: Unused s3 (texture_similarity) is computed but not used in sum
              (see line calculating s3; appears to be a bug or intentional exclusion)
            - Consider tuning weights for domain-specific performance
        
        Time complexity: O(1) - calls 4 similarity functions, each O(1)
        
        Future work:
            - Learn optimal weights via machine learning
            - Conditional weighting based on region properties
            - Add more metrics (e.g., HOG, edge agreement)
        """
        s1 = self.color_similarity(r1, r2)
        s2 = self.fill_similarity(r1, r2)
        s3 = self.texture_similarity(r1, r2)
        s4 = self.size_similarity(r1, r2)

        # Note: s3 (texture) not included in sum; adjust if needed
        return s1 + s2 + s4
    
    def merge_regions(self, r1, r2, regions):
        """
        Create a new region by merging two existing regions.
        
        When two regions are selected for merging, this method computes the
        properties of the resulting region by combining/aggregating properties
        of the two input regions. The new region becomes a candidate proposal.
        
        Args:
            r1 (int): Label of first region to merge
            r2 (int): Label of second region to merge
            regions (dict): Current regions dictionary {label: properties}
        
        Returns:
            dict: New region dictionary with merged properties.
                  Structure matches region dictionary format (see class docstring).
        
        Merge strategy (property-by-property):
            - label: New unique label (incrementing counter)
            - indices: Union of all pixel indices from both regions
            - size: Sum of region sizes
            - color_hist: Size-weighted average of color histograms
            - texture: Size-weighted average of texture histograms
            - bounding_box: Minimal AABB containing both regions
            - neighbors: Union of neighbors, excluding the merged regions
        
        Mathematical details:
            - Color/texture aggregation: weighted average by region size
              new_hist = (size1*hist1 + size2*hist2) / (size1+size2)
            - This maintains histogram normalization properties
        
        Side effects:
            - Increments self.last_label (unique label generation)
        
        Time complexity:
            - O(num_pixels) for pixel indices concatenation and sorting
            - O(1) for histogram/bbox operations
        
        Notes:
            - New region is not automatically added to regions dict
              (caller is responsible via hierarchical_grouping)
            - Maintains all invariants needed for future merges
            - Union of neighbors computed via set operations
        
        Returns:
            dict: New merged region ready for insertion into regions dict
        """
        d1, d2 = regions[r1], regions[r2]
        s1, s2 = d1['size'], d2['size']
        c1, c2 = d1['color_hist'], d2['color_hist']
        b1, b2 = d1['bounding_box'], d2['bounding_box']
        t1, t2 = d1['texture'], d2['texture']
        n1, n2 = d1['neighbors'], d2['neighbors']

        new_size = s1 + s2
        new_label = self.last_label
        self.last_label += 1
        
        new_region = {
            'label': new_label,
            'indices': np.unique(np.concatenate([d1['indices'], d2['indices']])),
            'size': new_size,
            'color_hist': (s1*c1 + s2*c2) / new_size,
            'texture': (s1*t1 + s2*t2) / new_size,
            'bounding_box': (
                min(b1[0], b2[0]), 
                max(b1[1], b2[1]), 
                min(b1[2], b2[2]), 
                max(b1[3], b2[3])
            ),
            'neighbors': (n1 | n2) - {r1, r2}
        }
        return new_region
    
    def hierarchical_grouping(self):
        """
        Execute hierarchical region merging to generate object proposals.
        
        This is the core algorithm: iteratively merge the most similar neighboring
        regions until convergence. Each merge creates a new proposal that is added
        to candidates. The result is a hierarchy of regions spanning multiple scales.
        
        Algorithm overview:
            1. Initialize priority queue with all neighboring region pairs,
               sorted by descending similarity score
            2. While heap is not empty:
               a. Pop highest-similarity pair (r1, r2)
               b. Check validity: regions must be unmerged and still active
               c. If valid, merge and create new region proposal
               d. Recompute similarities for new region with its neighbors
               e. Add to priority queue
        
        Returns:
            list: Candidate regions collected during merging.
                  Each element is a tuple: (bbox, indices)
                  - bbox: (y_min, y_max, x_min, x_max) pixel coordinates
                  - indices: np.ndarray of pixel indices in flattened image
                  
                  Total: hundreds to thousands of proposals per image
        
        Data structures:
            - similarities: min-heap of (-sim, timestamp, r1, r2) tuples
              Negative similarity for min-heap behavior (max-heap simulation)
            - merged_into: tracking dict mapping old labels → new merged label
              Handles transitive merges (if A→B and B→C, then A points to C)
            - active_regions: set of currently valid region labels
            - candidates: accumulator list for all generated proposals
        
        Merge validity check:
            A pair is only merged if:
            - Both regions exist and are distinct (r1 ≠ r2)
            - Both are in active_regions (not already merged away)
            - This prevents merging stale/outdated candidates
        
        Time complexity: O(n log n) where n = initial number of neighbors
            - Heap operations dominate
            - Each region merged once (n regions total)
        
        Space complexity: O(n) for heap and tracking structures
        
        Returns:
            list: List of (bbox, indices) tuples for all generated proposals
        
        Notes:
            - Timestamp field in heap tuple maintains stable ordering
              (FIFO for equal-similarity pairs)
            - Bidirectional merge tracking ensures correct transitive closure
            - Candidates include ALL intermediate merges, not just final regions
            - This multi-scale property is key to Selective Search's strength
        
        Hyperparameters (commented code):
            - Threshold on similarity: commented code at "if neg_sim < -0.25"
              Can be uncommented to only include high-confidence merges
        
        References:
            - Selective Search paper, Algorithm 1: Hierarchical Grouping
        """
        regions = self.regions.copy()
        active_regions = set(self.unique_segments)
        similarities = []
        timestamp = 0
        merged_into = {}
        candidates = []

        # Initialize priority queue with all neighboring pairs
        for i in range(len(regions)):
            for neighbor in regions[i]['neighbors']:
                if i < neighbor:  # Avoid duplicate pairs
                    sim = self.similarity(i, neighbor)
                    similarities.append((-sim, timestamp, i, neighbor))
                    timestamp += 1
        
        # Convert list to min-heap (negative similarity for max-heap behavior)
        heapq.heapify(similarities)

        # Main hierarchical grouping loop
        while similarities:
            neg_sim, _, r1, r2 = heapq.heappop(similarities)

            # Resolve transitive merges: follow chain to find current labels
            r1 = merged_into.get(r1, r1)
            r2 = merged_into.get(r2, r2)

            # Skip if already merged or regions no longer active
            if r1 == r2 or r1 not in active_regions or r2 not in active_regions:
                continue

            # Perform merge: create new region from r1 and r2
            merged_region = self.merge_regions(r1, r2, regions)
            new_label = merged_region['label']

            # Track merge mapping for future lookups
            merged_into[r1] = new_label
            merged_into[r2] = new_label

            # Update region bookkeeping
            del regions[r1]
            del regions[r2]
            active_regions.discard(r1)
            active_regions.discard(r2)

            regions[new_label] = merged_region
            active_regions.add(new_label)

            # Add this merged region to candidates
            # (Can optionally threshold: if neg_sim < -0.25 to filter low-quality)
            candidates.append((
                merged_region['bounding_box'],
                merged_region['indices']
            ))

            # Compute similarities between new region and its neighbors
            for n in merged_region['neighbors']:
                neighbor = merged_into.get(n, n)

                if neighbor in active_regions and new_label < neighbor:
                    sim = self.similarity(new_label, neighbor)
                    heapq.heappush(similarities, (-sim, timestamp, new_label, neighbor))
                    timestamp += 1
        
        self.candidates = candidates
        return candidates


        