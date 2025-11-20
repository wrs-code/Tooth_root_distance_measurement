# CLAUDE.md - AI Assistant Development Guide

> **Last Updated**: 2025-11-20
> **Repository**: Tooth Root Distance Measurement System
> **Purpose**: Automated tooth root spacing measurement and orthodontic risk assessment from panoramic X-ray images

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Core Technologies](#core-technologies)
4. [Development Workflow](#development-workflow)
5. [Key Components](#key-components)
6. [Code Conventions](#code-conventions)
7. [Testing & Debugging](#testing--debugging)
8. [Common Tasks](#common-tasks)
9. [Technical Algorithms](#technical-algorithms)
10. [Future Development](#future-development)

---

## Project Overview

### Purpose
This system automatically measures tooth root spacing from panoramic X-ray images and performs orthodontic risk assessment. It's designed for clinical use in orthodontic treatment planning.

### Key Features
1. **U-Net Deep Learning Segmentation** - Automatic tooth detection and segmentation
2. **CEJ Line Detection** - Identifies Cemento-Enamel Junction (boundary between crown and root)
3. **Morphological Analysis** - Uses morphological operations for robust boundary detection
4. **Root Spacing Measurement** - Measures inter-root distances at multiple depths
5. **Risk Color Coding**:
   - 🔴 Red: < 3.2mm (Danger)
   - 🟡 Yellow: 3.2-4.0mm (Relatively Safe)
   - 🟢 Green: ≥ 4.0mm (Safe)
6. **Batch Processing** - Processes multiple X-ray images
7. **Comprehensive Visualization** - 4-panel analysis reports

### Clinical Context
- **Target Users**: Orthodontists, dental professionals
- **Input**: Panoramic X-ray images (PNG, JPG)
- **Output**: Visual analysis reports + text summaries
- **Safety Threshold**: Based on clinical orthodontic safety standards

---

## Codebase Structure

```
Tooth_root_distance_measurement/
├── input/                          # Input X-ray images
│   └── image.png                   # Sample X-ray image
├── output/                         # Generated analysis results (gitignored)
│   ├── *_analysis.png             # 4-panel visualization per image
│   └── summary_report.txt         # Aggregate statistics
├── Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net/
│   └── ...                        # Reference U-Net implementation
├── models/                        # Deep learning model weights
│   └── *.h5                       # Model files (gitignored)
├── tooth_cej_root_analyzer.py     # MAIN PROGRAM - Morphology-based system
├── unet_segmentation.py           # U-Net integration + post-processing
├── tooth_spacing_color_demo.py    # Legacy demo (geometric model)
├── debug_segmentation.py          # Debugging utilities
├── test_fixes.py                  # Unit tests
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git exclusions
├── README.md                      # Project introduction (Chinese)
├── CEJ_ANALYSIS_README.md         # Technical documentation (Chinese)
├── FIXES_APPLIED.md              # Patch history
├── TEETH_SEGMENTATION_FIX.md     # Segmentation improvements
├── 牙根距离测量方法说明.md         # Detailed method explanation (Chinese)
├── 口腔正畸牙根间距自动测量与风险标注软件技术需求文档.pdf
└── CLAUDE.md                      # This file
```

### Key Files

| File | Purpose | When to Modify |
|------|---------|----------------|
| `tooth_cej_root_analyzer.py` | Main analysis engine | Core algorithm changes, new features |
| `unet_segmentation.py` | Tooth segmentation | Segmentation improvements, model updates |
| `tooth_spacing_color_demo.py` | Legacy demo | Reference only (deprecated) |
| `debug_segmentation.py` | Debugging tools | Add new debug visualizations |
| `test_fixes.py` | Unit tests | After bug fixes or new features |
| `requirements.txt` | Dependencies | Adding new libraries |

---

## Core Technologies

### Deep Learning
- **Framework**: TensorFlow 2.8+
- **Architecture**: U-Net (for semantic segmentation)
- **Model**: Pretrained on dental X-ray dataset
- **Reference**: [SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net)

### Computer Vision
- **Library**: OpenCV 4.5+
- **Key Techniques**:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Bilateral filtering
  - Otsu's thresholding
  - Morphological operations (opening, closing, erosion)
  - Connected Component Analysis (CCA)

### Analysis & Visualization
- **NumPy**: Numerical operations
- **SciPy**: Gradient analysis, distance calculations
- **Matplotlib**: Multi-panel visualizations

---

## Development Workflow

### Git Branch Strategy

#### Branch Naming Convention
All development branches MUST follow this pattern:
```
claude/<feature-description>-<session-id>
```

**Examples**:
- `claude/debug-roi-edge-detection-01TUzZxsk7nwrbwUVdjpr6AH`
- `claude/rewrite-cej-detection-014U5Z6t4Hf2Q32vJs7cqhHv`
- `claude/claude-md-mi72xsgvoqra3h54-01M9awNfJCkpJv4qDvZh88kS` (current)

#### Important Rules
1. **ALWAYS** develop on the designated branch (see git status)
2. **NEVER** push to main/master directly
3. Branch names **MUST** start with `claude/` and end with session ID
4. Use descriptive feature names in the middle
5. After completion, create a Pull Request for review

### Commit Message Guidelines

#### Format
```
<type>: <concise description>

<optional detailed explanation>
```

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring
- `docs`: Documentation updates
- `test`: Test additions/modifications
- `style`: Code style changes (formatting)
- `chore`: Maintenance tasks

#### Examples
```bash
feat: 完全重写为基于形态学的牙根间距测量系统

fix: 修复中文字体显示乱码问题

refactor: 优化CEJ检测算法，移除ROI扩展
```

### Pull Request Workflow

1. **Before Creating PR**:
   ```bash
   # Ensure you're on the correct branch
   git status

   # Run tests
   python3 test_fixes.py

   # Test main functionality
   python3 tooth_cej_root_analyzer.py
   ```

2. **Create PR**:
   ```bash
   # Push to remote with -u flag
   git push -u origin <branch-name>

   # Create PR using gh CLI
   gh pr create --title "Feature: Description" --body "Summary..."
   ```

3. **PR Description Template**:
   ```markdown
   ## Summary
   - Brief description of changes
   - Key improvements or fixes

   ## Test Plan
   - [ ] Unit tests pass
   - [ ] Manual testing completed
   - [ ] Visualizations verified
   - [ ] No regression in existing features
   ```

---

## Key Components

### 1. ToothCEJAnalyzer Class
**Location**: `tooth_cej_root_analyzer.py:31`

**Purpose**: Main analysis orchestrator

**Key Methods**:
```python
def segment_teeth_from_image(image)
    # Segments teeth using U-Net + post-processing
    # Returns: teeth_data dict with contours, bboxes, etc.

def detect_cej_line(tooth_data)
    # Detects CEJ line based on width gradient
    # Returns: cej_curve (points), cej_center, cej_normal

def compute_tooth_long_axis(tooth_data)
    # Calculates tooth's principal axis
    # Returns: axis_top, axis_bottom, angle

def detect_alveolar_crest_boundary(tooth1, tooth2)
    # Finds bone ridge boundary between teeth (morphological)
    # Returns: boundary points array

def measure_spacing_along_depth(tooth1, tooth2)
    # Measures inter-root spacing at multiple depths
    # Returns: depth_spacings dict

def analyze_and_visualize(image_path)
    # Main pipeline: processes image -> generates report
```

**Configuration**:
```python
self.DANGER_THRESHOLD = 3.2   # mm, clinical safety threshold
self.WARNING_THRESHOLD = 4.0  # mm
self.pixels_per_mm = 10       # Calibration factor (adjust per X-ray)
```

### 2. UNetTeethSegmentation Class
**Location**: `unet_segmentation.py:15`

**Purpose**: U-Net model integration + post-processing

**Key Methods**:
```python
def load_model(model_path)
    # Loads pretrained U-Net model

def segment_teeth(image)
    # Runs inference + post-processing
    # Post-processing pipeline (based on CCA_Analysis.py):
    #   1. Morphological opening (noise removal)
    #   2. Sharpening filter (edge enhancement)
    #   3. Erosion (separate adjacent teeth)
    #   4. Morphological closing (fill holes)
    #   5. Connected Component Analysis
    # Returns: segmentation mask
```

**Post-Processing Pipeline** (tooth_cej_root_analyzer.py:108-139):
```python
# 1. Opening: Remove noise (kernel=5x5, iterations=2)
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 2. Sharpening: Enhance edges
kernel_sharpening = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
sharpened = cv2.filter2D(opened, -1, kernel_sharpening)

# 3. Erosion: Separate touching teeth (iterations=1)
eroded = cv2.erode(sharpened, kernel, iterations=1)

# 4. Closing: Fill internal holes (iterations=1)
closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel, iterations=1)

# 5. Otsu thresholding
thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# 6. Connected Component Analysis
labels = cv2.connectedComponents(thresh, connectivity=8)[1]
```

**Reference**: Mirrors the approach in [CCA_Analysis.py](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net/blob/master/CCA_Analysis.py) from the source repository.

---

## Code Conventions

### Python Style
- **PEP 8** compliant
- **Line length**: 100 characters max (flexible for long formulas)
- **Docstrings**: Google-style format
- **Type hints**: Preferred but not mandatory

### Naming Conventions
```python
# Classes: PascalCase
class ToothCEJAnalyzer:
    pass

# Functions/Methods: snake_case
def detect_cej_line(tooth_data):
    pass

# Constants: UPPER_SNAKE_CASE
DANGER_THRESHOLD = 3.2

# Variables: snake_case
tooth_contour = ...
```

### Comments
- **Chinese**: Algorithm explanations, domain-specific terms
- **English**: Code structure comments, TODO markers
- **Docstrings**: English (for international collaboration potential)

**Example**:
```python
def detect_cej_line(self, tooth_data):
    """
    Detects the CEJ (Cemento-Enamel Junction) line of a tooth.

    Algorithm:
    1. Scan tooth contour from top to bottom
    2. Record width at each Y-coordinate
    3. Apply Gaussian smoothing
    4. Compute width gradient (first derivative)
    5. Find maximum negative gradient in middle region (20%-60% height)

    Args:
        tooth_data: Dictionary containing tooth information

    Returns:
        Tuple of (cej_curve, cej_center, cej_normal)
    """
    # 使用梯度分析检测CEJ线（宽度开始显著减小的位置）
    ...
```

### File Organization
```python
# 1. Shebang & encoding
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2. Module docstring
"""
Brief module description.
"""

# 3. Imports (grouped)
import os  # Standard library
import sys

import cv2  # Third-party
import numpy as np

from unet_segmentation import UNetTeethSegmentation  # Local

# 4. Constants
DANGER_THRESHOLD = 3.2

# 5. Classes
class MyClass:
    pass

# 6. Main execution guard
if __name__ == "__main__":
    main()
```

---

## Testing & Debugging

### Unit Tests
**File**: `test_fixes.py`

**Run Tests**:
```bash
python3 test_fixes.py
```

**Add New Tests**:
```python
def test_new_feature():
    """Test description"""
    # Arrange
    input_data = ...

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_value, "Failure message"
```

### Debug Visualization
**File**: `debug_segmentation.py`

**Usage**:
```python
from debug_segmentation import visualize_segmentation_steps

# Visualize intermediate steps
visualize_segmentation_steps(image, mask, teeth_data)
```

**Debug Outputs**:
- Saves intermediate images to `output/debug_*.png`
- Shows preprocessing, segmentation, CEJ detection stages

### Common Issues & Solutions

#### Issue: "检测到 X 颗牙齿" (Too few teeth detected)
**Causes**:
1. U-Net model quality
2. Image preprocessing parameters
3. Morphological operation iterations

**Solutions**:
```python
# 1. Check U-Net model
analyzer.unet_segmenter.load_model('path/to/better/model.h5')

# 2. Adjust preprocessing (in segment_teeth_from_image)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))  # Increase clipLimit

# 3. Tune morphological parameters (in unet_segmentation.py)
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Reduce iterations
```

#### Issue: "中文字体显示为方框"
**Cause**: Missing Chinese fonts

**Solution**:
```bash
# Install fonts
apt-get install fonts-wqy-zenhei fonts-wqy-microhei

# Verify in code
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
```

#### Issue: "CEJ线不准确"
**Causes**:
1. Gradient search range too broad/narrow
2. Width curve too noisy
3. Unusual tooth morphology

**Solutions**:
```python
# Adjust search range (in detect_cej_line)
search_start = 0.15  # Was 0.20
search_end = 0.65    # Was 0.60

# Increase smoothing
width_smooth = gaussian_filter1d(widths, sigma=3)  # Was sigma=2
```

#### Issue: "负间距值 (Negative spacing)"
**Cause**: Teeth overlapping or touching

**Diagnosis**:
```python
# Check segmentation quality
debug_segmentation.visualize_segmentation_steps(image, mask, teeth_data)
```

**Solutions**:
1. Increase erosion iterations to separate teeth
2. Verify U-Net segmentation quality
3. May indicate actual tooth overlap (clinical finding)

---

## Common Tasks

### Task 1: Process New X-ray Images

```bash
# 1. Place images in input folder
cp /path/to/xray.png input/

# 2. Run analysis
python3 tooth_cej_root_analyzer.py

# 3. Check results
ls -lh output/
cat output/summary_report.txt
```

### Task 2: Adjust Risk Thresholds

**File**: `tooth_cej_root_analyzer.py:36-37`

```python
class ToothCEJAnalyzer:
    def __init__(self):
        # Modify these values
        self.DANGER_THRESHOLD = 3.0    # Changed from 3.2
        self.WARNING_THRESHOLD = 3.8   # Changed from 4.0
```

### Task 3: Calibrate Pixel-to-MM Conversion

**Method 1: Manual Calibration**
```python
# Measure a known distance in pixels
known_distance_mm = 10.0  # e.g., a tooth is ~10mm
measured_pixels = 120

pixels_per_mm = measured_pixels / known_distance_mm
```

**Method 2: From X-ray Metadata** (future enhancement)
```python
# Extract from DICOM metadata or ruler in image
pixels_per_mm = extract_from_metadata(xray_image)
```

**Update in Code**:
```python
# tooth_cej_root_analyzer.py:40
self.pixels_per_mm = 12  # Adjusted value
```

### Task 4: Add New Visualization

**Location**: `tooth_cej_root_analyzer.py:analyze_and_visualize`

**Example: Add 5th panel**
```python
# After existing 4 subplots
fig = plt.figure(figsize=(24, 10))  # Increase width
ax5 = plt.subplot(1, 5, 5)  # Add 5th panel

# Plot custom visualization
ax5.plot(custom_data_x, custom_data_y)
ax5.set_title('Custom Analysis')
ax5.set_xlabel('X Label')
ax5.set_ylabel('Y Label')
```

### Task 5: Batch Process with Custom Parameters

```python
import glob
from tooth_cej_root_analyzer import ToothCEJAnalyzer

# Create analyzer with custom params
analyzer = ToothCEJAnalyzer()
analyzer.DANGER_THRESHOLD = 3.5
analyzer.pixels_per_mm = 12

# Process all images
for img_path in glob.glob('input/*.png'):
    print(f"Processing {img_path}...")
    analyzer.analyze_and_visualize(img_path)
```

### Task 6: Export Results to CSV

**Add this method to ToothCEJAnalyzer**:
```python
def export_to_csv(self, results, output_path):
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Tooth_Pair', 'Min_Spacing_mm',
                        'Avg_Spacing_mm', 'Risk_Level'])

        for image_name, data in results.items():
            for pair, spacing_info in data.items():
                writer.writerow([
                    image_name,
                    pair,
                    spacing_info['min'],
                    spacing_info['avg'],
                    spacing_info['risk']
                ])
```

---

## Technical Algorithms

### Algorithm 1: CEJ Line Detection

**Principle**: CEJ is where tooth width begins to decrease significantly (crown-root boundary)

**Implementation**: `tooth_cej_root_analyzer.py:246-368`

**Steps**:
```python
1. Extract tooth contour bounding box
2. For each Y-coordinate, compute contour width:
   - Find leftmost and rightmost contour points at Y
   - Width = rightmost_x - leftmost_x
3. Apply Gaussian smoothing to width curve (σ=2)
4. Compute gradient (first derivative) of smoothed width
5. Find maximum negative gradient in search range (20%-60% of tooth height)
6. Extract CEJ curve points within ±5 pixels of detected Y
7. Compute CEJ center (midpoint of curve)
8. Calculate normal vector (perpendicular to curve direction)
```

**Key Code**:
```python
# Gradient calculation
width_gradient = np.gradient(width_smooth)

# Find maximum negative gradient in search range
search_mask = (ys >= search_start * h) & (ys <= search_end * h)
negative_gradients = width_gradient.copy()
negative_gradients[~search_mask] = 0  # Mask outside search range

cej_idx = np.argmin(negative_gradients)
cej_y = ys[cej_idx] + bbox_y
```

### Algorithm 2: Alveolar Crest Boundary Detection

**Principle**: Use morphological operations to find bone ridge boundary between teeth

**Implementation**: `tooth_cej_root_analyzer.py:370-500`

**Steps**:
```python
1. Create binary mask with both teeth
2. Apply morphological dilation (expand tooth regions)
3. Find overlap region (teeth meet after expansion)
4. Extract boundary of overlap region
5. Filter boundary points within inter-tooth region
6. Sort points top-to-bottom for ordered boundary line
```

**Key Code**:
```python
# Dilate both teeth to find connection
kernel = np.ones((5, 5), np.uint8)
dilated1 = cv2.dilate(mask1, kernel, iterations=3)
dilated2 = cv2.dilate(mask2, kernel, iterations=3)

# Find overlap (boundary region)
overlap = cv2.bitwise_and(dilated1, dilated2)

# Extract boundary contour
contours = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
```

### Algorithm 3: Depth-based Spacing Measurement

**Principle**: Measure inter-root spacing along tooth axis at multiple depths

**Implementation**: `tooth_cej_root_analyzer.py:502-600`

**Steps**:
```python
1. Define measurement depths (e.g., 0 to -15mm in 0.5mm increments)
2. For each depth:
   a. Find tooth1's boundary at this depth (rightmost point)
   b. Find tooth2's boundary at this depth (leftmost point)
   c. Compute spacing = tooth2_left - tooth1_right
   d. Assign color based on spacing value
3. Aggregate results: min, max, average spacing
```

**Key Code**:
```python
for depth in np.arange(0, -15, -0.5):  # 0 to -15mm
    # Sample point along tooth1's axis at this depth
    sample1 = cej1 + normal1 * (depth * pixels_per_mm)

    # Find tooth1's right boundary at this depth
    tooth1_right = find_boundary_point(contour1, sample1, direction='right')

    # Find tooth2's left boundary
    tooth2_left = find_boundary_point(contour2, sample2, direction='left')

    # Calculate spacing
    spacing_pixels = tooth2_left[0] - tooth1_right[0]
    spacing_mm = spacing_pixels / pixels_per_mm

    # Color code
    color = get_color_for_spacing(spacing_mm)
```

### Algorithm 4: Long Axis Calculation

**Principle**: Based on minimum area rotated rectangle (from CCA_Analysis.py)

**Implementation**: `tooth_cej_root_analyzer.py:93-180`

**Steps**:
```python
1. Compute minimum area rectangle enclosing tooth contour
2. Extract rectangle's 4 vertices
3. Order vertices: top-left, top-right, bottom-right, bottom-left
4. Calculate midpoints of top and bottom edges
5. Line connecting these midpoints = tooth's long axis
6. Compute axis angle relative to vertical
```

**Key Code** (mirrors SerdarHelli's approach):
```python
# Minimum area rectangle
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect).astype(int)

# Order points
ordered = self.order_points(box)
(tl, tr, br, bl) = ordered

# Midpoints
top_mid = self.midpoint(tl, tr)
bottom_mid = self.midpoint(bl, br)

# Axis vector
axis_vector = (bottom_mid[0] - top_mid[0], bottom_mid[1] - top_mid[1])
axis_angle = np.arctan2(axis_vector[1], axis_vector[0]) * 180 / np.pi
```

---

## Future Development

### High Priority

1. **Automatic Pixel-to-MM Calibration**
   - Detect ruler in X-ray image
   - Use DICOM metadata if available
   - Estimate from standard tooth sizes

2. **Improved U-Net Model**
   - Retrain on larger dataset
   - Fine-tune on diverse X-ray qualities
   - Implement model versioning

3. **3D Visualization**
   - Interactive depth-spacing exploration
   - Rotatable 3D tooth models
   - WebGL-based viewer

4. **Clinical Report Export**
   - PDF generation with matplotlib/ReportLab
   - Include diagnosis recommendations
   - Multilingual support (Chinese/English)

### Medium Priority

5. **Web Interface**
   - Flask/FastAPI backend
   - React/Vue frontend
   - Drag-and-drop X-ray upload
   - Real-time processing status

6. **Multiple X-ray View Support**
   - Periapical radiographs
   - Bitewing radiographs
   - CBCT (Cone Beam CT) integration

7. **Database Integration**
   - Patient record management
   - Historical comparison
   - Treatment outcome tracking

8. **Advanced Risk Models**
   - Machine learning risk prediction
   - Integration with clinical history
   - Personalized thresholds

### Low Priority (Research)

9. **Automated Treatment Planning**
   - Suggest tooth movement strategies
   - Simulate treatment outcomes
   - Integration with orthodontic planning software

10. **Mobile App**
    - iOS/Android deployment
    - TensorFlow Lite model
    - Offline processing capability

---

## Dependencies

### Python Version
- **Required**: Python 3.7+
- **Recommended**: Python 3.9+

### Core Dependencies
```
numpy>=1.19.0          # Numerical operations
opencv-python>=4.5.0   # Computer vision
matplotlib>=3.3.0      # Visualization
scipy>=1.5.0           # Scientific computing
tensorflow>=2.8.0      # Deep learning (U-Net)
Pillow>=8.0.0          # Image I/O
```

### System Dependencies (Linux)
```bash
# Chinese fonts (for visualization)
apt-get install fonts-wqy-zenhei fonts-wqy-microhei

# OpenCV dependencies
apt-get install libgl1-mesa-glx libglib2.0-0
```

### Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system fonts (if needed)
sudo apt-get update
sudo apt-get install fonts-wqy-zenhei fonts-wqy-microhei
```

---

## Environment Setup

### For Development

```bash
# 1. Clone repository
git clone https://github.com/wrs-code/Tooth_root_distance_measurement.git
cd Tooth_root_distance_measurement

# 2. Create feature branch
git checkout -b claude/<feature-name>-<session-id>

# 3. Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Install system dependencies (Linux/Ubuntu)
sudo apt-get install fonts-wqy-zenhei fonts-wqy-microhei

# 5. Verify setup
python3 test_fixes.py
python3 tooth_cej_root_analyzer.py

# 6. Check outputs
ls output/
```

### For Production/Deployment

```bash
# Use Docker (recommended)
# TODO: Add Dockerfile in future PR

# Or install system-wide
pip install -r requirements.txt
# Configure as systemd service or cron job
```

---

## Additional Resources

### Documentation Files
- `README.md` - Project overview (Chinese)
- `CEJ_ANALYSIS_README.md` - Technical guide (Chinese)
- `牙根距离测量方法说明.md` - Detailed method explanation (Chinese)
- `FIXES_APPLIED.md` - Recent bug fixes and patches
- `TEETH_SEGMENTATION_FIX.md` - Segmentation optimization history
- `口腔正畸牙根间距自动测量与风险标注软件技术需求文档.pdf` - Requirements document

### External References
- **U-Net Source Repository**: [SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net)
- **U-Net Paper**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **Dental Dataset**: H. Abdi et al., "Automatic segmentation of mandible in panoramic x-ray"

### Getting Help
- **GitHub Issues**: Report bugs or request features
- **Code Comments**: Detailed inline documentation
- **Git History**: Check commit messages for context

---

## Changelog

### 2025-11-20
- **Major Rewrite**: Morphology-based root spacing measurement system
- **CEJ Detection**: Implemented gradient-based CEJ line detection
- **Visualization**: Added 4-panel analysis reports
- **Fixes**: Chinese font display, U-Net post-processing optimization
- **Documentation**: Created this CLAUDE.md guide

### Previous
See git history for detailed commit log:
```bash
git log --oneline --decorate --graph
```

---

## Contact & Contribution

### Repository
- **GitHub**: [wrs-code/Tooth_root_distance_measurement](https://github.com/wrs-code/Tooth_root_distance_measurement)
- **Issues**: Use GitHub Issues for bug reports and feature requests

### Contribution Guidelines
1. Fork repository
2. Create feature branch following naming convention
3. Make changes with clear commit messages
4. Add tests for new features
5. Update documentation (including this file)
6. Submit Pull Request with detailed description

### License
TBD (See LICENSE file once added)

---

**End of CLAUDE.md**

*This document is maintained by AI assistants working on the project. Keep it updated as the codebase evolves.*
