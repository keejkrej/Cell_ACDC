# PyAMA-Core to Cell-ACDC Plugin Migration Plan

## Table of Contents

1. [What PyAMA-Core Does](#1-what-pyama-core-does)
2. [What Cell-ACDC Does](#2-what-cell-acdc-does)
3. [What to Take from PyAMA-Core](#3-what-to-take-from-pyama-core)
4. [What to Take from Cell-ACDC](#4-what-to-take-from-cell-acdc)
5. [Easiest First Step as Proof of Concept](#5-easiest-first-step-as-proof-of-concept)
6. [Optimal User Experience](#6-optimal-user-experience)
7. [What to Add Later On](#7-what-to-add-later-on)

---

## 1. What PyAMA-Core Does

### 1.1 Core Purpose

PyAMA-Core processes time-lapse microscopy images through a **5-step sequential workflow** to extract quantitative cell traces from phase contrast and fluorescence microscopy data.

### 1.2 Workflow Steps

**Step 1: Copying**

- Extracts frames from ND2/CZI microscopy files
- Saves as memory-mapped NumPy arrays (`.npy` files)
- Format: `(T, H, W)` uint16 arrays per channel
- Output: `{basename}_fov_{fov:03d}_{pc|fl}_ch_{channel_id}.npy`

**Step 2: Segmentation**

- Uses **LOG-STD thresholding** method for phase contrast images
- Computes local log standard deviation in sliding windows
- Automatic threshold selection from histogram
- Morphological cleanup (opening, closing, hole filling)
- Output: Binary masks `(T, H, W)` bool - `{basename}_fov_{fov:03d}_seg_ch_{pc_id}.npy`

**Step 3: Correction**

- **Tiled interpolation background correction** for fluorescence channels
- Divides frame into overlapping tiles
- Computes tile medians from background pixels (excluding dilated segmentation mask)
- Uses `scipy.interpolate.RectBivariateSpline` for bicubic interpolation
- Subtracts interpolated background surface from original frames
- Output: Background-corrected FL stacks `(T, H, W)` float32 - `{basename}_fov_{fov:03d}_fl_corrected_ch_{fl_id}.npy`

**Step 4: Tracking**

- **IoU-based Hungarian assignment** tracking
- Extracts connected components from binary segmentation per frame
- Computes Intersection over Union (IoU) between consecutive frames
- Uses Hungarian algorithm for optimal assignment
- Assigns consistent cell IDs across frames
- Output: Labeled tracking stack `(T, H, W)` uint16 - `{basename}_fov_{fov:03d}_seg_labeled_ch_{pc_id}.npy`

**Step 5: Extraction**

- Extracts quantitative features per cell per time point
- Combines morphological features (from PC) and intensity features (from FL channels)
- Filters traces (short traces, border cells)
- Generates CSV with temporal traces
- Output: `{basename}_fov_{fov:03d}_traces.csv`

### 1.3 Key Characteristics

- **File-based data flow**: Each step saves to disk, next step loads from disk
- **Memory-mapped arrays**: Efficient for large time-lapse datasets
- **Batch processing**: Processes multiple FOVs in batches
- **Channel separation**: PC channel for segmentation, FL channels for correction/extraction
- **CLI-based**: Configured via Python dictionaries/YAML, executed via Python scripts

### 1.4 Output Structure

```
output_dir/
├── processing_results.yaml
├── fov_000/
│   ├── {basename}_fov_000_pc_ch_0.npy
│   ├── {basename}_fov_000_fl_ch_1.npy
│   ├── {basename}_fov_000_seg_ch_0.npy
│   ├── {basename}_fov_000_seg_labeled_ch_0.npy
│   ├── {basename}_fov_000_fl_corrected_ch_1.npy
│   └── {basename}_fov_000_traces.csv
└── fov_001/...
```

---

## 2. What Cell-ACDC Does

### 2.1 Core Purpose

Cell-ACDC is a **GUI-first, CLI-second** application for cell segmentation, tracking, and quantitative analysis of time-lapse microscopy data. It provides an extensive ecosystem of models, trackers, and preprocessing tools.

### 2.2 Architecture

**Entry Point:**

- GUI mode (default): `cellacdc` → Opens main launcher window
- CLI mode: `cellacdc -p config.ini` → Processes workflow automatically

**Module-Based Structure:**

- **Module 0**: Data structure creation (converts ND2/CZI to structured folders)
- **Module 1**: Data prep (image preprocessing: align, filter, crop, etc.)
- **Module 2**: Segmentation module (runs segmentation/tracking workflows)
- **Module 3**: Main GUI (visualization, annotation, measurements)
- **Module 4**: SpotMAX (if installed) - Spot tracking

### 2.3 Key Features

**Segmentation Models:**

- Cellpose (v2, v3, v4)
- Omnipose
- YeaZ
- StarDist
- BABY
- DeepSea
- InstanSeg
- Segment Anything
- And many more...

**Tracking Algorithms:**

- CellACDC tracker (IoU-based)
- BABY tracker
- BayesianTracker
- Arboretum
- And more...

**Preprocessing:**

- Gaussian filtering
- Hot pixel removal
- Normalization
- Image alignment
- ROI selection
- Custom recipe builder

**Data Flow System:**

- File-based persistence (matches PyAMA approach)
- Uses `loadData` (posData) structure for encapsulation
- Standard naming convention: `{basename}_{suffix}.{ext}`
- Supports multiple segmentation variants

**Measurements System:**

- `cellacdc.features.get_acdc_df_features()` for feature extraction
- Measurements table: `{basename}_acdc_output.csv`
- MultiIndex DataFrame: `(frame_i, Cell_ID)`
- Extensive feature library (morphological, intensity, etc.)

### 2.4 Data Structure

**Cell-ACDC File Convention:**

```
Experiment/
└── Position_n/
    └── Images/
        ├── {basename}_channel1.tif
        ├── {basename}_channel2.tif
        ├── {basename}_segm.npz              # Segmentation masks
        ├── {basename}_acdc_output.csv       # Measurements table
        ├── {basename}_metadata.csv          # Image metadata
        └── {basename}_segmInfo.csv          # Segmentation info
```

**Core Data Object: `loadData` (posData)**

- Encapsulates one position's data
- Stores paths to all relevant files
- Provides methods to load/save data
- Contains in-memory data (images, segmentation, measurements)

### 2.5 User Interaction Model

**GUI Mode (Primary):**

- Main launcher window with module buttons
- Each module opens as separate window/dialog
- Interactive dialogs for configuration (Qt dialogs)
- Progress bars and status messages
- File selection via `QFileDialog`

**CLI Mode (Secondary):**

- INI configuration files
- Workflow types: 'segmentation and/or tracking', 'measurements'
- Progress bars using `tqdm`
- Log file output
- Headless execution

---

## 3. What to Take from PyAMA-Core

### 3.1 Core Workflow Structure

✅ **Keep the 5-step sequential workflow:**

1. Copying → Segmentation → Correction → Tracking → Extraction
2. This proven structure matches the scientific workflow

### 3.2 Unique Algorithms

✅ **Keep PyAMA's specialized algorithms:**

**Step 3: Tiled Interpolation Background Correction**

- PyAMA's tiled interpolation method is unique and effective
- Handles spatially varying background well
- Uses `scipy.interpolate.RectBivariateSpline` for smooth interpolation
- Should be implemented as-is or with minor adaptations

**Step 2: LOG-STD Segmentation (Optional)**

- Can be kept as a "thresholding" model option
- Useful for phase contrast images
- Can coexist with Cell-ACDC's model library

### 3.3 File Structure and Naming

✅ **Keep PyAMA's file naming convention:**

- `{basename}_fov_{fov:03d}_{type}_ch_{channel_id}.npy`
- Maintains compatibility with existing PyAMA workflows
- Allows easy comparison and validation

### 3.4 Memory-Mapped Arrays

✅ **Keep memory-mapped array approach:**

- Efficient for large time-lapse datasets
- Use `.npy` files with `mmap_mode='r'` for reading
- Supports random access without loading entire stacks

### 3.5 Batch Processing Pattern

✅ **Keep batch processing architecture:**

- Sequential copying per batch (I/O bound)
- Parallel processing of Steps 2-5 (CPU bound)
- Worker-based parallelization
- Context merging after batch completion

### 3.6 Feature Extraction Approach

✅ **Keep PyAMA's feature extraction structure:**

- Per-channel feature extraction
- Channel suffixing: `{feature}_ch_{channel_id}`
- Trace filtering (short traces, border cells)
- CSV output format with consistent column naming

### 3.7 Channel Configuration

✅ **Keep channel separation model:**

- PC channel for segmentation (required)
- FL channels for correction/extraction (optional)
- Flexible channel selection per experiment

---

## 4. What to Take from Cell-ACDC

### 4.1 Segmentation Models

✅ **Replace PyAMA's LOG-STD with Cell-ACDC's model library:**

- Use `cellacdc.core.segm_model_segment()` function
- Support all Cell-ACDC models (Cellpose, Omnipose, YeaZ, etc.)
- Leverage `cellacdc.models.*.acdcSegment.Model.segment()` interfaces
- Allow model selection via configuration

**Integration Points:**

- `cellacdc.core.segm_model_segment()` - Core segmentation function
- `cellacdc.preprocess` - Preprocessing pipeline
- `cellacdc.models.*` - Model implementations

### 4.2 Tracking Algorithms

✅ **Replace PyAMA's IoU tracking with Cell-ACDC trackers:**

- Use `cellacdc.trackers.*.tracker.track()` interfaces
- Support CellACDC tracker, BABY, BayesianTracker, etc.
- Leverage existing tracker implementations
- Allow tracker selection via configuration

**Integration Points:**

- `cellacdc.trackers.CellACDC.CellACDC_tracker.tracker.track()`
- `cellacdc.trackers.BABY.BABY_tracker.tracker.track()`
- Other trackers as available

### 4.3 Preprocessing System

✅ **Use Cell-ACDC's preprocessing recipes:**

- Gaussian filtering, hot pixel removal, normalization
- Recipe builder system
- Configurable preprocessing steps
- Use `cellacdc.preprocess` module

### 4.4 File I/O Infrastructure

✅ **Adapt Cell-ACDC's loading system:**

- Use `cellacdc.acdc_bioio_bioformats` for ND2/CZI reading
- Leverage existing file readers
- Add memory-mapping option where needed
- Use `cellacdc.load.loadData()` as base, adapt for PyAMA structure

### 4.5 Measurements/Features System

✅ **Use Cell-ACDC's feature extraction:**

- `cellacdc.features.get_acdc_df_features()` for core features
- Extensive feature library (morphological, intensity, etc.)
- Measurement functions from `cellacdc.measurements`
- Adapt to PyAMA's per-channel feature structure

### 4.6 CLI System

✅ **Integrate with Cell-ACDC's CLI:**

- Use Cell-ACDC's `.ini` configuration format
- Create `PyamaWorkflowKernel` similar to `SegmKernel`
- Hook into `cellacdc/cli.py` workflow system
- Support `cellacdc -p config.ini` execution

### 4.7 Data Loading Pattern

✅ **Follow Cell-ACDC's data loading pattern:**

- Use `loadData` structure (adapted for PyAMA)
- Build paths using conventions
- Load files incrementally as needed
- Support resumability

### 4.8 Error Handling and Logging

✅ **Use Cell-ACDC's error handling patterns:**

- Progress reporting via signals/progress bars
- Logging system
- Error messages and recovery
- Support cancellation/resumption

### 4.9 GUI Integration (Future)

✅ **Integrate with Cell-ACDC's GUI system:**

- Use Qt dialogs for configuration
- Progress visualization widgets
- File selection dialogs
- Menu integration (Utilities → PyAMA Workflow)

---

## 5. Easiest First Step as Proof of Concept

### 5.1 Recommended POC: Steps 1-2 (Copying + Segmentation)

**Why This Combination:**

- Step 1 (Copying) is straightforward file I/O
- Step 2 (Segmentation) demonstrates Cell-ACDC integration
- Produces visible output (segmentation masks)
- Can validate against known results
- Minimal dependencies beyond Cell-ACDC

### 5.2 POC Implementation Plan

**Phase 1: Plugin Structure (1-2 days)**

```
cellacdc/plugins/pyama_workflow/
├── __init__.py
├── plugin.py              # Main entry point
├── workflow_runner.py      # Basic orchestration
└── services/
    ├── copying_service.py  # Step 1: Adapt Cell-ACDC loading
    └── segmentation_adapter.py  # Step 2: Bridge to Cell-ACDC models
```

**Phase 2: Copying Service (2-3 days)**

- Adapt `cellacdc.acdc_bioio_bioformats` for ND2/CZI reading
- Save as memory-mapped `.npy` files
- Follow PyAMA naming: `{basename}_fov_{fov:03d}_pc_ch_{pc_id}.npy`
- Support FOV selection and channel configuration

**Phase 3: Segmentation Adapter (2-3 days)**

- Create adapter that wraps `cellacdc.core.segm_model_segment()`
- Convert labeled masks to binary masks: `binary = labeled > 0`
- Save as `{basename}_fov_{fov:03d}_seg_ch_{pc_id}.npy`
- Support model selection (start with Cellpose v4)

**Phase 4: CLI Integration (1-2 days)**

- Create minimal `.ini` config format
- Register plugin in `cellacdc/cli.py`
- Add `PyamaWorkflowKernel` for Steps 1-2
- Support: `cellacdc --pyama-workflow config.ini`

**Phase 5: Testing (1-2 days)**

- Test with sample ND2/CZI file
- Validate output format matches PyAMA
- Compare segmentation results

**Total Estimated Time: 1-2 weeks**

### 5.3 Minimal Configuration File

```ini
[workflow]
type = pyama_workflow
output_dir = ./output

[paths_info]
paths = /path/to/data.nd2
fov_indices = 0

[channels]
phase_contrast_channel = 0

[segmentation]
model = cellpose_v4
diameter = 30.0
```

### 5.4 Success Criteria for POC

- ✅ Successfully extracts frames from ND2/CZI file
- ✅ Saves memory-mapped arrays in PyAMA format
- ✅ Runs Cell-ACDC segmentation model
- ✅ Produces binary segmentation masks
- ✅ Output files match PyAMA naming convention
- ✅ Can be executed via CLI with config file

### 5.5 Alternative POC: Step 3 Only (Correction Service)

**Why This Could Be Easier:**

- Standalone algorithm (no dependencies on other steps)
- Unique PyAMA contribution
- Can test with synthetic data
- Demonstrates value-add over Cell-ACDC

**Implementation:**

- Implement tiled interpolation algorithm
- Create `correction_service.py`
- Test with binary masks + fluorescence images
- Output corrected stacks

**Use Case:** Users can run Cell-ACDC segmentation, then apply PyAMA correction

---

## 6. Optimal User Experience

### 6.1 Primary Mode: CLI with Configuration File

**Target:** All users, especially those familiar with PyAMA or batch processing

**Usage:**

```bash
cellacdc -p pyama_workflow_config.ini
```

**Configuration File Format (INI):**

```ini
[workflow]
type = pyama_workflow
output_dir = /path/to/output
batch_size = 2
n_workers = 4

[paths_info]
paths = 
    /path/to/experiment1.nd2
    /path/to/experiment2.nd2
fov_indices = 
    0
    1
    2

[channels]
phase_contrast_channel = 0
fluorescence_channels = 
    1
    2

[segmentation]
model = cellpose_v4
model_type = cyto2
diameter = 30.0
flow_threshold = 0.4
cellprob_threshold = 0.0
min_size = 15
preprocessing_recipe = 
    gaussian_filter:sigma=1.0
    normalize_image

[tracking]
tracker = CellACDC
IoA_thresh = 0.4
min_size = 50
max_size = 10000

[correction]
tile_size = 75
overlap = 25
dilation_size = 5
apply_to_channels = 
    1
    2

[extraction]
features_pc = 
    area
    perimeter
    aspect_ratio
features_fl = 
    intensity_total
    intensity_mean
    intensity_max
min_trace_length = 5
border_width = 50
time_units = min
```

**Advantages:**

- ✅ Reproducible (config file = workflow definition)
- ✅ Batch processing friendly
- ✅ Works well in scripts/automation
- ✅ Easy to version control
- ✅ Matches Cell-ACDC's existing CLI pattern

### 6.2 Secondary Mode: Python API

**Target:** Advanced users, developers, custom pipelines

**Usage:**

```python
from cellacdc.plugins.pyama_workflow import PyamaWorkflow

# From config file
workflow = PyamaWorkflow.from_config('config.ini')
workflow.run()

# Or programmatic
workflow = PyamaWorkflow(
    input_files=['/path/to/data.nd2'],
    output_dir='/path/to/output',
    fov_indices=[0, 1, 2],
    pc_channel=0,
    fl_channels=[1, 2],
    segmentation_model='cellpose_v4',
    segmentation_params={'diameter': 30.0},
    tracker='CellACDC',
    tracker_params={'IoA_thresh': 0.4},
    correction_params={'tile_size': 75},
    extraction_features={
        'pc': ['area', 'perimeter'],
        'fl': ['intensity_mean', 'intensity_total']
    }
)
workflow.run()
```

**Advantages:**

- ✅ Full programmatic control
- ✅ Easy integration into custom pipelines
- ✅ Step-by-step execution possible
- ✅ Custom progress monitoring

### 6.3 Future Mode: GUI Integration

**Target:** Interactive users, beginners

**Features:**

- Wizard-style interface for configuration
- Real-time progress visualization
- Preview segmentation on sample frame
- Interactive parameter tuning
- Result visualization

**Integration Points:**

- Cell-ACDC's main GUI → Utilities → PyAMA Workflow
- Use Qt dialogs following Cell-ACDC patterns
- Progress bars and status messages

### 6.4 User Workflow Examples

**Example 1: Quick Start**

```bash
# 1. Create minimal config
cat > quick_start.ini << EOF
[workflow]
type = pyama_workflow
output_dir = ./results

[paths_info]
paths = /data/my_experiment.nd2
fov_indices = 0

[channels]
phase_contrast_channel = 0
fluorescence_channels = 1
EOF

# 2. Run
cellacdc -p quick_start.ini
```

**Example 2: Batch Processing**

```ini
[workflow]
type = pyama_workflow
output_dir = ./batch_results
batch_size = 4
n_workers = 8

[paths_info]
paths = /data/experiment.nd2
fov_indices = 0, 1, 2, 3, 4, 5, 6, 7
```

**Example 3: Custom Model**

```ini
[segmentation]
model = omnipose
model_type = bact_phase_omni
diameter = 20.0
preprocessing_recipe = 
    gaussian_filter:sigma=1.5
    remove_hot_pixels
    normalize_image
```

### 6.5 Progress and Feedback

**CLI Mode:**

- Progress bars using `tqdm` for each step
- Log file: `{output_dir}/logs/pyama_workflow_{timestamp}.log`
- Real-time status messages to console

**API Mode:**

- Optional progress callback function
- Returns status dictionaries per step
- Can check intermediate results

**GUI Mode (Future):**

- Visual progress bars
- Status messages in dialog
- Estimated time remaining
- Cancel/pause functionality

### 6.6 Error Handling

- Clear, actionable error messages
- Contextual information (parameter values)
- Suggestions for fixes
- Documentation references
- Resumption support (detect existing files, resume from last step)

### 6.7 Output Structure

```
output_dir/
├── logs/
│   └── pyama_workflow_2024-01-15_14-30-00.log
├── fov_000/
│   ├── experiment_fov_000_pc_ch_0.npy          # Raw PC stack
│   ├── experiment_fov_000_fl_ch_1.npy         # Raw FL stack
│   ├── experiment_fov_000_seg_ch_0.npy        # Binary segmentation
│   ├── experiment_fov_000_seg_labeled_ch_0.npy # Tracked labels
│   ├── experiment_fov_000_fl_corrected_ch_1.npy # Corrected FL
│   └── experiment_fov_000_traces.csv          # Feature traces
└── fov_001/...
```

---

## 7. What to Add Later On

### 7.1 Enhanced Segmentation Options

**Phase 2: Additional Models**

- Support for all Cell-ACDC models (currently support 1-2 for POC)
- Custom model loading
- Model ensemble/voting
- Model comparison tools

**Phase 3: Advanced Preprocessing**

- Full preprocessing recipe builder integration
- Custom preprocessing functions
- Preprocessing visualization

### 7.2 Advanced Tracking Features

**Phase 2: Additional Trackers**

- Support all Cell-ACDC trackers
- Tracker comparison tools
- Tracker parameter optimization

**Phase 3: Tracking Quality Metrics**

- Tracking accuracy metrics
- Visual tracking validation
- Manual correction interface

### 7.3 GUI Integration

**Phase 2: Basic GUI**

- Configuration wizard
- Progress visualization
- Result preview

**Phase 3: Full GUI**

- Interactive parameter tuning
- Live segmentation preview
- Visualization tools
- Result export interfaces

### 7.4 Advanced Background Correction

**Phase 2: Additional Methods**

- Multiple background correction algorithms
- Method comparison
- Parameter optimization

**Phase 3: Adaptive Correction**

- Automatic parameter selection
- Quality metrics for correction
- Correction validation tools

### 7.5 Enhanced Feature Extraction

**Phase 2: Extended Features**

- All Cell-ACDC features
- Custom feature definitions
- Feature selection tools

**Phase 3: Advanced Analysis**

- Trace analysis tools
- Statistical analysis
- Visualization of traces
- Export to analysis formats (HDF5, etc.)

### 7.6 Performance Optimizations

**Phase 2: Parallelization**

- GPU acceleration for segmentation
- Optimized batch processing
- Memory management improvements

**Phase 3: Distributed Processing**

- Cloud processing support
- Cluster computing integration
- Job queue system

### 7.7 Data Format Support

**Phase 2: Additional Formats**

- Support more microscopy formats
- Export to other formats
- Format conversion tools

**Phase 3: Data Integration**

- Integration with other analysis tools
- Database support
- Metadata management

### 7.8 Workflow Extensions

**Phase 2: Additional Steps**

- Pre-processing steps (alignment, etc.)
- Post-processing steps (filtering, etc.)
- Quality control steps

**Phase 3: Workflow Builder**

- Visual workflow builder
- Custom step definitions
- Workflow templates

### 7.9 Validation and Testing

**Phase 2: Comparison Tools**

- Compare with original PyAMA outputs
- Validation against ground truth
- Performance benchmarking

**Phase 3: Quality Assurance**

- Automated testing suite
- Validation datasets
- Continuous integration

### 7.10 Documentation and Examples

**Phase 2: User Documentation**

- Complete user guide
- Configuration examples
- Troubleshooting guide

**Phase 3: Advanced Documentation**

- API documentation
- Developer guide
- Tutorial videos
- Example datasets

### 7.11 Integration with Cell-ACDC Ecosystem

**Phase 2: Native Integration**

- Menu integration in Cell-ACDC GUI
- Shared configuration system
- Unified logging

**Phase 3: Deep Integration**

- Shared data structures
- Cross-plugin workflows
- Unified visualization

---

## Implementation Phases Summary

### Phase 1: Foundation (Weeks 1-2) - POC

- [x] Plugin directory structure
- [x] Copying service (adapt Cell-ACDC loading)
- [x] Segmentation adapter (Cell-ACDC models)
- [x] Basic CLI integration
- [x] Minimal configuration support

### Phase 2: Core Services (Weeks 3-4)

- [ ] Tracking adapter (Cell-ACDC trackers)
- [ ] Correction service (PyAMA tiled interpolation)
- [ ] Extraction service (Cell-ACDC features)
- [ ] Full workflow runner
- [ ] Batch processing support

### Phase 3: Integration (Weeks 5-6)

- [ ] Configuration management
- [ ] Error handling and logging
- [ ] Progress reporting
- [ ] Resumption support
- [ ] Testing and validation

### Phase 4: CLI & Polish (Weeks 7-8)

- [ ] Complete CLI integration
- [ ] Example configurations
- [ ] Documentation
- [ ] Performance optimization

### Phase 5: GUI & Advanced Features (Weeks 9+)

- [ ] GUI integration
- [ ] Additional models/trackers
- [ ] Advanced features
- [ ] Workflow extensions

---

## Key Design Principles

1. **File-Based Persistence**: Each step saves to disk, enabling resumability
2. **Convention Over Configuration**: Standard naming and structure
3. **Modularity**: Steps can be run independently
4. **Compatibility**: Maintain PyAMA file structure for validation
5. **Extensibility**: Leverage Cell-ACDC's ecosystem
6. **User Choice**: Support multiple interaction modes (CLI, API, GUI)

---

## Success Criteria

- ✅ Plugin successfully processes ND2/CZI files through all 5 steps
- ✅ Output CSV matches PyAMA format (or provides conversion)
- ✅ Works with at least 3 different Cell-ACDC segmentation models
- ✅ Works with at least 2 different Cell-ACDC trackers
- ✅ Handles batch processing of multiple FOVs
- ✅ Provides clear error messages and logging
- ✅ Performance is comparable or better than original PyAMA
- ✅ Seamless integration with Cell-ACDC ecosystem

---

## Notes

- The plugin architecture allows gradual migration from PyAMA to Cell-ACDC
- Users can choose which components to use (Cell-ACDC models vs PyAMA algorithms)
- The plugin can serve as a template for other workflow integrations
- Maintaining PyAMA's file structure allows easy comparison and validation
- The design balances PyAMA's proven workflow with Cell-ACDC's extensive capabilities
