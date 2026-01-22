# OpenMagFDM v1.2.0 Release Notes

## Overview

v1.2.0 focuses on WebUI enhancements, improving the user experience for analysis visualization and result management.

## New Features

### 1. Dashboard Plot Configuration
- **Plot Configure Dialog**: Configure axis ranges (x, y, z) and colorscale for each plot independently
- **Auto/Fixed Range Modes**: Switch between automatic scaling and fixed ranges
- **Colorscale Selection**: Choose from multiple color schemes (Viridis, Jet, Hot, etc.)
- **Persistent Settings**: Plot configurations are maintained across visualization updates

### 2. File Manager Enhancements
- **Result Preview Panel**:
  - Two-panel layout with folder list and preview
  - Preview displays: Input Image + Calculated B magnitude (not stored Az data)
  - Automatic B field calculation using rotation operator: B = ∇×(Az/μ)
- **Description/Memo System**:
  - Add descriptions to analysis result folders
  - Descriptions shown in dropdown lists with truncation ("..." for long text)
  - Gray color (#6c757d) and smaller font (11px) to distinguish from folder names
  - Full description visible on hover
- **Result Management**:
  - Bulk delete support with checkbox selection
  - Rename result folders
  - Preview updates correctly on folder selection
  - Description editing in preview panel

### 3. Coordinate System Improvements
- **Polar Coordinate Support**: Properly handle dr/dtheta for polar coordinates in B calculation
- **Image Orientation**: Correct vertical flipping for visualization (analysis y-up → image y-down)

## Bug Fixes

- Fixed Plot Configure showing "null" for plot titles
- Fixed plot range settings not being applied after configuration
- Fixed File Manager preview not updating on repeated folder clicks
- Fixed image display in preview area (switched from Plotly to native `<img>` tag)
- Fixed container ID restoration in plotHeatmapInDiv to prevent element not found errors
- Fixed dx/dy undefined error for polar coordinates

## Technical Improvements

- **Custom Select Component**: Implemented styled dropdown for Run & Preview results list
- **Data Attribute Management**: Proper setting of `data-plot-id` and `data-plot-type` on GridStack widgets
- **Parallel API Calls**: Fetch descriptions for all results in parallel for better performance
- **Plot Config Application**: Correctly apply xRange, yRange, zRange, and colorscale to all plot types
- **Magnetic Field Calculation**: On-the-fly B magnitude calculation from Az and Mu data

## API Endpoints

Added/enhanced endpoints for result management:
- `GET /api/user-outputs/:folderName/description` - Get result folder description
- `PUT /api/user-outputs/:folderName/description` - Update result folder description
- `DELETE /api/user-outputs/:folderName` - Delete result folder
- `PUT /api/user-outputs/:folderName/rename` - Rename result folder
- `DELETE /api/user-outputs/bulk` - Bulk delete result folders

## Commits Since v1.1.2

```
42cb28a Add description/memo display to Run & Preview result dropdown
e5aa1f8 Fix Plot Configure settings not being applied to plots
b4cd25d Fix Plot Configure missing plot type attributes
45329c3 Fix File Manager preview update on repeated clicks
14cf4d4 Fix File Manager preview image display and folder selection
461b0ba Fix File Manager preview and Plot Configure title issues
57457d9 Improve File Manager preview error handling
d4ea8c2 Fix File Manager preview error and expand colorscale options
1c62fa8 Feature 2: Add Plot Configure functionality to Dashboard
2a52230 Feature 3: Add harmonic mean interpolation for permeability
```

## Breaking Changes

None. This release is fully backward compatible with v1.1.x.

## Known Issues

None currently identified.

## Installation

Download the appropriate package for your platform:
- **Windows**: `OpenMagFDM-Installer-Windows-x86_64.exe` (recommended) or `OpenMagFDM-Windows-x86_64.zip`
- **Linux**: `OpenMagFDM-Linux-x86_64.tar.gz`
- **macOS**: `OpenMagFDM-macOS-x86_64.tar.gz`
- **WebUI Standalone**: Available separately for each platform

## Next Release

v1.3.0 will focus on core solver enhancements:
1. Robin Boundary Conditions
2. Anti-aliasing Interpolation
3. Flux Linkage Calculation
4. Material Presets
5. Adaptive Mesh Coarsening

See [CLAUDE.md](../CLAUDE.md) for detailed implementation plan.

---

**Full Changelog**: https://github.com/Marble-GP/OpenMagFDM/compare/v1.1.2...v1.2.0
