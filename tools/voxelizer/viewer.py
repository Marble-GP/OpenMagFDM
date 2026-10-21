"""Interactive VTK viewer for MagVoxelizer's material-labelled cell grid.

Opacity is a display property only. Neither model.vti nor the manifest is modified.
"""
import argparse
import colorsys
import json
import math
from pathlib import Path
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def opacity(text):
    value = float(text)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("opacity must be between 0 (transparent) and 1 (opaque)")
    return value


def material_opacity(text):
    try:
        label, value = text.split("=", 1)
        label = int(label)
        if not 1 <= label <= 4294967295:
            raise ValueError()
        return label, opacity(value)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("use MATERIAL_ID=OPACITY, e.g. 1=0.3") from exc


def read_grid(directory, max_cells):
    """Read the actual cell IDs, checking the completed conversion's manifest."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "openmagfdm.voxel-grid/1":
        raise ValueError("unsupported or incomplete voxel grid manifest")
    if not 0 < manifest["cell_count"] <= max_cells:
        raise ValueError(f"viewer cell limit exceeded ({max_cells}); use --max-cells explicitly")
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(directory / "model.vti"))
    reader.Update()
    data = reader.GetOutput()
    if reader.GetErrorCode() or data.GetNumberOfCells() != manifest["cell_count"]:
        raise ValueError("VTI read failed or cell count differs from manifest")
    if list(data.GetDimensions()) != [n + 1 for n in manifest["dimensions_cells"]]:
        raise ValueError("grid dimensions differ from manifest")
    if list(data.GetOrigin()) != manifest["origin_m"] or list(data.GetSpacing()) != manifest["spacing_m"]:
        raise ValueError("grid coordinates differ from manifest")
    materials = data.GetCellData().GetArray("material_id")
    solids = data.GetCellData().GetArray("solid_id")
    for array in (materials, solids):
        if (array is None or array.GetNumberOfComponents() != 1 or
                array.GetDataType() != vtk.VTK_UNSIGNED_INT or
                array.GetNumberOfTuples() != data.GetNumberOfCells()):
            raise ValueError("VTI must contain UInt32 material_id and solid_id cell arrays")
    ids, quantities = np.unique(vtk_to_numpy(materials), return_counts=True)
    counts = dict(zip(map(int, ids), map(int, quantities)))
    expected = {0: manifest["air_cells"]}
    for body in manifest["solids"]:
        label = body["material_id"]
        expected[label] = expected.get(label, 0) + body["voxel_cells"]
    if counts != {label: count for label, count in expected.items() if count}:
        raise ValueError("material cell counts differ from manifest")
    return data, manifest, counts


def extract_material(data, label):
    """Extract the boundary of actual labelled cells, including enclosed cavities.

    No artificial box or filled gaps are substituted for the air cell region.
    The temporary unstructured cell grid is released after extracting its surface.
    """
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(data)
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "material_id")
    threshold.SetLowerThreshold(label)
    threshold.SetUpperThreshold(label)
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    threshold.Update()
    count = threshold.GetOutput().GetNumberOfCells()
    boundary = vtk.vtkDataSetSurfaceFilter()
    boundary.SetInputConnection(threshold.GetOutputPort())
    boundary.Update()
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(boundary.GetOutput())
    return mesh, count


def color_for(label, index):
    if label == 0:
        return (.35, .72, .90)
    palette = [(.36, .52, .67), (.72, .76, .81), (.94, .43, .22)]
    if index < len(palette):
        return palette[index]
    return colorsys.hsv_to_rgb((index * .618034) % 1, .55, .8)


class VoxelViewer:
    def __init__(self, data, counts, air_opacity=.06, solid_opacity=1.,
                 overrides=None, offscreen=False, max_faces=2000000):
        self.data = data
        self.counts = counts
        self.solid_ids = sorted(label for label in counts if label)
        self.selected_index = 0
        self.actors = {}
        self.meshes = {}
        self.widgets = []
        self.edges = True
        self.renderer = vtk.vtkRenderer()
        # Reserve the bottom of the window for controls; model does not sit behind sliders.
        self.renderer.SetViewport(0, .24, 1, 1)
        self.renderer.SetBackground(.96, .97, .99)
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(100)
        self.renderer.SetOcclusionRatio(0.)
        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName("OpenMagFDM - voxel materials and air")
        self.window.SetSize(1150, 950)
        self.window.SetAlphaBitPlanes(1)
        self.window.SetMultiSamples(0)
        self.window.SetOffScreenRendering(offscreen)
        self.window.AddRenderer(self.renderer)
        self.controls = vtk.vtkRenderer()
        self.controls.SetViewport(0, 0, 1, .24)
        self.controls.SetBackground(.92, .94, .97)
        self.window.AddRenderer(self.controls)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.style.SetDefaultRenderer(self.renderer)
        self.interactor.SetInteractorStyle(self.style)
        overrides = overrides or {}
        missing = set(overrides) - set(self.solid_ids)
        if missing:
            raise ValueError(f"opacity specified for absent material IDs: {sorted(missing)}")
        # Actor color is independent of the numeric magnitude of a sparse material ID.
        face_count = 0
        for label in sorted(counts):
            mesh, actual_count = extract_material(data, label)
            if actual_count != counts[label]:
                raise ValueError(f"material {label}: extraction cell count differs")
            face_count += mesh.GetNumberOfCells()
            if face_count > max_faces:
                raise ValueError("viewer surface limit exceeded; use --max-faces explicitly")
            self.meshes[label] = mesh
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            mapper.ScalarVisibilityOff()
            # Air/solid share interfaces: displace air polygon depth slightly for display.
            # This never changes coordinates or cell labels in the input data.
            if label == 0:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1., 1.)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            index = self.solid_ids.index(label) if label else 0
            actor.GetProperty().SetColor(*color_for(label, index))
            actor.GetProperty().SetEdgeColor(.16, .23, .30)
            actor.GetProperty().SetLineWidth(.5)
            actor.GetProperty().SetEdgeVisibility(label != 0)
            actor.GetProperty().SetOpacity(
                air_opacity if label == 0 else overrides.get(label, solid_opacity))
            self.actors[label] = actor
            self.renderer.AddActor(actor)
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(data)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        self.outline = vtk.vtkActor()
        self.outline.SetMapper(mapper)
        self.outline.GetProperty().SetColor(.28, .49, .65)
        self.renderer.AddActor(self.outline)
        self.status = vtk.vtkTextActor()
        self.status.GetTextProperty().SetColor(.12, .18, .25)
        self.status.GetTextProperty().SetFontSize(18)
        self.status.SetDisplayPosition(28, 12)
        self.controls.AddViewProp(self.status)
        self.air_slider = self.add_slider("Air opacity (ID 0)", air_opacity, .10, .43, self.air_changed)
        self.material_slider = self.add_slider("Material opacity", solid_opacity, .59, .92,
                                               self.material_changed)
        self.interactor.AddObserver("KeyPressEvent", self.key_press)
        self.refresh_selection()
        bounds = data.GetBounds()
        center = [(bounds[2*a] + bounds[2*a+1]) * .5 for a in range(3)]
        scale = max(bounds[2*a+1] - bounds[2*a] for a in range(3))
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(center[0]+scale, center[1]-1.4*scale, center[2]+1.5*scale)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()

    def add_slider(self, title, value, left, right, callback):
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(0)
        rep.SetMaximumValue(1)
        rep.SetValue(value)
        rep.SetTitleText(title)
        rep.SetLabelFormat("%.2f")
        rep.SetTitleHeight(.095)
        rep.SetLabelHeight(.08)
        for point, x in ((rep.GetPoint1Coordinate(), left), (rep.GetPoint2Coordinate(), right)):
            point.SetCoordinateSystemToNormalizedViewport()
            point.SetValue(x, .70)
        rep.GetTitleProperty().SetColor(.12, .18, .25)
        rep.GetLabelProperty().SetColor(.12, .18, .25)
        rep.GetSliderProperty().SetColor(.20, .45, .68)
        rep.GetTubeProperty().SetColor(.65, .72, .80)
        widget = vtk.vtkSliderWidget()
        widget.SetInteractor(self.interactor)
        widget.SetCurrentRenderer(self.controls)
        widget.SetRepresentation(rep)
        widget.SetAnimationModeToAnimate()
        widget.AddObserver("InteractionEvent", callback)
        self.widgets.append(widget)
        return widget

    def set_opacity(self, label, value):
        value = opacity(value)
        if label in self.actors:
            self.actors[label].GetProperty().SetOpacity(value)

    def air_changed(self, widget, _event):
        self.set_opacity(0, widget.GetRepresentation().GetValue())
        self.window.Render()

    def material_changed(self, widget, _event):
        if self.solid_ids:
            self.set_opacity(self.solid_ids[self.selected_index], widget.GetRepresentation().GetValue())
        self.window.Render()

    def select_material(self, delta):
        if self.solid_ids:
            self.selected_index = (self.selected_index + delta) % len(self.solid_ids)
            self.refresh_selection()

    def refresh_selection(self):
        rep = self.material_slider.GetRepresentation()
        if self.solid_ids:
            label = self.solid_ids[self.selected_index]
            rep.SetTitleText(f"Material {label} opacity  [Left / Right]")
            rep.SetValue(self.actors[label].GetProperty().GetOpacity())
        else:
            rep.SetTitleText("No solid material")
        self.status.SetInput(
            f"Air: {self.counts.get(0, 0):,} cells | Solids: "
            f"{sum(v for k, v in self.counts.items() if k):,} cells | "
            "Opacity 0 = transparent, 1 = opaque\n"
            "Left / Right: select material   E: mesh edges   R: reset camera\n"
            "Air is stored in model.vti. Display settings do not change the grid.")

    def key_press(self, obj, _event):
        key = obj.GetKeySym()
        if key in ("Left", "Right"):
            self.select_material(-1 if key == "Left" else 1)
        elif key.lower() == "e":
            self.edges = not self.edges
            for label, actor in self.actors.items():
                actor.GetProperty().SetEdgeVisibility(self.edges and label != 0)
        elif key.lower() == "r":
            self.renderer.ResetCamera()
        self.window.Render()

    def render(self):
        self.interactor.Initialize()
        for widget in self.widgets:
            widget.EnabledOn()
        self.window.Render()

    def screenshot(self, path):
        if path.exists():
            raise ValueError("screenshot already exists; choose a new path")
        if not path.parent.is_dir():
            raise ValueError("screenshot parent directory does not exist")
        self.render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.window)
        capture.ReadFrontBufferOff()
        capture.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        if writer.GetErrorCode() or not path.is_file():
            raise OSError("failed to write screenshot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="completed voxel output directory")
    parser.add_argument("--air-opacity", type=opacity, default=.06)
    parser.add_argument("--solid-opacity", type=opacity, default=1.)
    parser.add_argument("--material-opacity", type=material_opacity, action="append", default=[])
    parser.add_argument("--screenshot", type=Path, help="render to a new PNG instead of opening a window")
    parser.add_argument("--max-cells", type=int, default=2000000)
    parser.add_argument("--max-faces", type=int, default=2000000)
    args = parser.parse_args()
    viewer = None
    try:
        data, manifest, counts = read_grid(args.directory, args.max_cells)
        if len(counts) > 64:
            raise ValueError("viewer supports up to 64 material regions")
        viewer = VoxelViewer(data, counts, args.air_opacity, args.solid_opacity,
                             dict(args.material_opacity), bool(args.screenshot), args.max_faces)
        if args.screenshot:
            viewer.screenshot(args.screenshot)
            print(f"Saved {args.screenshot}")
            print(f"Depth peeling used: {bool(viewer.renderer.GetLastRenderingUsedDepthPeeling())}")
        else:
            viewer.render()
            viewer.interactor.Start()
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Viewer error: {exc}", file=sys.stderr)
        return 1
    finally:
        if viewer is not None:
            viewer.window.Finalize()


if __name__ == "__main__":
    sys.exit(main())
