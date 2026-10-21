"""Optional integration with an actual VTK reader and offscreen rendering.

python vtk_reader_smoke.py NEW_OUTPUT_DIRECTORY [preview.png]
Requires vtk (the C++ voxelizer itself does not).
"""
import json
from pathlib import Path
import sys
import vtk

directory = Path(sys.argv[1])
manifest = json.loads((directory / "manifest.json").read_text())
volume = vtk.vtkXMLImageDataReader()
volume.SetFileName(str(directory / "model.vti"))
volume.Update()
data = volume.GetOutput()
assert volume.GetErrorCode() == 0
assert data.GetNumberOfCells() == manifest["cell_count"]
assert list(data.GetDimensions()) == [n + 1 for n in manifest["dimensions_cells"]]
assert list(data.GetOrigin()) == manifest["origin_m"]
assert list(data.GetSpacing()) == manifest["spacing_m"]
material = data.GetCellData().GetArray("material_id")
solid = data.GetCellData().GetArray("solid_id")
counts = [0] * len(manifest["solids"])
for i in range(data.GetNumberOfCells()):
    owner = int(solid.GetValue(i))
    if owner:
        counts[owner - 1] += 1
        assert int(material.GetValue(i)) == manifest["solids"][owner - 1]["material_id"]
    else:
        assert material.GetValue(i) == 0
assert counts == [s["voxel_cells"] for s in manifest["solids"]]

surface = vtk.vtkXMLPolyDataReader()
surface.SetFileName(str(directory / "surface.vtp"))
surface.Update()
mesh = surface.GetOutput()
assert surface.GetErrorCode() == 0
assert mesh.GetNumberOfPolys() == manifest["surface_faces"]
assert mesh.GetNumberOfPoints() == 4 * manifest["surface_faces"]
for i in range(mesh.GetNumberOfCells()):
    assert mesh.GetCellType(i) == vtk.VTK_QUAD
assert mesh.GetCellData().GetArray("material_id").GetNumberOfTuples() == mesh.GetNumberOfPolys()
print(f"VTK {vtk.vtkVersion.GetVTKVersion()}: {data.GetNumberOfCells()} volume cells, "
      f"{mesh.GetNumberOfPolys()} exterior quads, IDs and units verified")

if len(sys.argv) > 2:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(4)
    lut.SetRange(0, 3)
    lut.SetTableValue(0, .9, .92, .95, 1)
    lut.SetTableValue(1, .36, .52, .67, 1)
    lut.SetTableValue(2, .72, .76, .81, 1)
    lut.SetTableValue(3, .94, .43, .22, 1)
    lut.Build()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(surface.GetOutputPort())
    mapper.SetScalarModeToUseCellFieldData()
    mapper.SelectColorArray("material_id")
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(0, 3)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(.15, .23, .31)
    actor.GetProperty().SetLineWidth(.4)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(.96, .97, .99)
    renderer.AddActor(actor)
    label = vtk.vtkTextActor()
    pitch = " x ".join(f"{s * 1000:g}" for s in manifest["spacing_m"])
    materials = len({s["material_id"] for s in manifest["solids"]})
    label.SetInput(f"OpenMagFDM | STEP -> voxels\n{pitch} mm | "
                   f"{len(manifest['solids'])} bodies | {materials} material labels")
    label.GetTextProperty().SetFontSize(22)
    label.GetTextProperty().SetColor(.12, .18, .25)
    label.SetDisplayPosition(30, 30)
    renderer.AddViewProp(label)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(.040, -.055, .060)
    camera.SetFocalPoint(0, 0, .003)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCamera()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(1100, 850)
    window.AddRenderer(renderer)
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(sys.argv[2])
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    window.Finalize()
    print(f"Preview saved: {sys.argv[2]}")
