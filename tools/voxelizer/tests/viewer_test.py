"""Viewer regression: air is real cell data, and opacity never edits the grid."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viewer import VoxelViewer, extract_material, material_opacity, opacity, read_grid


def fixture(path, labels, size):
    data = vtk.vtkImageData()
    data.SetDimensions(*[n + 1 for n in size])
    data.SetOrigin(-.01, 0, .02)
    data.SetSpacing(.001, .002, .003)
    labels = np.array(labels, dtype=np.uint32)
    for name in ("material_id", "solid_id"):
        array = numpy_to_vtk(labels, deep=True)
        array.SetName(name)
        data.GetCellData().AddArray(array)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path / "model.vti"))
    writer.SetInputData(data)
    writer.Write()
    ids, counts = np.unique(labels, return_counts=True)
    manifest = {
        "schema": "openmagfdm.voxel-grid/1",
        "cell_count": len(labels),
        "air_cells": int(np.sum(labels == 0)),
        "dimensions_cells": size,
        "origin_m": list(data.GetOrigin()),
        "spacing_m": list(data.GetSpacing()),
        "solids": [{"material_id": int(i), "voxel_cells": int(n)}
                   for i, n in zip(ids, counts) if i],
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return data


class ViewerTest(unittest.TestCase):
    def test_air_cavity_and_opacity_controls_preserve_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            labels = [0] * 27
            labels[13] = 1  # isolated material cube surrounded by 26 air cells
            fixture(path, labels, [3, 3, 3])
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
            data, _, counts = read_grid(path, 100)
            self.assertEqual(counts, {0: 26, 1: 1})
            air, air_count = extract_material(data, 0)
            self.assertEqual(air_count, 26)
            # Outer 3x3 faces plus six inner cavity faces; not just a bounding box.
            self.assertEqual(air.GetNumberOfCells(), 6 * 9 + 6)
            solid, solid_count = extract_material(data, 1)
            self.assertEqual(solid_count, 1)
            self.assertEqual(solid.GetNumberOfCells(), 6)
            viewer = VoxelViewer(data, counts, offscreen=True)
            try:
                viewer.render()
                for value in (0., .2, 1.):
                    viewer.air_slider.GetRepresentation().SetValue(value)
                    viewer.air_slider.InvokeEvent("InteractionEvent")
                    self.assertAlmostEqual(viewer.actors[0].GetProperty().GetOpacity(), value)
                    viewer.material_slider.GetRepresentation().SetValue(1 - value)
                    viewer.material_slider.InvokeEvent("InteractionEvent")
                    self.assertAlmostEqual(viewer.actors[1].GetProperty().GetOpacity(), 1 - value)
                self.assertEqual(list(vtk_to_numpy(data.GetCellData().GetArray("material_id"))), labels)
                viewer.interactor.SetKeyEventInformation(0, 0, "e", 0, "e")
                viewer.interactor.InvokeEvent("KeyPressEvent")
                self.assertFalse(viewer.actors[1].GetProperty().GetEdgeVisibility())
                screenshot = path / "preview.png"
                viewer.screenshot(screenshot)
                self.assertTrue(screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                with self.assertRaisesRegex(ValueError, "already exists"):
                    viewer.screenshot(screenshot)
            finally:
                viewer.window.Finalize()
            after = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in before}
            self.assertEqual(before, after)

    def test_sparse_ids_selection_and_no_air(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            fixture(path, [17, 4294967295], [2, 1, 1])
            data, _, counts = read_grid(path, 100)
            viewer = VoxelViewer(data, counts, overrides={4294967295: .4}, offscreen=True)
            try:
                self.assertNotIn(0, viewer.actors)
                viewer.render()
                viewer.interactor.SetKeyEventInformation(0, 0, "\0", 0, "Right")
                viewer.interactor.InvokeEvent("KeyPressEvent")
                self.assertEqual(viewer.solid_ids[viewer.selected_index], 4294967295)
                self.assertAlmostEqual(viewer.material_slider.GetRepresentation().GetValue(), .4)
                viewer.material_slider.GetRepresentation().SetValue(.2)
                viewer.material_slider.InvokeEvent("InteractionEvent")
                self.assertAlmostEqual(viewer.actors[4294967295].GetProperty().GetOpacity(), .2)
                self.assertEqual(viewer.actors[17].GetProperty().GetOpacity(), 1)
                viewer.select_material(1)
                self.assertEqual(viewer.solid_ids[viewer.selected_index], 17)
            finally:
                viewer.window.Finalize()
            with self.assertRaisesRegex(ValueError, "cell limit"):
                read_grid(path, 1)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["air_cells"] = 1
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "counts differ"):
                read_grid(path, 100)

    def test_invalid_opacity(self):
        for value in ("nan", "inf", "-0.1", "1.1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                opacity(value)
        self.assertEqual(material_opacity("4294967295=0"), (4294967295, 0))


if __name__ == "__main__":
    unittest.main()
