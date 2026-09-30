"""Small TIFF integration checks for the preprocessing output contract."""
import argparse
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from PIL import Image
from affine import Affine

import preprocess_sar
import sar_preprocess_config as defaults


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "tests/fixtures/taxonomy.json"


class PreprocessTests(unittest.TestCase):
    def test_padded_edges_raw_backup_and_depth_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            images, labels, splits = (base / name for name in ("source_images", "source_labels", "splits"))
            for path in (images, labels, splits):
                path.mkdir()
            pixels = np.arange(600 * 600, dtype=np.uint32).reshape(600, 600).astype(np.uint16)
            for stem in ("train_scene", "val_scene"):
                with rasterio.open(images / f"{stem}.tif", "w", driver="GTiff", width=600, height=600,
                                   count=1, dtype="uint16", transform=Affine(1, 0, 0, 0, -1, 600)) as dst:
                    dst.write(pixels if stem == "train_scene" else np.bitwise_xor(pixels, 12345), 1)
            fine = "Car-Medium_Car-Buick_Excelle_GT"
            external = "Contest-Car-Model"
            coarse = "Car-Car-Car"
            (labels / "train_scene.txt").write_text(
                f"{external} 570 570 590 570 590 590 570 590\n"
                f"{coarse} 20 20 40 20 40 40 20 40\n", encoding="utf-8")
            (labels / "val_scene.txt").write_text(
                f"{external} 570 570 590 570 590 590 570 590\n", encoding="utf-8")
            (splits / "train.txt").write_text("train_scene.tif\n", encoding="utf-8")
            (splits / "val.txt").write_text("val_scene.tif\n", encoding="utf-8")
            cfg = SimpleNamespace(**{name: getattr(defaults, name) for name in dir(defaults) if name.isupper()})
            cfg.RAW_IMAGES, cfg.RAW_LABELS, cfg.OUTPUT_DATASET = str(images), str(labels), str(base / "prepared")
            cfg.OFFICIAL_SPLIT_DIR, cfg.TAXONOMY_JSON = str(splits), str(TAXONOMY)
            cfg.SOURCE_SCHEMA_CONFIRMED = True
            cfg.NEGATIVE_KEEP_RATIO = 1.0
            cfg.CLASS_ALIASES = {external: fine}
            result = preprocess_sar.run(cfg)
            output = Path(cfg.OUTPUT_DATASET)

            # Exercise the command teammates will invoke, not only run(cfg).
            cli_config = base / "smoke_config.py"
            cli_values = {name: getattr(cfg, name) for name in vars(cfg)}
            cli_values["OUTPUT_DATASET"] = str(base / "prepared_cli")
            cli_config.write_text("".join(f"{name} = {value!r}\n" for name, value in cli_values.items()),
                                  encoding="utf-8")
            completed = subprocess.run([sys.executable, str(ROOT / "preprocess_sar.py"), "--config",
                                        str(cli_config)], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(completed.stdout)["split_images"], {"train": 4, "val": 4})
            self.assertTrue((base / "prepared_cli/preprocess_manifest.json").is_file())

            self.assertEqual(result["split_images"], {"train": 4, "val": 4})
            self.assertEqual((output / "test.txt").read_text(), "")
            for split in ("train", "val"):
                for name in (output / f"{split}.txt").read_text().splitlines():
                    with Image.open(output / name) as tile:
                        self.assertEqual(tile.size, (512, 512))
                    self.assertTrue((output / name.replace("images/", "labels/").replace(".png", ".txt")).exists())
            train_edge = next(p for p in (output / "labels").glob("train_scene*_r00001_c00001.txt"))
            val_edge = next(p for p in (output / "labels").glob("val_scene*_r00001_c00001.txt"))
            backup_edge = output / "labels_original" / train_edge.name
            self.assertTrue(train_edge.read_text().startswith("Car-Medium_Car-Medium_Car "))
            edge_coordinates = [float(value) for value in train_edge.read_text().split()[1:]]
            self.assertTrue(all(0 <= value <= 88 for value in edge_coordinates))
            self.assertTrue(backup_edge.read_text().startswith(external + " "))
            self.assertEqual((output / "source_labels_original/train_scene.txt").read_bytes(),
                             (labels / "train_scene.txt").read_bytes())
            self.assertTrue(val_edge.read_text().startswith(external + " "))
            self.assertEqual(json.loads((output / "class_map.json").read_text())[external], fine)
            self.assertIn(coarse, "".join(p.read_text() for p in (output / "labels").glob("train_scene*.txt")))
            with Image.open(output / "images" / train_edge.with_suffix(".png").name) as image:
                self.assertEqual(image.getpixel((100, 100)), 0)  # Outside the 600x600 source.
                self.assertNotEqual(image.getpixel((70, 70)), 0)  # Real edge data is retained.
            manifest = json.loads((output / "preprocess_manifest.json").read_text())
            self.assertEqual(manifest["train_depth_limit"], 2)

            # The source-only release can run this test without the full framework.
            if not (ROOT / "SARdetect示例/D.codes/2.utils/dataset_adapter.py").is_file():
                return
            # Read the generated files through SARdetect's real label adapter.
            sys.path.insert(0, str(ROOT / "SARdetect示例/D.codes/2.utils"))
            sys.modules.setdefault("cv2", types.ModuleType("cv2"))
            import dataset_adapter
            previous_root = dataset_adapter.ROOT
            dataset_adapter.ROOT = base / "framework"
            try:
                adapted = dataset_adapter.prepare(output, TAXONOMY)
                train = json.loads((adapted / "annotations/train.json").read_text())
                val = json.loads((adapted / "annotations/val.json").read_text())
                test = json.loads((adapted / "annotations/test.json").read_text())
            finally:
                dataset_adapter.ROOT = previous_root
            self.assertEqual(sorted(row["mixed_label_depth"] for row in train["annotations"]), [1, 2])
            self.assertEqual([row["mixed_label_depth"] for row in val["annotations"]], [3])
            self.assertEqual(test["images"], [])

            import competition_workflow
            (base / "A.dataset").mkdir()
            (base / "A.dataset/contest").symlink_to(output, target_is_directory=True)
            previous_root = competition_workflow.ROOT
            competition_workflow.ROOT = base
            try:
                competition_workflow.analyze(argparse.Namespace(dataset="contest", group_regex=None,
                                                                ratios=None, seed=3929, config=None))
            finally:
                competition_workflow.ROOT = previous_root
            analysis = next((base / "C.results/0.settings/contest").glob("analysis-*/analysis.json"))
            report = json.loads(analysis.read_text())
            self.assertEqual(report["statistics"]["train"]["depth_counts"], [1, 1, 0])
            self.assertEqual(report["original_train_label_depth"]["depth_counts"], [1, 0, 1])

    def test_polygon_clipping_keeps_coordinates_inside_tile(self):
        clipped = preprocess_sar.clip_polygon([(500, 500), (530, 500), (530, 530), (500, 530)],
                                               512, 512, 600, 600)
        quad = preprocess_sar.four_point_approximation(clipped)
        self.assertIsNotNone(quad)
        self.assertTrue(all(512 <= x <= 600 and 512 <= y <= 600 for x, y in quad))
        self.assertAlmostEqual(preprocess_sar.polygon_area(clipped) / 900, 18 * 18 / 900)

    def test_default_label_template_works_at_root_and_in_subdirectory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.assertEqual(preprocess_sar.label_path(Path("a.tif"), root, defaults), root / "a.txt")
            self.assertEqual(preprocess_sar.label_path(Path("sub/a.tif"), root, defaults), root / "sub/a.txt")


if __name__ == "__main__":
    unittest.main()
