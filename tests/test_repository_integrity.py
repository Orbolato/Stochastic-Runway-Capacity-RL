import csv
import hashlib
from pathlib import Path
import unittest
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Git blob IDs of the immutable study artifacts in the initial public package.
# These are checked from file bytes, not from Git metadata, so accidental local
# modification is detected even outside a Git checkout.
EXPECTED_BLOBS = {
    "data/data.csv": "5660ea374ddd186dff838907a8b76ec20c0f0472",
    "data/data_gru.Rda": "9490d9594af504bc33ababa050997ef819bfcc20",
    "data/new_taf_list.csv": "61b4ead4316413a0d939700632657b6a7d02c8cd",
    "data/std_taf.csv": "f747e78347f78ab230dd593b58f4812e42a60fff",
    "data/raw-data/metar_sbgr_2019_2023.Rda": "280267e00f9c562c1ac0543dc711750b8307a217",
    "data/raw-data/taf.csv": "1f7b0e8a77b8ca6a38a58ca39b97fb1926e8b916",
    "results/results.txt": "f5f3d8a719f7f7e208a0fbc09d2d623096cef97b",
    "results/returns_209_test_days.csv": "95c4a05e05cf8985f117a62821f927d45d6d4471",
    "results/returns_7_test_days.csv": "aca9f98a98b0d9e3beca5df6ef50021f48ac79ca",
    "curves/VMC_10R_10L_RCCE_9900.csv": "86bf2aceb16f5a0be22d82a00b2cedcf9b334a6c",
    "curves/VMC_28L_28R_RCCE_9900.csv": "a92c23ca7a38debfd41b3ba2f0c3b643d5721466",
    "curves/IMC_10R_10L_RCCE_9500.csv": "1ed3217222502e98a3138a951043ca2127e66027",
    "curves/IMC_28L_28R_RCCE_9500.csv": "713280ef3400fcf338cc2579bd705adac123c425",
    "curves/LIMC_10R_10L_RCCE_9000.csv": "4019d6f642544f9a24ed240d440bebff71ec5c1e",
    "curves/LIMC_28L_28R_RCCE_9000.csv": "ec347fc3c44b7c76b211a7f551c7fee2947c8065",
    "trained_models/ppo/db_approach_1/ppo_model_db_1.zip": "d2f5b24acf193f9be5dc601bb81b0f4cb2b7a9d5",
    "trained_models/ppo/db_approach_2/ppo_model_db_2.zip": "7cb73562169deddc0d2dce5009f4c458e727413f",
    "trained_models/ppo/fb_approach_1/ppo_model_fb_1_2022-12-10.zip": "7ca1bb8bf11207caa24bbd0ac07e7006f7f4abee",
    "trained_models/ppo/fb_approach_1/ppo_model_fb_1_2022-12-30.zip": "c58b67cfcb0976f836a31ac503e4aa5de2085c55",
    "trained_models/ppo/fb_approach_1/ppo_model_fb_1_2022-8-17.zip": "6b5b158abdd2a88513d1061614fcc7d8c2969915",
    "trained_models/ppo/fb_approach_1/ppo_model_fb_1_2023-1-12.zip": "eedc3b74c1180a0e71a9f475dee9308a13fe4962",
    "trained_models/ppo/fb_approach_1/ppo_model_fb_1_2023-5-12.zip": "097c752ba894fe5dd087b83066b6d46ed7b4db79",
    "trained_models/ppo/fb_approach_1/ppo_model_fb_1_2023-8-5.zip": "ea32232da4823b373cc3f5317cd29638cde19f63",
    "trained_models/ppo/fb_approach_1/ppo_model_fb_1_2023-9-15.zip": "69aabff6eeea10759c77dea0d99626672b01c7b2",
    "trained_models/ppo/fb_approach_2/ppo_model_fb_2_2022-12-10.zip": "ee0cfaf0af9126a0b63fe06587d8d8128b429a02",
    "trained_models/ppo/fb_approach_2/ppo_model_fb_2_2022-12-30.zip": "61847d8651e89e184723ad4ab5d0cc4efefb3f04",
    "trained_models/ppo/fb_approach_2/ppo_model_fb_2_2022-8-17.zip": "5f276bce61b0bace069d6bf38b27e878c3cbaa0d",
    "trained_models/ppo/fb_approach_2/ppo_model_fb_2_2023-1-12.zip": "31f015900c1a6c986a9f3f5cde1bf55e09268c1f",
    "trained_models/ppo/fb_approach_2/ppo_model_fb_2_2023-5-12.zip": "a07eeb578a623501246fc6c7566a0c8ce0363b3d",
    "trained_models/ppo/fb_approach_2/ppo_model_fb_2_2023-8-5.zip": "f482e0679f4118f043f3183d8a444ec38820b588",
    "trained_models/ppo/fb_approach_2/ppo_model_fb_2_2023-9-15.zip": "0bb86dfc79ae84dd70c310056efca9a16e7103be",
}


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def normalize_day(value: str):
    y, m, d = (int(x) for x in value.strip().split("-"))
    return y, m, d


class RepositoryIntegrityTests(unittest.TestCase):
    def test_immutable_artifact_blob_ids(self):
        for rel, expected in EXPECTED_BLOBS.items():
            path = ROOT / rel
            self.assertTrue(path.is_file(), rel)
            self.assertEqual(git_blob_sha(path), expected, rel)

    def test_processed_dataset_shape_and_operational_days(self):
        path = ROOT / "data" / "data.csv"
        day_counts = {}
        rows = 0
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self.assertIn("day", reader.fieldnames)
            for row in reader:
                rows += 1
                day = normalize_day(row["day"])
                day_counts[day] = day_counts.get(day, 0) + 1

        self.assertEqual(rows, 66816)
        self.assertEqual(len(day_counts), 696)
        self.assertEqual(set(day_counts.values()), {96})
        for d in range(24, 29):
            self.assertNotIn((2023, 4, d), day_counts)

    def test_archived_result_tables(self):
        with (ROOT / "results" / "returns_209_test_days.csv").open(
            newline="", encoding="utf-8-sig"
        ) as f:
            rows_209 = list(csv.DictReader(f))
        with (ROOT / "results" / "returns_7_test_days.csv").open(
            newline="", encoding="utf-8-sig"
        ) as f:
            rows_7 = list(csv.DictReader(f))

        self.assertEqual(len(rows_209), 209)
        self.assertEqual(len(rows_7), 7)
        # This historical archived row is intentionally preserved; see PROVENANCE.md.
        self.assertIn((2023, 4, 28), {normalize_day(r["day"]) for r in rows_209})

    def test_trained_model_archives_are_valid_zip_files(self):
        models = sorted((ROOT / "trained_models" / "ppo").rglob("*.zip"))
        self.assertEqual(len(models), 16)
        for model in models:
            with zipfile.ZipFile(model) as zf:
                self.assertIsNone(zf.testzip(), str(model))
                self.assertTrue(zf.namelist(), str(model))

    def test_restricted_source_records_are_not_committed(self):
        self.assertFalse((ROOT / "data" / "raw-data" / "traffic-data").exists())

    def test_citation_metadata_points_to_article_and_repository(self):
        with (ROOT / "CITATION.cff").open(encoding="utf-8") as f:
            cff = yaml.safe_load(f)
        self.assertEqual(
            cff["preferred-citation"]["doi"],
            "10.1016/j.tra.2025.104620",
        )
        self.assertEqual(
            cff["repository-code"],
            "https://github.com/Orbolato/Stochastic-Runway-Capacity-RL",
        )


if __name__ == "__main__":
    unittest.main()
