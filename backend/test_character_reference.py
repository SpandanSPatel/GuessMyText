"""
Unit tests for GuessMyText Character Reference feature:
  1. Validates manifest.json integrity and schema across all 67 classes.
  2. Confirms exact distribution: 10 digits, 26 uppercase, 26 lowercase, 5 operators.
  3. Tests GET /api/character-references endpoint.
  4. Verifies static asset folder structure exists and is ready for manual copy.
  5. Verifies graceful handling of empty image lists and configured image entries.
"""

import json
import unittest
from pathlib import Path
from app import app, MANIFEST_PATH, CHARACTER_REF_DIR


class TestCharacterReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_manifest_file_exists_and_valid_json(self):
        """Test that manifest.json exists and is valid JSON."""
        self.assertTrue(MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("metadata", data)
        self.assertIn("characters", data)

    def test_02_all_67_classes_represented_exactly_once(self):
        """Test that exactly 67 classes exist across the 4 required categories."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        chars = data["characters"]
        self.assertEqual(len(chars), 67, f"Expected 67 character classes, found {len(chars)}")

        digits = [c for c in chars.values() if c["category"] == "digit"]
        uppercase = [c for c in chars.values() if c["category"] == "uppercase"]
        lowercase = [c for c in chars.values() if c["category"] == "lowercase"]
        operators = [c for c in chars.values() if c["category"] == "operator"]

        self.assertEqual(len(digits), 10, f"Expected 10 digits, found {len(digits)}")
        self.assertEqual(len(uppercase), 26, f"Expected 26 uppercase, found {len(uppercase)}")
        self.assertEqual(len(lowercase), 26, f"Expected 26 lowercase, found {len(lowercase)}")
        self.assertEqual(len(operators), 5, f"Expected 5 operators, found {len(operators)}")

        # Check expected character glyphs
        expected_digits = [str(i) for i in range(10)]
        expected_uppercase = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
        expected_lowercase = [chr(c) for c in range(ord('a'), ord('z') + 1)]
        expected_operators = ['+', '-', '*', '÷', '%']

        found_digits = [d["character"] for d in digits]
        found_uppercase = [u["character"] for u in uppercase]
        found_lowercase = [l["character"] for l in lowercase]
        found_operators = [o["character"] for o in operators]

        self.assertEqual(sorted(found_digits), sorted(expected_digits))
        self.assertEqual(sorted(found_uppercase), sorted(expected_uppercase))
        self.assertEqual(sorted(found_lowercase), sorted(expected_lowercase))
        self.assertEqual(sorted(found_operators), sorted(expected_operators))

    def test_03_character_entry_schema(self):
        """Test that every character entry satisfies the manifest schema."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        chars = data["characters"]

        for char_id, entry in chars.items():
            self.assertIn("id", entry)
            self.assertIn("character", entry)
            self.assertIn("category", entry)
            self.assertIn("folder", entry)
            self.assertIn("images", entry)
            self.assertIsInstance(entry["images"], list)

    def test_04_subdirectories_exist(self):
        """Test that all 67 subdirectories exist under static/character-references/."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        chars = data["characters"]

        for char_id, entry in chars.items():
            target_folder = CHARACTER_REF_DIR / entry["folder"]
            self.assertTrue(
                target_folder.exists(),
                f"Target subdirectory missing for {char_id}: {target_folder}"
            )

    def test_05_api_character_references_endpoint(self):
        """Test GET /api/character-references returns 200 with full catalog."""
        res = self.client.get("/api/character-references")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["metadata"]["total_classes"], 67)
        self.assertEqual(len(data["characters"]), 67)

    def test_06_configured_images_exist_and_accessible(self):
        """Test that every image in manifest.json exists on disk and is served via static route."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        chars = data["characters"]

        total_images = 0
        for char_id, entry in chars.items():
            for img in entry["images"]:
                total_images += 1
                img_file = CHARACTER_REF_DIR / img["path"]
                self.assertTrue(img_file.exists(), f"Image file missing on disk: {img_file}")
                self.assertGreater(img_file.stat().st_size, 0, f"Image file is empty: {img_file}")

                # Verify Flask static route returns HTTP 200
                res = self.client.get(f"/static/character-references/{img['path']}")
                self.assertEqual(
                    res.status_code, 200,
                    f"Static asset not accessible via HTTP 200: /static/character-references/{img['path']}"
                )
                res.close()

        self.assertEqual(total_images, 79, f"Expected 79 reference images, found {total_images}")


if __name__ == "__main__":
    unittest.main()
