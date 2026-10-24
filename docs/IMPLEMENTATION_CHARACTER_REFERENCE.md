# GuessMyText — Character Reference Page

## 1. Objective

Create a new Character Reference page in GuessMyText that displays reference images for the characters in the original 67-class dataset.

Each character will have its own card containing zero or more selected handwriting-reference images.

The page must be fully implemented before the actual images are selected. Image selection and copying will be performed manually later, using a class-to-image-number list supplied by the project owner.

The implementation must integrate with the existing application without affecting V1, V2, V3, V4, model inference, or other pages.

## 2. Dataset and Character Classes

The original dataset contains 67 classes:

* Digits: `0–9` — 10 classes.
* Uppercase letters: `A–Z` — 26 classes.
* Lowercase letters: `a–z` — 26 classes.
* Mathematical operators: `÷`, `-`, `*`, `%`, `+` — 5 classes.

Total: 67 classes.

The reference page documents the original character collection, not just the specialized classes used by V2, V3, and V4.

Do not modify the existing dataset, training scripts, models, or universal label mapping.

## 3. Page Design

Follow GuessMyText's existing visual design, navigation, typography, spacing, components, and theme.

Do not introduce an unrelated design system or redesign existing pages.

### 3.1 Page Header

Title:

**Character Reference**

Description:

"Explore the characters supported by GuessMyText and view examples of the handwriting styles represented in its recognition dataset."

### 3.2 Summary Metrics

Display these metrics near the top of the page:

* Total character classes: 67.
* Digits: 10.
* Uppercase letters: 26.
* Lowercase letters: 26.
* Operators: 5.

Use compact summary elements consistent with the existing UI.

### 3.3 Search

Provide a search field with placeholder:

`Search characters...`

Search should match the character represented by each card.

Examples:

* Searching for `A` shows the uppercase A card.
* Searching for `7` shows the digit 7 card.
* Searching for `+` shows the addition-operator card.

Search should work together with category filters.

### 3.4 Category Filters

Provide the following categories:

* All
* Digits
* Uppercase
* Lowercase
* Operators

The default category is All.

Changing categories must update the visible cards without navigating away from the page.

### 3.5 Character Cards

Create one card for each of the 67 character classes.

Every card should contain:

* A prominent character label.
* The category of the character.
* Zero or more selected reference images.
* Variant labels where images are present.
* A small indication of the number of configured reference variants.

The number of images must be dynamic. Do not assume that every character has the same number of variants.

Examples:

* Character `0`: one selected image.
* Character `1`: two selected images.
* Character `A`: three selected images.
* Character `B`: no selected images yet.

These are examples of the intended behavior, not the actual image selections.

When no images are configured, show a neutral placeholder such as:

"References not added yet."

Do not use random images, generated handwriting, external images, or other characters as substitutes.

### 3.6 Image Preview

If compatible with the existing UI, clicking a reference image should open a larger preview.

The preview should:

* Display the selected image at a larger size.
* Identify the associated character and variant.
* Provide a clear way to close the preview.
* Work with keyboard accessibility and appropriate dialog semantics.

This is secondary to the core gallery and must not delay its implementation.

## 4. Reference Image Selection Workflow

The project owner will later provide a list specifying the selected image numbers for each class.

Image numbers refer to the existing source images numbered from 1 through 300 for each class located in `New/Images/`.

### Verified Source Image Conventions (from `New/Images/`)
The repository contains 67 class folders inside `New/Images/`, each containing exactly 300 PNG images named `1.png` through `300.png`:
* **Digits (`0–9`):** `New/Images/0 - Numbers/` through `New/Images/9 - Numbers/`
* **Uppercase (`A–Z`):** `New/Images/A - Uppercase Letters/` through `New/Images/Z - Uppercase Letters/`
* **Lowercase (`a–z`):** `New/Images/a - Lowercase Letters/` through `New/Images/z - Lowercase Letters/`
* **Operators (`+`, `-`, `*`, `÷`, `%`):**
  * Addition (`+`): `New/Images/plus - Operators/`
  * Subtraction (`-`): `New/Images/minus - Operators/`
  * Multiplication (`*`): `New/Images/multiply - Operators/`
  * Division (`÷`): `New/Images/divide - Operators/`
  * Modulo / Percent (`%`): `New/Images/percent - Operators/`

The agent must not select images automatically or decide which images should be used.

The manual copy workflow is:

1. The project owner prepares a class-to-image-number list (e.g. `{"1": [12, 105], "A": [7, 88]}`).
2. The project owner copies the selected images from `New/Images/<Class Folder>/<num>.png` into the target destination directory:
   `GuessMyText/backend/static/character-references/<folder>/variant_<k>.png`
3. The project owner registers the copied images in `GuessMyText/backend/static/character-references/manifest.json`.
4. The Character Reference page automatically displays the configured variants without restarting or code modifications.

The application does not access `New/Images/` at runtime; it serves exclusively from `backend/static/character-references/`.

## 5. Verified Static Asset Structure

The static directory for character reference images is located at:

`GuessMyText/backend/static/character-references/`

Subdirectories have been pre-created for all 67 classes:

```text
GuessMyText/backend/static/character-references/
├── manifest.json
├── digits/
│   ├── 0/ ... 9/
├── uppercase/
│   ├── A/ ... Z/
├── lowercase/
│   ├── a/ ... z/
└── operators/
    ├── plus/
    ├── minus/
    ├── multiply/
    ├── divide/
    └── modulo/
```

Use semantic directory names for operators (`plus`, `minus`, `multiply`, `divide`, `modulo`) to avoid URL encoding issues.

## 6. Central Reference Manifest

Create a central manifest describing all 67 characters and their selected reference images.

Use a simple JSON manifest or another format consistent with the existing project.

The manifest must include:

* Character identifier.
* Display character.
* Category.
* List of reference-image paths.
* Optional human-readable variant labels.

Example:

```json
{
  "1": {
    "character": "1",
    "category": "digit",
    "images": [
      {
        "path": "digits/1/variant_1.png",
        "label": "Variant 1"
      },
      {
        "path": "digits/1/variant_2.png",
        "label": "Variant 2"
      }
    ]
  },
  "A": {
    "character": "A",
    "category": "uppercase",
    "images": []
  },
  "+": {
    "character": "+",
    "category": "operator",
    "images": []
  }
}
```

This is an illustrative schema. Follow the final implementation's actual schema consistently.

All 67 characters should exist in the manifest, even when their image lists are empty.

The frontend must not hardcode image counts or assume a fixed number of variants.

The manifest must not modify or replace the existing model label mapping. It is only for the reference gallery.

## 7. Loading and Missing-Image Behavior

Because the page will initially have no selected images, it must work correctly with empty image lists.

Requirements:

* Render all 67 character cards even when no images have been added.
* Show the configured reference count for each character.
* Show the empty-state placeholder when a character has no images.
* Avoid broken-image icons.
* Handle missing static files gracefully.
* Keep the rest of the gallery functional if one image fails to load.
* Never substitute another image silently.

If the project uses a manifest served by Flask, validate the data and provide a clear error if the manifest itself cannot be loaded.

Do not add a database, upload API, or image-management backend for this feature.

## 8. Frontend Architecture

Reuse existing page layouts, navigation, buttons, cards, and styles wherever practical.

Keep the reference data separate from presentation components.

Recommended conceptual separation:

* Reference page: page layout and interactions.
* Character card: one character and its selected images.
* Category filters: filtering controls.
* Search: character lookup.
* Image preview: optional enlarged-image dialog.
* Manifest: character metadata and image paths.

These are conceptual responsibilities, not required filenames. Follow the actual project architecture and avoid unnecessary component fragmentation.

## 9. Navigation

Add a Character Reference entry to the existing navigation system.

Use the existing routing conventions.

The page should be accessible from the normal application interface and should work with direct navigation and page refresh according to the project's routing setup.

Do not break or rename existing routes.

## 10. Responsive Behavior and Accessibility

The gallery should:

* Use a responsive grid.
* Avoid horizontal overflow.
* Display cards consistently across screen sizes.
* Keep character labels legible.
* Preserve each reference image's aspect ratio.
* Avoid stretching or distorting images.
* Use descriptive image alternative text.
* Support keyboard navigation for interactive controls.
* Provide visible focus indicators.
* Maintain sufficient contrast.

Do not force every character card to display the same number of images.

## 11. Explicitly Out of Scope

Do not implement:

* Automatic image selection.
* Automatic image copying.
* Image generation.
* Dataset retraining.
* Changes to CNN models or model mappings.
* Database storage for reference images.
* User image uploads.
* Automatic discovery of all images in `New/images/`.
* Changes to V1–V4 recognition behavior.
* Changes to the Segmentation Preview visibility setting.
* Modifications to `CaptureMyText_Copy/`.

The page should simply display manually selected static images.

## 12. Testing and Acceptance Criteria

Verify the following:

### Page and Navigation

* The Character Reference page loads.
* The page is accessible through the existing navigation.
* Existing pages and routes continue to work.

### Character Collection

* All 67 classes are represented.
* Category counts are correct.
* Each character appears exactly once in the gallery.
* Operators are represented correctly.

### Search and Filters

* Searching for a character returns the correct card.
* All category filters work.
* Search and category filtering work together.
* Empty search results have a sensible message.

### Image Manifest

* Empty image lists render correctly.
* One image displays correctly.
* Multiple images display correctly.
* Variant labels are correct.
* Missing files are handled gracefully.
* One broken image does not break the gallery.

### Presentation

* Cards remain responsive.
* Images are not distorted.
* No horizontal overflow occurs at supported screen sizes.
* The optional enlarged preview opens and closes correctly if implemented.

### Regression

* V1–V4 remain functional.
* Model inference and label mappings are unchanged.
* Existing banner behavior is unchanged.
* The Segmentation Preview visibility setting is unchanged.
* `CaptureMyText_Copy/` remains untouched.

Report only tests that were actually executed.

## 13. Completion Report

At completion, report:

* Files created and modified.
* Actual page route.
* Actual static asset directory.
* Manifest location and schema.
* Final source-image naming assumptions verified from the repository.
* How to copy selected images and register them.
* Tests executed and their actual results.
* Any unresolved issues.

Do not claim unverified functionality as complete.
