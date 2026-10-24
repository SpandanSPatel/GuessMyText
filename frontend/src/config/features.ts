/**
 * GuessMyText Application Feature Flags
 *
 * Use this centralized configuration setting to control feature visibility across
 * the entire GuessMyText application.
 *
 * SHOW_SEGMENTATION_PREVIEW:
 *   - false: Hides the Segmentation Preview debugging section across all pages
 *            (V2 Line Recognition, V3 Paragraph Recognition, V4 Math Solver),
 *            and hides technical model consensus status messages
 *            ("All characters recognized with majority or unanimous consensus.").
 *            Leaves no empty cards, reserved blank areas, or layout gaps.
 *   - true:  Restores the full visual Segmentation Preview on all supported pages,
 *            including character crops, model input tensors (32x32, 64x64, 128x128),
 *            paragraph canvas diagnostics, and consensus status notes.
 */
export const SHOW_SEGMENTATION_PREVIEW = false;
