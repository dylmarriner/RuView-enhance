//! Layer 1 — the deterministic local renderer ("sensor truth").
//!
//! Rasterizes a real [`SensingUpdate`] frame with no model inference and no
//! randomness: same input frame -> same output pixels, always. This is a CPU
//! software rasterizer (`tiny-skia`), not `wgpu` — see ADR-352
//! "Consequences" for why headless GPU rendering is a stated follow-up
//! rather than this milestone's implementation.
//!
//! Text/glyph rendering (numeric vitals readouts, the burned-in "SYNTHETIC"
//! label required on the Layer-1+2 *composite*) is not implemented yet — it
//! needs an embedded font dependency this milestone deliberately deferred.
//! Vitals are instead rendered as deterministic, data-proportional bars so
//! the frame stays real and legible without fabricating a text rendering
//! stack under time pressure. The composite stream (Layer 2, a later phase)
//! must not ship without real burned-in text — flagged as required
//! follow-up, not silently dropped.

use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Rect, Stroke, Transform};

use crate::sensing_client::{NodeInfo, Person, SensingUpdate};

/// Fixed COCO-17-style skeleton bone list, matched by keypoint `name`. Bones
/// whose endpoint names are absent from a given `Person`'s keypoints (a
/// low-confidence or partially-observed frame) are simply skipped — never
/// interpolated or invented.
const BONES: &[(&str, &str)] = &[
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
];

#[derive(Debug, Clone, Copy)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    /// Real sensing-server image-plane keypoint coordinates are scaled into
    /// canvas space by this factor (keeps the renderer independent of the
    /// sensing-server's own coordinate convention).
    pub keypoint_scale: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 288,
            keypoint_scale: 1.0,
        }
    }
}

fn find_kp<'a>(person: &'a Person, name: &str) -> Option<&'a crate::sensing_client::Keypoint> {
    person.keypoints.iter().find(|k| k.name == name)
}

fn presence_color(motion_level: &str) -> Color {
    match motion_level {
        "present_moving" => Color::from_rgba8(255, 140, 0, 255),
        "present_still" => Color::from_rgba8(0, 200, 160, 255),
        _ => Color::from_rgba8(90, 90, 100, 255),
    }
}

/// Render one real, live [`SensingUpdate`] frame deterministically. Returns
/// an RGBA pixmap; the caller decides how to persist/broadcast it.
#[must_use]
pub fn render_frame(update: &SensingUpdate, cfg: RenderConfig) -> Pixmap {
    let mut pixmap = Pixmap::new(cfg.width, cfg.height).expect("nonzero render dimensions");
    pixmap.fill(Color::from_rgba8(12, 14, 18, 255));

    draw_room_bounds(&mut pixmap, &update.nodes, cfg);
    draw_nodes(&mut pixmap, &update.nodes, &update.classification.motion_level, cfg);
    for person in &update.persons {
        draw_skeleton(&mut pixmap, person, cfg);
    }
    if let Some(vitals) = &update.vital_signs {
        draw_vitals_bars(&mut pixmap, vitals, cfg);
    }

    pixmap
}

fn draw_room_bounds(pixmap: &mut Pixmap, nodes: &[NodeInfo], cfg: RenderConfig) {
    if nodes.is_empty() {
        return;
    }
    let mut paint = Paint::default();
    paint.set_color(Color::from_rgba8(40, 44, 52, 255));
    paint.anti_alias = true;
    let margin = 16.0_f32;
    let rect = Rect::from_xywh(
        margin,
        margin,
        cfg.width as f32 - 2.0 * margin,
        cfg.height as f32 - 2.0 * margin,
    );
    if let Some(rect) = rect {
        let stroke = Stroke {
            width: 1.5,
            ..Default::default()
        };
        if let Some(path) = PathBuilder::from_rect(rect).into() {
            pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }
}

/// Maps a node's real `[x, y, z]` room-metric position onto the schematic
/// room-bounds rectangle. This is a schematic top-down layout, not a photo
/// of the room — real sensor-reported positions, deterministically mapped.
fn node_canvas_pos(node: &NodeInfo, cfg: RenderConfig) -> (f32, f32) {
    let margin = 16.0_f32;
    let usable_w = cfg.width as f32 - 2.0 * margin;
    let usable_h = cfg.height as f32 - 2.0 * margin;
    // Real deployments so far span roughly a 0-5m room (see synthetic_deployment
    // and the live node positions observed, e.g. [2.0, 0.0, 1.5]); clamp so an
    // out-of-expected-range position still renders inside the frame instead of
    // silently vanishing off-canvas.
    let room_span_m = 6.0_f32;
    let nx = (node.position[0] as f32 / room_span_m).clamp(0.0, 1.0);
    let ny = (node.position[1] as f32 / room_span_m).clamp(0.0, 1.0);
    (margin + nx * usable_w, margin + ny * usable_h)
}

fn draw_nodes(pixmap: &mut Pixmap, nodes: &[NodeInfo], room_motion_level: &str, cfg: RenderConfig) {
    for node in nodes {
        let (cx, cy) = node_canvas_pos(node, cfg);
        let color = node
            .node_inference
            .as_ref()
            .map(|ni| presence_color(&ni.classification))
            .unwrap_or_else(|| presence_color(room_motion_level));
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;
        let radius = 4.0_f32;
        if let Some(path) = PathBuilder::from_circle(cx, cy, radius) {
            pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
        }
    }
}

fn draw_skeleton(pixmap: &mut Pixmap, person: &Person, cfg: RenderConfig) {
    let mut paint = Paint::default();
    paint.set_color(Color::from_rgba8(80, 200, 255, 220));
    paint.anti_alias = true;
    let stroke = Stroke {
        width: 2.0,
        ..Default::default()
    };

    for (a, b) in BONES {
        if let (Some(ka), Some(kb)) = (find_kp(person, a), find_kp(person, b)) {
            let mut pb = PathBuilder::new();
            pb.move_to(ka.x as f32 * cfg.keypoint_scale, ka.y as f32 * cfg.keypoint_scale);
            pb.line_to(kb.x as f32 * cfg.keypoint_scale, kb.y as f32 * cfg.keypoint_scale);
            if let Some(path) = pb.finish() {
                pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
            }
        }
    }

    let mut joint_paint = Paint::default();
    joint_paint.set_color(Color::from_rgba8(255, 255, 255, 255));
    joint_paint.anti_alias = true;
    for kp in &person.keypoints {
        // Confidence gates visibility rather than being silently ignored —
        // an unobserved joint (confidence 0.0, as seen in the live sample)
        // must not render as if it were seen.
        if kp.confidence <= 0.0 {
            continue;
        }
        let x = kp.x as f32 * cfg.keypoint_scale;
        let y = kp.y as f32 * cfg.keypoint_scale;
        if let Some(path) = PathBuilder::from_circle(x, y, 2.5) {
            pixmap.fill_path(&path, &joint_paint, FillRule::Winding, Transform::identity(), None);
        }
    }
}

/// Deterministic, data-proportional vitals bars in the top-left corner. See
/// module docs for why this is bars, not numeral text, for this milestone.
fn draw_vitals_bars(pixmap: &mut Pixmap, vitals: &crate::sensing_client::VitalSigns, cfg: RenderConfig) {
    let _ = cfg;
    let bars: [(f32, Color); 2] = [
        (
            vitals.breathing_rate_bpm.unwrap_or(0.0) as f32 / 30.0, // physiological max ~30 bpm
            Color::from_rgba8(120, 220, 255, 255),
        ),
        (
            vitals.heart_rate_bpm.unwrap_or(0.0) as f32 / 180.0, // physiological max ~180 bpm
            Color::from_rgba8(255, 100, 130, 255),
        ),
    ];
    let bar_w = 60.0_f32;
    let bar_h = 6.0_f32;
    for (i, (frac, color)) in bars.iter().enumerate() {
        let y = 12.0 + i as f32 * (bar_h + 4.0);
        let frac = frac.clamp(0.0, 1.0);
        let mut bg_paint = Paint::default();
        bg_paint.set_color(Color::from_rgba8(50, 50, 55, 255));
        if let Some(rect) = Rect::from_xywh(12.0, y, bar_w, bar_h) {
            let path = PathBuilder::from_rect(rect);
            pixmap.fill_path(&path, &bg_paint, FillRule::Winding, Transform::identity(), None);
        }
        let mut fg_paint = Paint::default();
        fg_paint.set_color(*color);
        if let Some(rect) = Rect::from_xywh(12.0, y, bar_w * frac, bar_h) {
            let path = PathBuilder::from_rect(rect);
            pixmap.fill_path(&path, &fg_paint, FillRule::Winding, Transform::identity(), None);
        }
    }
}
