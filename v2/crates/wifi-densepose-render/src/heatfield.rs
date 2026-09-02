//! Animated Layer-1 heatfield compositor.
//!
//! Confirmed direction (2026-09-01, user via team-lead): the static
//! `assets/room_csi_composite.png` experiment (a real `signal_field` grid
//! screen-blended onto `assets/room-style-1.png`) becomes Layer 1's real
//! continuous per-frame target, not a one-off still. This module renders
//! that composite every frame from the real live `signal_field` grid.
//!
//! `room-style-1.png` (embedded, see `../assets/`) is an illustrated,
//! AI-generated stylistic backdrop — a generic bedroom scene, not a captured
//! photo and not a model of any real deployment's actual room geometry. It
//! is a decorative anchor only; the warm heat overlay is the only part of
//! the frame driven by real per-frame sensor data. Per this repo's honesty
//! discipline, the composite must not be presented as a photo of the real
//! room, and the ADR-352 SYNTHETIC label applies once this composite is fed
//! through Layer 2 (fal.ai) — Layer 1 alone renders it locally,
//! deterministically, with no model inference.
//!
//! Grid axis order: the live `signal_field.grid_size` was observed as
//! `[20, 1, 20]` (400 values). The server source for this field is not in
//! this checked-out worktree (same drift noted in `sensing_client.rs`), so
//! the exact semantic axis order (which of the two size-20 axes is room-x
//! vs room-depth) is inferred, not authoritative: this module treats the
//! flattened `values` as a row-major `grid_size[0] x grid_size[2]` 2D grid
//! and ignores the singleton middle axis. Good enough for a real, live,
//! data-driven heat overlay; not a claim about which physical axis is which.

use image::{Rgba, RgbaImage};
use std::sync::OnceLock;

use crate::sensing_client::SignalField;

const ROOM_STYLE_BG_BYTES: &[u8] = include_bytes!("../assets/room-style-1.png");

fn decoded_background() -> &'static RgbaImage {
    static CACHE: OnceLock<RgbaImage> = OnceLock::new();
    CACHE.get_or_init(|| {
        image::load_from_memory(ROOM_STYLE_BG_BYTES)
            .expect("embedded room-style-1.png must decode")
            .to_rgba8()
    })
}

/// Resized-once cache of the background at the render target size. Avoids
/// re-resizing a 2.5MB source image every frame; only the heat overlay
/// varies per call.
fn background_at(width: u32, height: u32) -> RgbaImage {
    static RESIZED: OnceLock<(u32, u32, RgbaImage)> = OnceLock::new();
    let cached = RESIZED.get_or_init(|| {
        let resized = image::imageops::resize(
            decoded_background(),
            width,
            height,
            image::imageops::FilterType::Triangle,
        );
        (width, height, resized)
    });
    if cached.0 == width && cached.1 == height {
        cached.2.clone()
    } else {
        // A run that changes render dimensions mid-process is not the
        // expected path (RenderConfig is fixed per-run); resize fresh rather
        // than serve a mismatched cached size.
        image::imageops::resize(
            decoded_background(),
            width,
            height,
            image::imageops::FilterType::Triangle,
        )
    }
}

/// Bilinear-sample the real signal_field grid at normalized `(u, v)` in
/// `[0, 1] x [0, 1]`, using the min/max of *this frame's* real values for
/// contrast normalization (documented per-frame rescaling, not fabricated
/// data — see module docs).
fn sample_field(field: &SignalField, u: f32, v: f32, min: f64, max: f64) -> f32 {
    let rows = field.grid_size[0].max(1);
    let cols = field.grid_size[2].max(1);
    let fx = (u.clamp(0.0, 1.0) * (cols as f32 - 1.0).max(0.0)).max(0.0);
    let fy = (v.clamp(0.0, 1.0) * (rows as f32 - 1.0).max(0.0)).max(0.0);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(cols - 1);
    let y1 = (y0 + 1).min(rows - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let at = |r: usize, c: usize| -> f64 {
        field.values.get(r * cols + c).copied().unwrap_or(0.0)
    };
    let v00 = at(y0, x0);
    let v10 = at(y0, x1);
    let v01 = at(y1, x0);
    let v11 = at(y1, x1);
    let top = v00 + (v10 - v00) * tx as f64;
    let bottom = v01 + (v11 - v01) * tx as f64;
    let raw = top + (bottom - top) * ty as f64;

    let span = (max - min).max(1e-9);
    (((raw - min) / span).clamp(0.0, 1.0)) as f32
}

/// Warm black -> deep-orange -> bright-yellow ramp matching the reference
/// `room_csi_composite.png` glow.
fn warm_color(intensity: f32) -> (u8, u8, u8) {
    let i = intensity.clamp(0.0, 1.0);
    let r = (255.0 * (i * 1.4).min(1.0)) as u8;
    let g = (200.0 * (i * i)) as u8;
    let b = (60.0 * (i * i * i)) as u8;
    (r, g, b)
}

/// Render one real, live composited Layer-1 frame: the illustrated room
/// backdrop with the real `signal_field` grid screen-blended on top.
/// `overlay_strength` scales how strongly the real field shows through
/// (kept in `[0, 1]`; the reference composite uses a visible but not
/// overpowering glow).
#[must_use]
pub fn composite(width: u32, height: u32, field: &SignalField, overlay_strength: f32) -> RgbaImage {
    let base = background_at(width, height);
    let mut out = base.clone();

    if field.values.is_empty() {
        return out;
    }
    let min = field.values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = field.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    for y in 0..height {
        let v = y as f32 / (height.max(1) - 1).max(1) as f32;
        for x in 0..width {
            let u = x as f32 / (width.max(1) - 1).max(1) as f32;
            let intensity = sample_field(field, u, v, min, max);
            if intensity <= 0.02 {
                continue; // no measurable signal here — leave the backdrop untouched
            }
            let (or, og, ob) = warm_color(intensity);
            let base_px = base.get_pixel(x, y).0;
            let alpha = intensity * overlay_strength.clamp(0.0, 1.0);

            // Screen blend: result = 255 - (255-base)*(255-overlay)/255,
            // then linearly mixed back toward the base by `alpha` so a
            // low-intensity cell barely perturbs the backdrop.
            let screen = |b: u8, o: u8| -> u8 {
                255 - (((255 - b as u32) * (255 - o as u32)) / 255) as u8
            };
            let blended = [
                screen(base_px[0], or),
                screen(base_px[1], og),
                screen(base_px[2], ob),
            ];
            let mix = |b: u8, s: u8| -> u8 { (b as f32 * (1.0 - alpha) + s as f32 * alpha).round() as u8 };
            out.put_pixel(
                x,
                y,
                Rgba([mix(base_px[0], blended[0]), mix(base_px[1], blended[1]), mix(base_px[2], blended[2]), 255]),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_zero_field_leaves_backdrop_unchanged() {
        let field = SignalField {
            grid_size: [4, 1, 4],
            values: vec![0.0; 16],
        };
        let out = composite(32, 32, &field, 1.0);
        let bg = background_at(32, 32);
        assert_eq!(out.as_raw(), bg.as_raw());
    }

    #[test]
    fn hot_cell_visibly_changes_its_region() {
        let mut values = vec![0.0; 16];
        values[5] = 10.0; // one hot cell amid a flat field
        let field = SignalField {
            grid_size: [4, 1, 4],
            values,
        };
        let out = composite(64, 64, &field, 1.0);
        let bg = background_at(64, 64);
        assert_ne!(out.as_raw(), bg.as_raw(), "a real hot cell must visibly perturb the frame");
    }
}
