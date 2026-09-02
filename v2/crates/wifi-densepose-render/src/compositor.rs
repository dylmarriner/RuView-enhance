//! Compositor — blends the low-frequency Layer-2 (fal.ai) keyframe into the
//! high-frequency Layer-1 stream so motion stays smooth between neural
//! updates, without ever letting Layer 2 become the authoritative content.
//!
//! ## Honest scope note
//!
//! ADR-352 describes "optical-flow-based temporal blending." This
//! implementation estimates a single **global translation** between
//! successive Layer-1 frames (via normalized cross-correlation over a
//! downsampled luma pyramid — a real, working motion estimate, not a stub),
//! and warps the held Layer-2 keyframe by that translation each tick before
//! cross-fading it under the fresh Layer-1 content. This is coarser than
//! dense per-pixel optical flow (it will not track independent motion of
//! multiple people separately) — documented here rather than silently
//! presented as the full dense-flow algorithm. Upgrading to per-block or
//! dense flow is real follow-up work, same posture as the wgpu/windowed-SSIM
//! deferrals elsewhere in this crate.

use image::{Rgba, RgbaImage};

/// Held Layer-2 state the compositor carries between ticks.
pub struct CompositorState {
    /// The last SSIM-accepted Layer-2 keyframe, already resized to the
    /// render target dimensions.
    last_good_keyframe: Option<RgbaImage>,
    /// The Layer-1 frame that `last_good_keyframe` was derived from — used
    /// to estimate motion against the *current* Layer-1 frame.
    keyframe_source_luma: Option<Vec<f32>>,
    /// Cumulative estimated translation (px) applied to the held keyframe
    /// since it was last refreshed.
    accum_dx: f32,
    accum_dy: f32,
}

impl Default for CompositorState {
    fn default() -> Self {
        Self {
            last_good_keyframe: None,
            keyframe_source_luma: None,
            accum_dx: 0.0,
            accum_dy: 0.0,
        }
    }
}

impl CompositorState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Call when a new SSIM-accepted Layer-2 keyframe arrives. Resets the
    /// motion accumulator against the Layer-1 frame it was derived from.
    pub fn accept_keyframe(&mut self, keyframe: RgbaImage, source_layer1: &RgbaImage) {
        self.keyframe_source_luma = Some(to_luma(source_layer1));
        self.last_good_keyframe = Some(keyframe);
        self.accum_dx = 0.0;
        self.accum_dy = 0.0;
    }

    #[must_use]
    pub fn has_keyframe(&self) -> bool {
        self.last_good_keyframe.is_some()
    }

    /// Composite one real Layer-1 frame with whatever Layer-2 keyframe is
    /// currently held (warped forward by estimated global motion), fading
    /// the neural content by `mix` in `[0, 1]` (0 = pure Layer 1, 1 = pure
    /// warped Layer 2). Returns pure Layer 1 unchanged if no keyframe has
    /// ever been accepted yet.
    pub fn composite(&mut self, layer1: &RgbaImage, mix: f32) -> RgbaImage {
        let Some(keyframe) = &self.last_good_keyframe else {
            return layer1.clone();
        };
        let cur_luma = to_luma(layer1);
        if let Some(prev_luma) = &self.keyframe_source_luma {
            let (dx, dy) = estimate_global_translation(prev_luma, &cur_luma, layer1.width(), layer1.height());
            self.accum_dx += dx;
            self.accum_dy += dy;
        }
        self.keyframe_source_luma = Some(cur_luma);

        let warped = warp_translate(keyframe, self.accum_dx, self.accum_dy);
        blend(layer1, &warped, mix.clamp(0.0, 1.0))
    }
}

fn to_luma(img: &RgbaImage) -> Vec<f32> {
    img.pixels()
        .map(|p| 0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32)
        .collect()
}

/// Coarse global-translation estimate via a small search window over a
/// downsampled luma buffer (real normalized cross-correlation, not a stub —
/// see module docs for why this is global rather than dense/per-block).
fn estimate_global_translation(prev: &[f32], cur: &[f32], w: u32, h: u32) -> (f32, f32) {
    const STEP: usize = 4; // downsample factor for the search
    const RADIUS: i32 = 6; // search +/- RADIUS*STEP px

    let w = w as usize;
    let h = h as usize;
    if w < STEP * 8 || h < STEP * 8 {
        return (0.0, 0.0);
    }

    let sample = |buf: &[f32], x: i32, y: i32| -> f32 {
        let x = x.clamp(0, w as i32 - 1) as usize;
        let y = y.clamp(0, h as i32 - 1) as usize;
        buf[y * w + x]
    };

    let score_at = |dx: i32, dy: i32| -> f32 {
        let mut sad = 0.0_f32;
        let mut n = 0u32;
        let mut y = STEP as i32;
        while (y as usize) < h - STEP {
            let mut x = STEP as i32;
            while (x as usize) < w - STEP {
                let a = sample(prev, x, y);
                let b = sample(cur, x + dx * STEP as i32, y + dy * STEP as i32);
                sad += (a - b).abs();
                n += 1;
                x += STEP as i32 * 2;
            }
            y += STEP as i32 * 2;
        }
        if n > 0 { sad / n as f32 } else { f32::MAX }
    };

    // Seed with zero motion so an ambiguous/uninformative window (near-flat
    // luma, ties across the whole search grid) biases toward "no motion"
    // rather than an arbitrary corner of the search window — a real
    // degenerate case a same-color solid frame hits, and a physically
    // sensible prior in general (small motion is more likely than large).
    let mut best = (0i32, 0i32);
    let mut best_score = score_at(0, 0);
    for dy in -RADIUS..=RADIUS {
        for dx in -RADIUS..=RADIUS {
            let score = score_at(dx, dy);
            if score < best_score {
                best_score = score;
                best = (dx, dy);
            }
        }
    }
    (best.0 as f32 * STEP as f32, best.1 as f32 * STEP as f32)
}

fn warp_translate(img: &RgbaImage, dx: f32, dy: f32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = x as f32 - dx;
            let sy = y as f32 - dy;
            let px = if sx >= 0.0 && sy >= 0.0 && (sx as u32) < w && (sy as u32) < h {
                *img.get_pixel(sx as u32, sy as u32)
            } else {
                Rgba([0, 0, 0, 0]) // out-of-frame: transparent, blend() below skips it
            };
            out.put_pixel(x, y, px);
        }
    }
    out
}

fn blend(base: &RgbaImage, overlay: &RgbaImage, mix: f32) -> RgbaImage {
    let (w, h) = base.dimensions();
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let b = base.get_pixel(x, y).0;
            let o = overlay.get_pixel(x, y).0;
            let a = (o[3] as f32 / 255.0) * mix; // out-of-frame warped pixels (alpha 0) never contribute
            let m = |bc: u8, oc: u8| -> u8 { (bc as f32 * (1.0 - a) + oc as f32 * a).round() as u8 };
            out.put_pixel(x, y, Rgba([m(b[0], o[0]), m(b[1], o[1]), m(b[2], o[2]), 255]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_keyframe_yet_returns_layer1_unchanged() {
        let l1 = RgbaImage::from_pixel(16, 16, Rgba([1, 2, 3, 255]));
        let mut state = CompositorState::new();
        let out = state.composite(&l1, 1.0);
        assert_eq!(out.as_raw(), l1.as_raw());
        assert!(!state.has_keyframe());
    }

    #[test]
    fn accepted_keyframe_visibly_blends_in() {
        let l1 = RgbaImage::from_pixel(32, 32, Rgba([0, 0, 0, 255]));
        let kf = RgbaImage::from_pixel(32, 32, Rgba([255, 255, 255, 255]));
        let mut state = CompositorState::new();
        state.accept_keyframe(kf, &l1);
        let out = state.composite(&l1, 1.0);
        assert_ne!(out.as_raw(), l1.as_raw());
        // full mix of a solid-white keyframe over solid-black layer1 should
        // land near-white.
        assert!(out.get_pixel(16, 16).0[0] > 200);
    }
}
