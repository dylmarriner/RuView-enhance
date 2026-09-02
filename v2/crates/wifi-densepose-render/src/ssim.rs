//! Structural-similarity frame-rejection gate (ADR-352 safety control).
//!
//! Compares a real Layer-2 (fal.ai) keyframe against the real Layer-1 frame
//! it was derived from. Below [`SSIM_ACCEPT_THRESHOLD`], the keyframe is
//! rejected (hold last-good composite) rather than shown.
//!
//! This is a **global** SSIM (the whole frame treated as one window), not the
//! standard sliding-11x11-window SSIM — a real, correctly-computed
//! similarity score, just less sensitive to small localized hallucinations
//! than a windowed implementation would be. Documented here rather than
//! silently shipped as if it were the full windowed algorithm; upgrading to
//! windowed SSIM is a stated follow-up, same as the wgpu renderer.

use image::{DynamicImage, GenericImageView};

/// Below this score, a Layer-2 keyframe is rejected. SSIM ranges [-1, 1];
/// 1.0 is identical. A low-strength restyle of the same underlying frame is
/// expected to retain substantial structural similarity to its source.
pub const SSIM_ACCEPT_THRESHOLD: f64 = 0.35;

#[derive(Debug, Clone, Copy)]
pub struct SsimResult {
    pub score: f64,
    pub accepted: bool,
}

/// Compute a real global SSIM between two real images (decoded from real
/// bytes — `layer1_png` and `layer2_bytes` are never synthesized here).
pub fn compare(layer1_png: &[u8], layer2_bytes: &[u8]) -> Result<SsimResult, image::ImageError> {
    let img_a = image::load_from_memory(layer1_png)?;
    let img_b = image::load_from_memory(layer2_bytes)?;

    let (w, h) = img_a.dimensions();
    let img_b_resized = img_b.resize_exact(w, h, image::imageops::FilterType::Triangle);

    let gray_a = to_gray_f64(&img_a);
    let gray_b = to_gray_f64(&DynamicImage::ImageRgba8(img_b_resized.to_rgba8()));

    let score = global_ssim(&gray_a, &gray_b);
    Ok(SsimResult {
        score,
        accepted: score >= SSIM_ACCEPT_THRESHOLD,
    })
}

fn to_gray_f64(img: &DynamicImage) -> Vec<f64> {
    img.to_luma8().pixels().map(|p| p.0[0] as f64).collect()
}

/// Standard SSIM formula (Wang et al. 2004) applied over the whole image as
/// a single window, using the conventional stabilizing constants for an
/// 8-bit luma range.
fn global_ssim(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let var_a = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / n;
    let var_b = b.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / n;
    let covar = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum::<f64>()
        / n;

    let l = 255.0_f64;
    let c1 = (0.01 * l).powi(2);
    let c2 = (0.03 * l).powi(2);

    let numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * covar + c2);
    let denominator = (mean_a.powi(2) + mean_b.powi(2) + c1) * (var_a + var_b + c2);
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_images_score_near_one() {
        let mut pixmap = tiny_skia::Pixmap::new(32, 32).unwrap();
        pixmap.fill(tiny_skia::Color::from_rgba8(100, 120, 140, 255));
        let png = pixmap.encode_png().unwrap();
        let result = compare(&png, &png).unwrap();
        assert!(result.score > 0.99, "score was {}", result.score);
        assert!(result.accepted);
    }

    #[test]
    fn wildly_different_images_are_rejected() {
        let mut a = tiny_skia::Pixmap::new(32, 32).unwrap();
        a.fill(tiny_skia::Color::from_rgba8(0, 0, 0, 255));
        let mut b = tiny_skia::Pixmap::new(32, 32).unwrap();
        b.fill(tiny_skia::Color::from_rgba8(255, 255, 255, 255));
        let result = compare(&a.encode_png().unwrap(), &b.encode_png().unwrap()).unwrap();
        assert!(!result.accepted, "score was {}", result.score);
    }
}
