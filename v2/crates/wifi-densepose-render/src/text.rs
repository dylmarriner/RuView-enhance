//! Self-contained bitmap-font text stamping — the burned-in "SYNTHETIC /
//! AI-ENHANCED" label ADR-352 requires on any frame that carries Layer-2
//! (fal.ai) content. Deliberately not a font-rendering dependency: a fixed
//! 5x7 glyph table for the small fixed character set this overlay needs
//! (uppercase letters, digits, space, hyphen, slash) is real, legible, and
//! avoids pulling in a font-shaping stack for one short label string.
//!
//! This is genuinely burned into the pixel buffer — not a UI overlay layer
//! that could be stripped by a downstream consumer — matching ADR-352's
//! requirement that AI-enhanced output is never presented as raw sensor
//! truth.

use image::{Rgba, RgbaImage};

const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;

/// 5x7 bitmap glyphs, row-major top-to-bottom, MSB-first per row (bit 4 =
/// leftmost column). Covers the fixed character set this overlay needs.
fn glyph(c: char) -> Option<[u8; GLYPH_H]> {
    Some(match c {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
        'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10000, 0b10011, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'N' => [0b10001, 0b11001, 0b10101, 0b10101, 0b10011, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        '0' => [0b01110, 0b10011, 0b10101, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        ' ' => [0; GLYPH_H],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        _ => return None,
    })
}

/// Stamp `text` (uppercase; unsupported characters render as a blank cell)
/// onto `img` at `(x0, y0)` with each glyph pixel scaled to `scale x scale`
/// device pixels, in `color` at `alpha` over a solid dark backing plate so
/// the label stays legible over any backdrop content.
pub fn stamp(img: &mut RgbaImage, x0: i64, y0: i64, text: &str, scale: i64, color: (u8, u8, u8), alpha: f32) {
    let (w, h) = img.dimensions();
    let plate_w = text.chars().count() as i64 * (GLYPH_W as i64 + 1) * scale;
    let plate_h = GLYPH_H as i64 * scale;
    blend_rect(img, x0 - scale, y0 - scale, plate_w + scale, plate_h + 2 * scale, (0, 0, 0), 0.55_f32);

    for (i, ch) in text.to_ascii_uppercase().chars().enumerate() {
        let Some(rows) = glyph(ch) else { continue };
        let gx = x0 + i as i64 * (GLYPH_W as i64 + 1) * scale;
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..GLYPH_W {
                let bit = (bits >> (GLYPH_W - 1 - col)) & 1;
                if bit == 0 {
                    continue;
                }
                let px = gx + col as i64 * scale;
                let py = y0 + row as i64 * scale;
                for dy in 0..scale {
                    for dx in 0..scale {
                        let x = px + dx;
                        let y = py + dy;
                        if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                            blend_pixel(img, x as u32, y as u32, color, alpha);
                        }
                    }
                }
            }
        }
    }
}

fn blend_rect(img: &mut RgbaImage, x0: i64, y0: i64, w: i64, h: i64, color: (u8, u8, u8), alpha: f32) {
    let (iw, ih) = img.dimensions();
    for y in y0.max(0)..(y0 + h).min(ih as i64) {
        for x in x0.max(0)..(x0 + w).min(iw as i64) {
            blend_pixel(img, x as u32, y as u32, color, alpha);
        }
    }
}

fn blend_pixel(img: &mut RgbaImage, x: u32, y: u32, color: (u8, u8, u8), alpha: f32) {
    let a = alpha.clamp(0.0, 1.0);
    let px = img.get_pixel_mut(x, y);
    let mix = |b: u8, s: u8| -> u8 { (b as f32 * (1.0 - a) + s as f32 * a).round() as u8 };
    *px = Rgba([mix(px.0[0], color.0), mix(px.0[1], color.1), mix(px.0[2], color.2), 255]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamping_synthetic_visibly_changes_the_frame() {
        let mut img = RgbaImage::from_pixel(200, 60, Rgba([10, 10, 10, 255]));
        let before = img.clone();
        stamp(&mut img, 4, 4, "SYNTHETIC", 2, (255, 255, 0), 1.0);
        assert_ne!(img.as_raw(), before.as_raw());
    }
}
