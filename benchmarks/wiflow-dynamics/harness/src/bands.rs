//! Band representation of a CSI window, and a minimal `.npy` writer.
//!
//! ## Representation, and the pre-registered choice it replaces (MEASURED)
//!
//! STATE.md pre-registered "27 bands of 20 channels, keeping all 20 frames", on the
//! reasoning that collapsing time would destroy the motion signal. `diag_repr`
//! measured that reasoning to be **backwards**. Holding the feature dimension fixed
//! at 540 and trading channel resolution against time resolution:
//!
//! | representation            | same-timestep pose PCK@20 | vs mean-pose bar |
//! |---------------------------|---------------------------|------------------|
//! | 27 bands x 20 frames      | 0.7495                    | **-0.0029**      |
//! | 54 x 10                   | 0.7668                    | +0.0145          |
//! | 108 x 5                   | 0.7782                    | +0.0259          |
//! | 135 x 4                   | 0.7942                    | +0.0419          |
//! | 270 x 2                   | 0.8127                    | +0.0603          |
//! | 540 x 1                   | 0.8150                    | +0.0627          |
//! | 270 bands, mean over time | 0.8132 (at HALF the dim)  | +0.0609          |
//!
//! The pre-registered choice was the worst of the set and the only one that fails to
//! beat the bar. Pose information lives in fine channel structure; averaging 20
//! adjacent channels destroys it, while averaging over the 20 frames costs ~nothing.
//!
//! So: `z` is **270 bands of 2 channels, averaged over all 20 frames** -- best NLL of
//! any variant tested, at half the dimension.
//!
//! ## Feature granularity vs ACTION granularity
//!
//! These are independent, and conflating them was the error above. Features stay fine
//! (270 bands) to preserve signal; the *action* is coarse -- one probe reveals a
//! contiguous group of 10 bands = 20 channels, a physically meaningful subcarrier
//! group. 27 probe groups keeps greedy selection tractable.

use std::io::Write;

pub const NUM_CHANNELS: usize = 540;
pub const NUM_FRAMES: usize = 20;

/// Feature granularity: 270 bands of 2 channels, averaged over all frames.
pub const NUM_BANDS: usize = 270;
pub const CHANNELS_PER_BAND: usize = NUM_CHANNELS / NUM_BANDS; // 2
/// One scalar per band (time is averaged out -- measured to be nearly free).
pub const BAND_DIM: usize = NUM_BANDS;

/// Action granularity: one probe reveals a contiguous group of bands.
pub const NUM_PROBE_GROUPS: usize = 27;
pub const BANDS_PER_GROUP: usize = NUM_BANDS / NUM_PROBE_GROUPS; // 10 bands = 20 channels

/// Feature dims revealed by probe group `g`.
pub fn group_dims(g: usize) -> std::ops::Range<usize> {
    (g * BANDS_PER_GROUP)..((g + 1) * BANDS_PER_GROUP)
}

/// Reduce one raw CSI window `[540 channels][20 frames]` to `[270 bands]`.
///
/// Accumulates in f64: a corrupted window can carry amplitudes to 3.4e38, and f32
/// summation there would overflow to inf and silently poison downstream statistics.
/// Corrupted windows are excluded upstream by the masks, but a reduction that
/// overflows *before* the mask is applied would be a trap.
pub fn window_to_bands(csi: &[f32], out: &mut [f32]) {
    debug_assert_eq!(csi.len(), NUM_CHANNELS * NUM_FRAMES);
    debug_assert_eq!(out.len(), BAND_DIM);
    for b in 0..NUM_BANDS {
        let mut acc = 0.0f64;
        for c in 0..CHANNELS_PER_BAND {
            for f in 0..NUM_FRAMES {
                acc += csi[(b * CHANNELS_PER_BAND + c) * NUM_FRAMES + f] as f64;
            }
        }
        out[b] = (acc / (CHANNELS_PER_BAND * NUM_FRAMES) as f64) as f32;
    }
}

/// Write a 2-D f32 array as a version-1.0 `.npy`.
pub fn write_npy_f32(path: &str, data: &[f32], rows: usize, cols: usize) -> Result<(), String> {
    assert_eq!(data.len(), rows * cols);
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // Header + magic(6) + version(2) + len(2) must be a multiple of 64.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');

    let f = std::fs::File::create(path).map_err(|e| format!("create {path}: {e}"))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
    w.write_all(b"\x93NUMPY\x01\x00").map_err(|e| e.to_string())?;
    w.write_all(&(header.len() as u16).to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(header.as_bytes()).map_err(|e| e.to_string())?;
    for chunk in data.chunks(1 << 16) {
        let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_all(&bytes).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}
