//! Reduce the 15.5 GB CSI array to band features once, so every later fit is cheap.
//!
//! Reads `csi_windows.npy` by memory map (never loads it), writes
//! `fixtures/band_features.npy` -- (360000, 540) f32, ~777 MB.

use std::time::Instant;
use wiflow_harness::bands::{write_npy_f32, window_to_bands, BAND_DIM, NUM_CHANNELS, NUM_FRAMES};
use wiflow_harness::npy::Npy;

const DATASET: &str = "/home/ruvultra/wiflow-std-bench/preprocessed_csi_data";

fn main() {
    let fixtures = std::env::args()
        .nth(1)
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures").to_string());

    let t0 = Instant::now();
    let csi = Npy::open(format!("{DATASET}/csi_windows.npy")).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        csi.shape,
        vec![360000, NUM_CHANNELS, NUM_FRAMES],
        "unexpected CSI shape {:?}",
        csi.shape
    );
    let n = csi.shape[0];
    let data = csi.f32_slice().unwrap_or_else(|e| panic!("{e}"));
    let per_window = NUM_CHANNELS * NUM_FRAMES;

    let mut out = vec![0.0f32; n * BAND_DIM];
    let mut nonfinite_windows = 0usize;
    for i in 0..n {
        let src = &data[i * per_window..(i + 1) * per_window];
        let dst = &mut out[i * BAND_DIM..(i + 1) * BAND_DIM];
        window_to_bands(src, dst);
        if dst.iter().any(|v| !v.is_finite()) {
            nonfinite_windows += 1;
        }
        if i % 60_000 == 0 && i > 0 {
            eprintln!("  {i}/{n} windows ({:.0?})", t0.elapsed());
        }
    }

    let path = format!("{fixtures}/band_features.npy");
    write_npy_f32(&path, &out, n, BAND_DIM).unwrap_or_else(|e| panic!("{e}"));

    // Report, do not silently drop: non-finite band features should appear only in
    // the known-corrupt windows, which the masks exclude from every fit.
    eprintln!(
        "wrote {path}\n  windows: {n}, band dim: {BAND_DIM}\n  \
         windows with non-finite band features: {nonfinite_windows} \
         (expected to be a subset of the 9,072 masked corrupt windows)\n  \
         elapsed: {:.1?}",
        t0.elapsed()
    );
}
