//! Radio tomographic imaging (RTI) over node-to-node links.
//!
//! Implementation of the elliptical weight model from Wilson & Patwari,
//! "Radio Tomographic Imaging with Wireless Networks" (IEEE TMC 2010), from
//! the published method; no code is copied. A person perturbs links whose
//! first Fresnel zone they stand in. Each link spreads its perturbation over the
//! ellipse { p : |p - tx| + |p - rx| < |tx - rx| + LAMBDA }, scaled by
//! 1/sqrt(link length) so long links (which pass more cells) do not dominate.
//! Summing over links gives a 2-D activity image; its peak is the estimate.
//!
//! Positions are the operator's `--node-positions` (metres). The image's
//! accuracy is bounded by those positions and by link geometry: with 4 nodes,
//! treat the peak as "which part of the room", not a precise coordinate.

/// Excess path length defining a link's sensitive ellipse (m). Wilson &
/// Patwari use ~0.02-0.2 m at 2.4 GHz; wider tolerates position error.
pub const LAMBDA_M: f64 = 0.3;
/// Grid cell size (m).
pub const CELL_M: f64 = 0.25;
/// Margin around the node bounding box (m).
pub const MARGIN_M: f64 = 0.5;

#[derive(Debug, Clone, PartialEq)]
pub struct ActivityMap {
    pub origin: [f64; 2],
    pub cell_m: f64,
    pub width: usize,
    pub height: usize,
    /// Row-major, `height` rows of `width` cells, normalised so the max is 1.
    pub cells: Vec<f64>,
    /// Centre of the strongest cell, if any link carried weight.
    pub peak: Option<[f64; 2]>,
}

/// Build the map. `links` holds (tx position, rx position, perturbation >= 0).
pub fn build(nodes: &[[f64; 2]], links: &[([f64; 2], [f64; 2], f64)]) -> Option<ActivityMap> {
    if nodes.len() < 2 {
        return None;
    }
    let min_x = nodes.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min) - MARGIN_M;
    let min_y = nodes.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min) - MARGIN_M;
    let max_x = nodes.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max) + MARGIN_M;
    let max_y = nodes.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max) + MARGIN_M;
    let width = (((max_x - min_x) / CELL_M).ceil() as usize).clamp(1, 400);
    let height = (((max_y - min_y) / CELL_M).ceil() as usize).clamp(1, 400);
    let mut cells = vec![0.0; width * height];
    let dist = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();

    for &(tx, rx, w) in links {
        if !(w.is_finite() && w > 0.0) {
            continue;
        }
        let d = dist(tx, rx);
        if d < 1e-3 {
            continue;
        }
        let scale = w / d.sqrt();
        for row in 0..height {
            for col in 0..width {
                let c = [min_x + (col as f64 + 0.5) * CELL_M, min_y + (row as f64 + 0.5) * CELL_M];
                if dist(c, tx) + dist(c, rx) < d + LAMBDA_M {
                    cells[row * width + col] += scale;
                }
            }
        }
    }
    let max = cells.iter().copied().fold(0.0, f64::max);
    // Overlapping ellipses produce plateaus of tied cells; the centroid of the
    // top plateau is the estimate, not its first (corner) cell.
    let peak = (max > 0.0).then(|| {
        let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
        for (i, &v) in cells.iter().enumerate() {
            if v >= max * (1.0 - 1e-9) {
                sx += min_x + ((i % width) as f64 + 0.5) * CELL_M;
                sy += min_y + ((i / width) as f64 + 0.5) * CELL_M;
                n += 1.0;
            }
        }
        [sx / n, sy / n]
    });
    if max > 0.0 {
        cells.iter_mut().for_each(|v| *v /= max);
    }
    Some(ActivityMap { origin: [min_x, min_y], cell_m: CELL_M, width, height, cells, peak })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NODES: [[f64; 2]; 4] = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];

    #[test]
    fn crossing_perturbed_links_localise_their_intersection() {
        // Diagonals 0-2 and 1-3 cross at (2, 2); perimeter links are quiet.
        let links = vec![
            (NODES[0], NODES[2], 1.0),
            (NODES[1], NODES[3], 1.0),
            (NODES[0], NODES[1], 0.0),
            (NODES[2], NODES[3], 0.0),
        ];
        let m = build(&NODES, &links).unwrap();
        let p = m.peak.unwrap();
        assert!((p[0] - 2.0).abs() <= 0.5 && (p[1] - 2.0).abs() <= 0.5, "peak {p:?}");
        assert!((m.cells.iter().copied().fold(0.0, f64::max) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn quiet_links_give_no_peak() {
        let links = vec![(NODES[0], NODES[2], 0.0), (NODES[1], NODES[3], f64::NAN)];
        let m = build(&NODES, &links).unwrap();
        assert!(m.peak.is_none());
        assert!(m.cells.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn needs_two_nodes() {
        assert!(build(&[[0.0, 0.0]], &[]).is_none());
    }
}
