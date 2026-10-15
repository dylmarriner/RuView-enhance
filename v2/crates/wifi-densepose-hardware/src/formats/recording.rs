//! Emit RuView's existing `.csi.jsonl` recording format.
//!
//! One JSON object per line, identical in shape to what
//! `wifi-densepose-sensing-server`'s `recording.rs` (`RecordedFrame`) and
//! `scripts/collect-training-data.py` (`CsiRecorder.write_frame`) write:
//!
//! ```json
//! {"timestamp": 1700000000.25, "subcarriers": [amp, ...], "rssi": -55.0,
//!  "noise_floor": -92.0, "features": {"node_id": 0, "n_antennas": 1, ...}}
//! ```
//!
//! * `subcarriers` holds per-subcarrier **amplitudes** (`sqrt(I^2+Q^2)`),
//!   stream-major (`n_antennas * n_subcarriers` values), exactly as the
//!   raw-CSI path of `collect-training-data.py` does.
//! * `features` is the free-form object the Python recorder fills with every
//!   other frame field; this module adds `n_subcarriers`, `freq_mhz`,
//!   `channel`, `seq`, `ppdu_type`, `flags`, `bandwidth_mhz`, `antenna_tx`,
//!   `antenna_rx`, `source_format` and (unless disabled) a `phase` array
//!   (radians, same length as `subcarriers`). Readers that only know the
//!   four top-level fields are unaffected.

use std::io::Write;

use serde_json::{json, Value};

use super::{freq_mhz_to_channel, CsiFormat};
use crate::csi_frame::{Bandwidth, CsiFrame};

/// Options for [`frame_to_record`].
#[derive(Debug, Clone)]
pub struct RecordingOptions {
    /// Include a `features.phase` array.
    pub include_phase: bool,
    /// Seconds added to the frame timestamp. Use it to anchor device-clock
    /// timestamps (Intel 5300, Atheros, ESP32) to a wall-clock start time.
    pub time_offset_secs: f64,
}

impl Default for RecordingOptions {
    fn default() -> Self {
        Self {
            include_phase: true,
            time_offset_secs: 0.0,
        }
    }
}

fn bandwidth_mhz(b: Bandwidth) -> u32 {
    match b {
        Bandwidth::Bw20 => 20,
        Bandwidth::Bw40 => 40,
        Bandwidth::Bw80 => 80,
        Bandwidth::Bw160 => 160,
    }
}

/// Build the JSON record for one frame. `amp_phase` overrides the
/// amplitude/phase derived from the integer I/Q (used for scaled Intel 5300
/// output); it must have the same length as `frame.subcarriers`.
pub fn frame_to_record(
    frame: &CsiFrame,
    source: CsiFormat,
    opts: &RecordingOptions,
    amp_phase: Option<(Vec<f64>, Vec<f64>)>,
) -> Value {
    let m = &frame.metadata;
    let (amp, phase) = match amp_phase {
        Some((a, p)) if a.len() == frame.subcarriers.len() && p.len() == a.len() => (a, p),
        _ => frame.to_amplitude_phase(),
    };
    let ts = m.timestamp.timestamp() as f64
        + m.timestamp.timestamp_subsec_nanos() as f64 / 1e9
        + opts.time_offset_secs;
    let mut features = json!({
        "node_id": m.node_id,
        "n_antennas": m.n_antennas,
        "n_subcarriers": m.n_subcarriers,
        "freq_mhz": m.channel_freq_mhz,
        "channel": freq_mhz_to_channel(m.channel_freq_mhz),
        "seq": m.sequence,
        "ppdu_type": m.ppdu_type.to_byte(),
        "flags": m.adr018_flags.to_byte(),
        "bandwidth_mhz": bandwidth_mhz(m.bandwidth),
        "antenna_tx": m.antenna_config.tx_antennas,
        "antenna_rx": m.antenna_config.rx_antennas,
        "source_format": source.id(),
    });
    if opts.include_phase {
        features["phase"] = json!(phase);
    }
    // Non-finite values are not representable in JSON; map to 0.
    let amp: Vec<f64> = amp
        .into_iter()
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect();
    json!({
        "timestamp": ts,
        "subcarriers": amp,
        "rssi": m.rssi_dbm as f64,
        "noise_floor": m.noise_floor_dbm as f64,
        "features": features,
    })
}

/// Write one frame as a JSONL line.
pub fn write_frame<W: Write>(
    out: &mut W,
    frame: &CsiFrame,
    source: CsiFormat,
    opts: &RecordingOptions,
    amp_phase: Option<(Vec<f64>, Vec<f64>)>,
) -> std::io::Result<()> {
    let v = frame_to_record(frame, source, opts, amp_phase);
    serde_json::to_writer(&mut *out, &v).map_err(std::io::Error::other)?;
    out.write_all(b"\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csi_frame::{
        Adr018Flags, AntennaConfig, CsiMetadata, PpduType, SubcarrierData,
    };
    use chrono::DateTime;

    fn frame() -> CsiFrame {
        CsiFrame {
            metadata: CsiMetadata {
                timestamp: DateTime::from_timestamp(1_700_000_000, 250_000_000).unwrap(),
                node_id: 3,
                n_antennas: 1,
                n_subcarriers: 2,
                channel_freq_mhz: 2437,
                rssi_dbm: -55,
                noise_floor_dbm: -92,
                bandwidth: Bandwidth::Bw20,
                antenna_config: AntennaConfig::default(),
                sequence: 9,
                ppdu_type: PpduType::HtLegacy,
                adr018_flags: Adr018Flags::default(),
            },
            subcarriers: vec![
                SubcarrierData { i: 3, q: 4, index: -1 },
                SubcarrierData { i: 0, q: 0, index: 1 },
            ],
        }
    }

    #[test]
    fn matches_recorded_frame_schema() {
        let v = frame_to_record(&frame(), CsiFormat::Esp32Csv, &RecordingOptions::default(), None);
        assert_eq!(v["timestamp"].as_f64().unwrap(), 1_700_000_000.25);
        assert_eq!(v["subcarriers"], json!([5.0, 0.0]));
        assert_eq!(v["rssi"].as_f64().unwrap(), -55.0);
        assert_eq!(v["noise_floor"].as_f64().unwrap(), -92.0);
        assert_eq!(v["features"]["channel"], 6);
        assert_eq!(v["features"]["source_format"], "esp32-csv");
        assert_eq!(v["features"]["phase"].as_array().unwrap().len(), 2);
        // Round-trips into the sensing server's RecordedFrame field set.
        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        struct RecordedFrame {
            timestamp: f64,
            subcarriers: Vec<f64>,
            rssi: f64,
            noise_floor: f64,
            features: serde_json::Value,
        }
        let line = serde_json::to_string(&v).unwrap();
        let rf: RecordedFrame = serde_json::from_str(&line).unwrap();
        assert_eq!(rf.subcarriers.len(), 2);
    }

    #[test]
    fn offset_override_and_no_phase() {
        let opts = RecordingOptions {
            include_phase: false,
            time_offset_secs: 10.0,
        };
        let v = frame_to_record(
            &frame(),
            CsiFormat::Intel5300,
            &opts,
            Some((vec![1.0, f64::NAN], vec![0.0, 0.0])),
        );
        assert_eq!(v["timestamp"].as_f64().unwrap(), 1_700_000_010.25);
        assert!(v["features"].get("phase").is_none());
        assert_eq!(v["subcarriers"], json!([1.0, 0.0]));
    }

    #[test]
    fn write_frame_emits_one_line() {
        let mut out = Vec::new();
        write_frame(&mut out, &frame(), CsiFormat::Atheros, &RecordingOptions::default(), None)
            .unwrap();
        assert_eq!(out.iter().filter(|&&b| b == b'\n').count(), 1);
        assert!(out.ends_with(b"\n"));
    }
}
