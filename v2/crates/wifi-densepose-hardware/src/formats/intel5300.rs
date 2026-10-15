//! Intel 5300 "Linux 802.11n CSI Tool" `.dat` reader.
//!
//! # File layout
//!
//! A log written by `log_to_file` is a sequence of records:
//!
//! ```text
//! u16 BE  field_len     bytes that follow (code byte + body)
//! u8      code          0xBB = beamforming (bfee) measurement; others skipped
//! body (field_len - 1 bytes), for code 0xBB:
//!   u32 LE timestamp_low   NIC clock, microseconds, wraps every ~71 min
//!   u16 LE bfee_count      driver packet counter
//!   u16     reserved
//!   u8 Nrx, u8 Ntx
//!   u8 rssi_a, rssi_b, rssi_c
//!   i8 noise, u8 agc, u8 antenna_sel
//!   u16 LE len             payload length, must equal (30*(Nrx*Ntx*16+3)+7)/8
//!   u16 LE fake_rate_n_flags
//!   payload[len]           30 subcarrier groups, each 3 bits of padding then
//!                          Nrx*Ntx complex values as bit-packed signed 8-bit
//!                          (imag, real are read as real first, then imag)
//! ```
//!
//! # Attribution
//!
//! The packed-bit extraction and the AGC/RSSI scaling follow the algorithm of
//! the original tool's MATLAB reference (`read_bfee`, `get_scaled_csi`) as
//! re-implemented in CSIKit (<https://github.com/Gi-z/CSIKit>, MIT licence,
//! `CSIKit/reader/readers/read_bfee.py`) and cross-checked with csiread
//! (<https://github.com/citysu/csiread>, MIT licence). This is an independent
//! Rust implementation with bounded reads.

use std::io::Read;

use chrono::{DateTime, Utc};

use super::util::{clamp_i8, device_time, read_body, read_full, skip, Fill};
use super::{FormatError, ReaderOptions};
use crate::csi_frame::{
    Adr018Flags, AntennaConfig, Bandwidth, CsiFrame, CsiMetadata, PpduType, SubcarrierData,
};

const FORMAT: &str = "intel5300";
/// Record code of a beamforming measurement.
pub const BFEE_CODE: u8 = 0xBB;
/// Fixed header length of a bfee body before the CSI payload.
pub const BFEE_HEADER_LEN: usize = 20;
/// Subcarrier groups reported by the 5300.
pub const NUM_SUBCARRIERS: usize = 30;

/// Reported subcarrier indices for a 20 MHz channel (802.11n grouping).
pub const SUBCARRIER_INDEX_20: [i16; 30] = [
    -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1, 1, 3, 5, 7, 9, 11, 13,
    15, 17, 19, 21, 23, 25, 27, 28,
];
/// Reported subcarrier indices for a 40 MHz channel.
pub const SUBCARRIER_INDEX_40: [i16; 30] = [
    -58, -54, -50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6, -2, 2, 6, 10, 14, 18, 22,
    26, 30, 34, 38, 42, 46, 50, 54, 58,
];

/// One decoded beamforming record.
#[derive(Debug, Clone, PartialEq)]
pub struct BfeeRecord {
    pub timestamp_low: u32,
    pub bfee_count: u16,
    pub n_rx: u8,
    pub n_tx: u8,
    pub rssi_a: u8,
    pub rssi_b: u8,
    pub rssi_c: u8,
    pub noise: i8,
    pub agc: u8,
    pub antenna_sel: u8,
    pub fake_rate_n_flags: u16,
    /// Raw signed 8-bit `(real, imag)` pairs laid out `[tx][rx][subcarrier]`:
    /// index `(tx * n_rx + rx) * 30 + sc`. The receive-antenna permutation
    /// from `antenna_sel` has been applied.
    pub csi: Vec<(i8, i8)>,
}

fn dbinv(x: f64) -> f64 {
    10f64.powf(x / 10.0)
}

impl BfeeRecord {
    /// Parse the body of a 0xBB record (the bytes after the code byte).
    pub fn parse(body: &[u8]) -> Result<Self, String> {
        if body.len() < BFEE_HEADER_LEN {
            return Err(format!(
                "bfee body is {} bytes, header needs {BFEE_HEADER_LEN}",
                body.len()
            ));
        }
        let timestamp_low = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
        let bfee_count = u16::from_le_bytes([body[4], body[5]]);
        let n_rx = body[8];
        let n_tx = body[9];
        if !(1..=3).contains(&n_rx) || !(1..=3).contains(&n_tx) {
            return Err(format!("invalid antenna counts Nrx={n_rx} Ntx={n_tx}"));
        }
        let (nrx, ntx) = (n_rx as usize, n_tx as usize);
        let calc_len = (NUM_SUBCARRIERS * (nrx * ntx * 16 + 3)).div_ceil(8);
        // Header byte map (offsets within body): 10..13 rssi a/b/c, 13 noise,
        // 14 agc, 15 antenna_sel, 16..18 len, 18..20 fake_rate_n_flags.
        let len_field = u16::from_le_bytes([body[16], body[17]]) as usize;
        let fake_rate_n_flags = u16::from_le_bytes([body[18], body[19]]);
        if len_field != calc_len {
            return Err(format!(
                "payload length field {len_field} != computed {calc_len} for Nrx={n_rx} Ntx={n_tx}"
            ));
        }
        let payload = &body[BFEE_HEADER_LEN..];
        if payload.len() < calc_len {
            return Err(format!(
                "payload is {} bytes, expected {calc_len}",
                payload.len()
            ));
        }

        let antenna_sel = body[15];
        let mut perm = [0usize, 1, 2];
        if n_rx == 3 {
            let p = [
                (antenna_sel & 0x3) as usize,
                ((antenna_sel >> 2) & 0x3) as usize,
                ((antenna_sel >> 4) & 0x3) as usize,
            ];
            // Only honour a genuine permutation of {0,1,2}.
            let mut seen = [false; 3];
            let valid = p.iter().all(|&x| {
                if x < 3 && !seen[x] {
                    seen[x] = true;
                    true
                } else {
                    false
                }
            });
            if valid {
                perm = p;
            }
        }

        let byte = |i: usize| -> u32 { payload.get(i).copied().unwrap_or(0) as u32 };
        let mut csi = vec![(0i8, 0i8); NUM_SUBCARRIERS * nrx * ntx];
        let mut index = 0usize;
        for sc in 0..NUM_SUBCARRIERS {
            index += 3;
            let rem = (index % 8) as u32;
            for j in 0..nrx {
                for k in 0..ntx {
                    let b = index / 8;
                    let real = (((byte(b) >> rem) | (byte(b + 1) << (8 - rem))) & 0xFF) as u8 as i8;
                    let imag =
                        (((byte(b + 1) >> rem) | (byte(b + 2) << (8 - rem))) & 0xFF) as u8 as i8;
                    csi[(k * nrx + perm[j]) * NUM_SUBCARRIERS + sc] = (real, imag);
                    index += 16;
                }
            }
        }

        Ok(Self {
            timestamp_low,
            bfee_count,
            n_rx,
            n_tx,
            rssi_a: body[10],
            rssi_b: body[11],
            rssi_c: body[12],
            noise: body[13] as i8,
            agc: body[14],
            antenna_sel,
            fake_rate_n_flags,
            csi,
        })
    }

    /// `true` when the iwlwifi rate flags mark a 40 MHz (HT40) frame.
    pub fn is_40mhz(&self) -> bool {
        self.fake_rate_n_flags & 0x0800 != 0
    }

    /// Raw `(real, imag)` for a subcarrier group / rx / tx antenna.
    pub fn csi_at(&self, sc: usize, rx: usize, tx: usize) -> Option<(i8, i8)> {
        if sc >= NUM_SUBCARRIERS || rx >= self.n_rx as usize || tx >= self.n_tx as usize {
            return None;
        }
        self.csi
            .get((tx * self.n_rx as usize + rx) * NUM_SUBCARRIERS + sc)
            .copied()
    }

    /// Total received signal strength in dBm (`get_total_rss.m`): the
    /// power-sum of the per-chain RSSI magnitudes minus the 44 dB offset and
    /// AGC. `None` if every chain reads zero.
    pub fn total_rss_dbm(&self) -> Option<f64> {
        let mag: f64 = [self.rssi_a, self.rssi_b, self.rssi_c]
            .iter()
            .filter(|&&r| r != 0)
            .map(|&r| dbinv(r as f64))
            .sum();
        if mag <= 0.0 {
            None
        } else {
            Some(10.0 * mag.log10() - 44.0 - self.agc as f64)
        }
    }

    /// Scale the raw CSI to absolute units, i.e. `sqrt(SNR)`-referenced
    /// complex gains (`get_scaled_csi.m`). Same `[tx][rx][sc]` layout as
    /// `csi`. Returns `None` if the CSI is all-zero or no RSSI is available.
    pub fn scaled_csi(&self) -> Option<Vec<(f64, f64)>> {
        let rss_db = self.total_rss_dbm()?;
        let csi_pwr: f64 = self
            .csi
            .iter()
            .map(|&(r, i)| (r as f64).powi(2) + (i as f64).powi(2))
            .sum();
        if csi_pwr <= 0.0 {
            return None;
        }
        let rssi_pwr = dbinv(rss_db);
        let scale = rssi_pwr / (csi_pwr / NUM_SUBCARRIERS as f64);
        // -127 means "noise unavailable" (monitor mode): assume -92 dBm.
        let noise_db = if self.noise == -127 {
            -92.0
        } else {
            self.noise as f64
        };
        let thermal = dbinv(noise_db);
        let quant = scale * (self.n_rx as f64 * self.n_tx as f64);
        let mut factor = (scale / (thermal + quant)).sqrt();
        match self.n_tx {
            2 => factor *= 2f64.sqrt(),
            // 4.5 dB is the approximation the original tool uses for 3 TX.
            3 => factor *= dbinv(4.5).sqrt(),
            _ => {}
        }
        Some(
            self.csi
                .iter()
                .map(|&(r, i)| (r as f64 * factor, i as f64 * factor))
                .collect(),
        )
    }

    /// Convert to a RuView [`CsiFrame`] using the raw (unscaled) integers.
    pub fn to_frame(&self, timestamp: DateTime<Utc>, channel_freq_mhz: u32) -> CsiFrame {
        let wide = self.is_40mhz();
        let table = if wide {
            &SUBCARRIER_INDEX_40
        } else {
            &SUBCARRIER_INDEX_20
        };
        let subcarriers = self
            .csi
            .iter()
            .enumerate()
            .map(|(n, &(r, i))| SubcarrierData {
                i: r as i16,
                q: i as i16,
                index: table[n % NUM_SUBCARRIERS],
            })
            .collect();
        let streams = self.n_rx as usize * self.n_tx as usize;
        CsiFrame {
            metadata: CsiMetadata {
                timestamp,
                node_id: 0,
                n_antennas: streams as u8,
                n_subcarriers: NUM_SUBCARRIERS as u16,
                channel_freq_mhz,
                rssi_dbm: self.total_rss_dbm().map(clamp_i8).unwrap_or(0),
                noise_floor_dbm: if self.noise == -127 { -92 } else { self.noise },
                bandwidth: if wide { Bandwidth::Bw40 } else { Bandwidth::Bw20 },
                antenna_config: AntennaConfig {
                    tx_antennas: self.n_tx,
                    rx_antennas: self.n_rx,
                },
                sequence: self.bfee_count as u32,
                ppdu_type: PpduType::HtLegacy,
                adr018_flags: Adr018Flags {
                    bw40: wide,
                    ..Adr018Flags::default()
                },
            },
            subcarriers,
        }
    }
}

/// Cheap structural check of the first record, used by format detection.
pub(crate) fn looks_like_intel5300(head: &[u8]) -> bool {
    if head.len() < 3 + BFEE_HEADER_LEN {
        return false;
    }
    let size = u16::from_be_bytes([head[0], head[1]]) as usize;
    if head[2] != BFEE_CODE || size < 1 + BFEE_HEADER_LEN {
        return false;
    }
    let end = (3 + (size - 1)).min(head.len());
    BfeeRecord::parse_header_only(&head[3..end]).is_some()
}

impl BfeeRecord {
    /// Header-only plausibility check (antenna counts and length field).
    fn parse_header_only(body: &[u8]) -> Option<()> {
        if body.len() < BFEE_HEADER_LEN {
            return None;
        }
        let (nrx, ntx) = (body[8] as usize, body[9] as usize);
        if !(1..=3).contains(&nrx) || !(1..=3).contains(&ntx) {
            return None;
        }
        let len = u16::from_le_bytes([body[16], body[17]]) as usize;
        (len == (NUM_SUBCARRIERS * (nrx * ntx * 16 + 3)).div_ceil(8)).then_some(())
    }
}

/// Streaming reader over a Linux 802.11n CSI Tool log.
///
/// Memory use is bounded by one record (<= 64 KiB). Iterating yields
/// `Result<CsiFrame, FormatError>`; use [`Intel5300Reader::next_record`] for
/// the raw [`BfeeRecord`] (needed for [`BfeeRecord::scaled_csi`]).
pub struct Intel5300Reader<R> {
    inner: R,
    offset: u64,
    buf: Vec<u8>,
    freq_mhz: u32,
    last_ts_low: Option<u32>,
    wraps: u64,
    done: bool,
}

impl<R: Read> Intel5300Reader<R> {
    pub fn new(inner: R, opts: &ReaderOptions) -> Self {
        Self {
            inner,
            offset: 0,
            buf: Vec::new(),
            freq_mhz: opts.channel_freq_mhz.unwrap_or(0),
            last_ts_low: None,
            wraps: 0,
            done: false,
        }
    }

    /// Unwrap the 32-bit microsecond NIC clock into a monotonic 64-bit value.
    fn unwrap_ts(&mut self, ts_low: u32) -> u64 {
        if let Some(last) = self.last_ts_low {
            // A large backwards jump is a counter wrap, a small one is reordering.
            if last > ts_low && last - ts_low > 0x8000_0000 {
                self.wraps += 1;
            }
        }
        self.last_ts_low = Some(ts_low);
        (self.wraps << 32) | ts_low as u64
    }

    /// Next raw record, skipping non-bfee records. `Ok(None)` at clean EOF.
    pub fn next_record(&mut self) -> Result<Option<(BfeeRecord, DateTime<Utc>)>, FormatError> {
        if self.done {
            return Ok(None);
        }
        loop {
            let start = self.offset;
            let mut hdr = [0u8; 3];
            match read_full(&mut self.inner, &mut hdr)? {
                Fill::Eof => {
                    self.done = true;
                    return Ok(None);
                }
                Fill::Partial(_) => {
                    self.done = true;
                    return Err(FormatError::Truncated {
                        format: FORMAT,
                        offset: start,
                    });
                }
                Fill::Full => {}
            }
            let field_len = u16::from_be_bytes([hdr[0], hdr[1]]) as usize;
            let code = hdr[2];
            if field_len == 0 {
                // No code byte can exist in a zero-length record; we only
                // consumed 2 bytes of it legitimately, so treat as corrupt.
                self.offset += 3;
                return Err(FormatError::Corrupt {
                    format: FORMAT,
                    offset: start,
                    reason: "zero-length record".into(),
                });
            }
            let body_len = field_len - 1;
            self.offset += 3;
            if code != BFEE_CODE {
                let skipped = skip(&mut self.inner, body_len as u64)?;
                self.offset += skipped;
                if skipped < body_len as u64 {
                    self.done = true;
                    return Err(FormatError::Truncated {
                        format: FORMAT,
                        offset: start,
                    });
                }
                continue;
            }
            match read_body(&mut self.inner, &mut self.buf, body_len)? {
                Fill::Full => {}
                _ => {
                    self.done = true;
                    return Err(FormatError::Truncated {
                        format: FORMAT,
                        offset: start,
                    });
                }
            }
            self.offset += body_len as u64;
            let rec = BfeeRecord::parse(&self.buf).map_err(|reason| FormatError::Corrupt {
                format: FORMAT,
                offset: start,
                reason,
            })?;
            let ts = device_time(self.unwrap_ts(rec.timestamp_low));
            return Ok(Some((rec, ts)));
        }
    }
}

impl<R: Read> Iterator for Intel5300Reader<R> {
    type Item = Result<CsiFrame, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_record() {
            Ok(Some((rec, ts))) => Some(Ok(rec.to_frame(ts, self.freq_mhz))),
            Ok(None) => None,
            Err(e) => {
                if e.is_fatal() {
                    self.done = true;
                }
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::*;

    /// Pack `(real, imag)` values (`[tx][rx][sc]`, no permutation) into a
    /// complete log record (length prefix + code + header + payload).
    pub fn encode_record(
        ts_low: u32,
        count: u16,
        n_rx: u8,
        n_tx: u8,
        rssi: [u8; 3],
        noise: i8,
        agc: u8,
        antenna_sel: u8,
        rate: u16,
        csi: &[(i8, i8)],
    ) -> Vec<u8> {
        let (nrx, ntx) = (n_rx as usize, n_tx as usize);
        let calc = (30 * (nrx * ntx * 16 + 3)).div_ceil(8);
        let mut payload = vec![0u8; calc + 3];
        let mut index = 0usize;
        for sc in 0..30 {
            index += 3;
            for j in 0..nrx {
                for k in 0..ntx {
                    let (r, i) = csi[(k * nrx + j) * 30 + sc];
                    for (bitpos, v) in [(index, r as u8), (index + 8, i as u8)] {
                        for b in 0..8 {
                            if v >> b & 1 == 1 {
                                let p = bitpos + b;
                                payload[p / 8] |= 1 << (p % 8);
                            }
                        }
                    }
                    index += 16;
                }
            }
        }
        payload.truncate(calc);
        let mut body = Vec::new();
        body.extend_from_slice(&ts_low.to_le_bytes());
        body.extend_from_slice(&count.to_le_bytes());
        body.extend_from_slice(&[0, 0, n_rx, n_tx, rssi[0], rssi[1], rssi[2], noise as u8, agc]);
        body.push(antenna_sel);
        body.extend_from_slice(&(calc as u16).to_le_bytes());
        body.extend_from_slice(&rate.to_le_bytes());
        body.extend_from_slice(&payload);
        let mut rec = Vec::new();
        rec.extend_from_slice(&((body.len() + 1) as u16).to_be_bytes());
        rec.push(BFEE_CODE);
        rec.extend_from_slice(&body);
        rec
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::encode_record;
    use super::*;

    fn opts() -> ReaderOptions {
        ReaderOptions::default()
    }

    /// Hand-assembled payload for Nrx=Ntx=1 (len 72). First entry
    /// real=+5 imag=-3, second entry real=-128 imag=127, rest zero. The bytes
    /// were derived by hand from the bit layout (3 pad bits, then 8-bit real,
    /// 8-bit imag, LSB first), independent of the encoder helper.
    #[test]
    fn hand_computed_payload_bits() {
        let mut body = vec![0u8; 20 + 72];
        body[8] = 1;
        body[9] = 1;
        body[16] = 72;
        body[18] = 0;
        body[20] = 0x28;
        body[21] = 0xE8;
        body[22] = 0x07;
        body[23] = 0xE0;
        body[24] = 0x1F;
        let rec = BfeeRecord::parse(&body).unwrap();
        assert_eq!(rec.csi_at(0, 0, 0), Some((5, -3)));
        assert_eq!(rec.csi_at(1, 0, 0), Some((-128, 127)));
        assert_eq!(rec.csi_at(2, 0, 0), Some((0, 0)));
        assert_eq!(rec.csi.len(), 30);
    }

    #[test]
    fn roundtrip_3x2_with_permutation() {
        let (nrx, ntx) = (3usize, 2usize);
        let csi: Vec<(i8, i8)> = (0..30 * nrx * ntx)
            .map(|n| ((n as i32 * 7 % 251 - 125) as i8, (n as i32 * 13 % 241 - 120) as i8))
            .collect();
        // antenna_sel: chain 0->2, 1->0, 2->1  (0b01_00_10 = 0x12)
        let sel = 0b01_00_10u8;
        let rec = encode_record(1000, 7, 3, 2, [40, 41, 42], -95, 30, sel, 0x0800, &csi);
        let r = BfeeRecord::parse(&rec[3..]).unwrap();
        let perm = [2usize, 0, 1];
        for tx in 0..ntx {
            for rx in 0..nrx {
                for sc in 0..30 {
                    let src = csi[(tx * nrx + rx) * 30 + sc];
                    assert_eq!(r.csi_at(sc, perm[rx], tx), Some(src));
                }
            }
        }
        assert!(r.is_40mhz());
        assert_eq!(r.bfee_count, 7);
        assert_eq!(r.noise, -95);
    }

    #[test]
    fn invalid_permutation_falls_back_to_identity() {
        let csi: Vec<(i8, i8)> = (0..90).map(|n| (n as i8, -(n as i8))).collect();
        let rec = encode_record(1, 1, 3, 1, [1, 1, 1], -90, 0, 0b00_00_00, 0, &csi);
        let r = BfeeRecord::parse(&rec[3..]).unwrap();
        assert_eq!(r.csi_at(4, 1, 0), Some(csi[30 + 4]));
    }

    #[test]
    fn reader_streams_and_skips_foreign_records() {
        let csi: Vec<(i8, i8)> = (0..30).map(|n| (n as i8, 1)).collect();
        let mut file = Vec::new();
        file.extend(encode_record(10, 1, 1, 1, [30, 0, 0], -92, 10, 0, 0, &csi));
        // A foreign record (code 0xC1, 5-byte body).
        file.extend_from_slice(&[0, 6, 0xC1, 1, 2, 3, 4, 5]);
        file.extend(encode_record(20, 2, 1, 1, [30, 0, 0], -92, 10, 0, 0, &csi));
        let frames: Vec<_> = Intel5300Reader::new(&file[..], &opts())
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(frames.len(), 2);
        let f = &frames[0];
        assert_eq!(f.subcarrier_count(), 30);
        assert_eq!(f.subcarriers[3].i, 3);
        assert_eq!(f.subcarriers[3].q, 1);
        assert_eq!(f.subcarriers[0].index, -28);
        assert_eq!(f.subcarriers[29].index, 28);
        assert_eq!(f.metadata.sequence, 1);
        assert_eq!(frames[1].metadata.sequence, 2);
        assert_eq!(f.metadata.timestamp.timestamp_micros(), 10);
        assert_eq!(f.metadata.n_antennas, 1);
        assert_eq!(f.metadata.noise_floor_dbm, -92);
    }

    #[test]
    fn timestamp_wrap_is_unwrapped() {
        let csi = vec![(1i8, 1i8); 30];
        let mut file = encode_record(0xFFFF_FFF0, 1, 1, 1, [30, 0, 0], -92, 10, 0, 0, &csi);
        file.extend(encode_record(0x10, 2, 1, 1, [30, 0, 0], -92, 10, 0, 0, &csi));
        let v: Vec<_> = Intel5300Reader::new(&file[..], &opts())
            .collect::<Result<_, _>>()
            .unwrap();
        let t0 = v[0].metadata.timestamp.timestamp_micros();
        let t1 = v[1].metadata.timestamp.timestamp_micros();
        assert_eq!(t1 - t0, 0x20);
    }

    #[test]
    fn bad_length_field_is_recoverable_and_next_record_still_read() {
        let csi = vec![(1i8, 1i8); 30];
        let mut bad = encode_record(1, 1, 1, 1, [30, 0, 0], -92, 10, 0, 0, &csi);
        bad[3 + 16] ^= 0xFF; // corrupt `len`
        let good = encode_record(2, 2, 1, 1, [30, 0, 0], -92, 10, 0, 0, &csi);
        let mut file = bad;
        file.extend(good);
        let mut it = Intel5300Reader::new(&file[..], &opts());
        let first = it.next().unwrap();
        assert!(matches!(first, Err(FormatError::Corrupt { .. })));
        assert!(!first.unwrap_err().is_fatal());
        assert_eq!(it.next().unwrap().unwrap().metadata.sequence, 2);
        assert!(it.next().is_none());
    }

    #[test]
    fn truncated_tail_reports_truncated() {
        let csi = vec![(1i8, 1i8); 30];
        let mut file = encode_record(1, 1, 1, 1, [30, 0, 0], -92, 10, 0, 0, &csi);
        file.truncate(file.len() - 5);
        let mut it = Intel5300Reader::new(&file[..], &opts());
        assert!(matches!(
            it.next().unwrap(),
            Err(FormatError::Truncated { .. })
        ));
        assert!(it.next().is_none());
    }

    #[test]
    fn oversized_declared_length_does_not_allocate_beyond_stream() {
        // Declares a 65535-byte record but supplies 4 bytes; must truncate
        // cleanly (allocation is capped by the u16 field anyway).
        let file = [0xFF, 0xFF, BFEE_CODE, 1, 2, 3, 4];
        let mut it = Intel5300Reader::new(&file[..], &opts());
        assert!(matches!(
            it.next().unwrap(),
            Err(FormatError::Truncated { .. })
        ));
    }

    /// Independent numeric reference computed with Python/`math`
    /// (rssi_a=50, agc=40, noise=-92, 1x1, all subcarriers (3,4)):
    /// total RSS = -34 dBm, factor = 0.99998019.
    #[test]
    fn scaling_matches_independent_reference() {
        let csi = vec![(3i8, 4i8); 30];
        let rec = encode_record(1, 1, 1, 1, [50, 0, 0], -92, 40, 0, 0, &csi);
        let r = BfeeRecord::parse(&rec[3..]).unwrap();
        assert!((r.total_rss_dbm().unwrap() - (-34.0)).abs() < 1e-9);
        let s = r.scaled_csi().unwrap();
        assert_eq!(s.len(), 30);
        assert!((s[0].0 - 3.0 * 0.9999801894237982).abs() < 1e-9);
        assert!((s[0].1 - 4.0 * 0.9999801894237982).abs() < 1e-9);
    }

    #[test]
    fn scaling_none_without_rssi_or_signal() {
        let csi = vec![(3i8, 4i8); 30];
        let rec = encode_record(1, 1, 1, 1, [0, 0, 0], -92, 40, 0, 0, &csi);
        assert!(BfeeRecord::parse(&rec[3..]).unwrap().scaled_csi().is_none());
        let zero = vec![(0i8, 0i8); 30];
        let rec = encode_record(1, 1, 1, 1, [50, 0, 0], -92, 40, 0, 0, &zero);
        assert!(BfeeRecord::parse(&rec[3..]).unwrap().scaled_csi().is_none());
    }

    #[test]
    fn detection_accepts_valid_and_rejects_noise() {
        let csi = vec![(1i8, 1i8); 120];
        let rec = encode_record(1, 1, 2, 2, [30, 0, 0], -92, 10, 0, 0, &csi);
        assert!(looks_like_intel5300(&rec));
        assert!(!looks_like_intel5300(&[0u8; 64]));
        assert!(!looks_like_intel5300(&rec[..10]));
    }
}
