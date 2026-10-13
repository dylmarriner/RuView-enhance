//! Atheros CSI Tool (`recv_csi` / `log_to_file`) `.dat` reader.
//!
//! # File layout
//!
//! ```text
//! [optional single byte 0xFF  => all multi-byte fields are big-endian]
//! repeated:
//!   u16 field_len        bytes in the rest of the record
//!   25-byte header:
//!     u64 timestamp      TSF, microseconds
//!     u16 csi_len        bytes of packed CSI
//!     u16 tx_channel     centre frequency (MHz) or channel number
//!     u8  err_info, noise_floor (i8), rate, bandwidth (0=20, 1=40 MHz),
//!         num_tones, nr, nc, rssi, rssi_1, rssi_2, rssi_3
//!     u16 payload_len    bytes of the received frame payload that follow
//!   csi_len bytes        10-bit packed signed I/Q (see below)
//!   payload_len bytes    ignored
//! ```
//!
//! CSI is a little-endian bit stream (bit 0 of byte 0 first) of 10-bit
//! two's-complement values: for each tone, for each `nc` (TX) index, for each
//! `nr` (RX) index: **imag** then **real**.
//!
//! # Attribution
//!
//! Field layout and the 10-bit unpacking follow CSIKit
//! (<https://github.com/Gi-z/CSIKit>, MIT, `read_atheros.py`) and csiread
//! (<https://github.com/citysu/csiread>, MIT, `_csiread.pyx`, class `Atheros`).
//! The two differ on (a) the antenna loop order (csiread: `nr` outer, CSIKit:
//! `nc` outer; CSIKit's order, which mirrors the original tool, is used here)
//! and (b) whether `+1` is added after sign conversion (CSIKit yes, csiread no;
//! default off, see [`ReaderOptions::atheros_plus_one`]). Both are only
//! observable for multi-antenna captures / by +-1 LSB. **UNVERIFIED against
//! real captures**: only synthetic fixtures were tested.
//!
//! RSSI semantics: the tool reports `rssi` as dB above the noise floor, so
//! `rssi_dbm = noise_floor + rssi` is reported (0 when no noise floor).

use std::io::Read;

use super::util::{channel_to_freq_mhz, clamp_i8, device_time, read_body, read_full, skip, Fill};
use super::{FormatError, ReaderOptions, MAX_VALUES_PER_FRAME};
use crate::csi_frame::{
    Adr018Flags, AntennaConfig, Bandwidth, CsiFrame, CsiMetadata, PpduType, SubcarrierData,
};

const FORMAT: &str = "atheros";
/// Header bytes after the 2-byte `field_len`.
pub const HEADER_LEN: usize = 25;
const BITS: usize = 10;

/// Decoded fixed header of one record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtherosHeader {
    pub timestamp: u64,
    pub csi_len: u16,
    pub tx_channel: u16,
    pub err_info: u8,
    pub noise_floor: i8,
    pub rate: u8,
    pub bandwidth: u8,
    pub num_tones: u8,
    pub nr: u8,
    pub nc: u8,
    pub rssi: u8,
    pub rssi_chains: [u8; 3],
    pub payload_len: u16,
}

impl AtherosHeader {
    pub fn parse(b: &[u8], big_endian: bool) -> Option<Self> {
        if b.len() < HEADER_LEN {
            return None;
        }
        let u16f = |o: usize| {
            let a = [b[o], b[o + 1]];
            if big_endian {
                u16::from_be_bytes(a)
            } else {
                u16::from_le_bytes(a)
            }
        };
        let mut ts = [0u8; 8];
        ts.copy_from_slice(&b[0..8]);
        Some(Self {
            timestamp: if big_endian {
                u64::from_be_bytes(ts)
            } else {
                u64::from_le_bytes(ts)
            },
            csi_len: u16f(8),
            tx_channel: u16f(10),
            err_info: b[12],
            noise_floor: b[13] as i8,
            rate: b[14],
            bandwidth: b[15],
            num_tones: b[16],
            nr: b[17],
            nc: b[18],
            rssi: b[19],
            rssi_chains: [b[20], b[21], b[22]],
            payload_len: u16f(23),
        })
    }

    /// Structural plausibility used for detection and per-record validation.
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=4).contains(&self.nr) || !(1..=4).contains(&self.nc) {
            return Err(format!("invalid antenna counts nr={} nc={}", self.nr, self.nc));
        }
        if self.num_tones == 0 {
            return Err("num_tones is zero".into());
        }
        let values = self.num_tones as usize * self.nr as usize * self.nc as usize;
        if values > MAX_VALUES_PER_FRAME {
            return Err(format!("{values} CSI values exceed limit"));
        }
        let need_bits = values * 2 * BITS;
        if (self.csi_len as usize) * 8 < need_bits {
            return Err(format!(
                "csi_len {} bytes too small for {values} complex values",
                self.csi_len
            ));
        }
        Ok(())
    }
}

/// `true` if `rest` (starting at a record's `field_len`) looks like a valid
/// first record in the given byte order.
fn first_record_plausible(rest: &[u8], big_endian: bool) -> bool {
    if rest.len() < 2 + HEADER_LEN {
        return false;
    }
    let a = [rest[0], rest[1]];
    let flen = if big_endian {
        u16::from_be_bytes(a)
    } else {
        u16::from_le_bytes(a)
    } as usize;
    match AtherosHeader::parse(&rest[2..], big_endian) {
        Some(h) => {
            flen >= HEADER_LEN + h.csi_len as usize
                && h.bandwidth <= 3
                && h.csi_len > 0
                && h.validate().is_ok()
        }
        None => false,
    }
}

/// Decide `(bytes_to_skip, big_endian)` from the first bytes of a file.
fn pick_endianness(prefix: &[u8]) -> (usize, bool) {
    if prefix.first() == Some(&0xFF)
        && first_record_plausible(&prefix[1..], true)
        && !first_record_plausible(prefix, false)
    {
        return (1, true);
    }
    (0, false)
}

/// Quick structural check of the first record (used by format detection).
pub(crate) fn looks_like_atheros(head: &[u8]) -> bool {
    first_record_plausible(head, false)
        || (head.first() == Some(&0xFF) && first_record_plausible(&head[1..], true))
}

/// A reader with a small push-back prefix so endianness can be determined
/// without seeking.
struct Pushback<R> {
    prefix: Vec<u8>,
    pos: usize,
    inner: R,
}

impl<R: Read> Read for Pushback<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos < self.prefix.len() {
            let n = (self.prefix.len() - self.pos).min(out.len());
            out[..n].copy_from_slice(&self.prefix[self.pos..self.pos + n]);
            self.pos += n;
            return Ok(n);
        }
        self.inner.read(out)
    }
}

/// Unpack `values` complex numbers from the 10-bit stream, returning
/// `(real, imag)` in file order (tone, nc, nr).
fn unpack(csi: &[u8], values: usize, plus_one: bool) -> Option<Vec<(i16, i16)>> {
    if csi.len() * 8 < values * 2 * BITS {
        return None;
    }
    let mut pos = 0usize;
    let mut take = |pos: &mut usize| -> i16 {
        let mut v = 0u32;
        for b in 0..BITS {
            let p = *pos + b;
            if csi[p / 8] >> (p % 8) & 1 == 1 {
                v |= 1 << b;
            }
        }
        *pos += BITS;
        let mut s = v as i32;
        if v & (1 << (BITS - 1)) != 0 {
            s -= 1 << BITS;
        }
        if plus_one {
            s += 1;
        }
        s as i16
    };
    let mut out = Vec::with_capacity(values);
    for _ in 0..values {
        let imag = take(&mut pos);
        let real = take(&mut pos);
        out.push((real, imag));
    }
    Some(out)
}

fn tone_indices(num_tones: usize) -> Vec<i16> {
    match num_tones {
        56 => (-28..=-1).chain(1..=28).collect(),
        114 => (-58..=-2).chain(2..=58).collect(),
        n => (0..n as i16).collect(),
    }
}

/// Streaming reader for Atheros CSI Tool logs.
pub struct AtherosReader<R> {
    inner: Pushback<R>,
    offset: u64,
    started: bool,
    big_endian: bool,
    buf: Vec<u8>,
    seq: u32,
    plus_one: bool,
    freq_override: Option<u32>,
    done: bool,
}

impl<R: Read> AtherosReader<R> {
    pub fn new(inner: R, opts: &ReaderOptions) -> Self {
        Self {
            inner: Pushback {
                prefix: Vec::new(),
                pos: 0,
                inner,
            },
            offset: 0,
            started: false,
            big_endian: false,
            buf: Vec::new(),
            seq: 0,
            plus_one: opts.atheros_plus_one,
            freq_override: opts.channel_freq_mhz,
            done: false,
        }
    }

    /// Determine endianness (optional 0xFF marker) from the first bytes.
    fn start(&mut self) -> Result<(), FormatError> {
        let mut prefix = vec![0u8; 1 + 2 + HEADER_LEN];
        let n = match read_full(&mut self.inner.inner, &mut prefix)? {
            Fill::Full => prefix.len(),
            Fill::Partial(n) => n,
            Fill::Eof => 0,
        };
        prefix.truncate(n);
        let (skip_n, be) = pick_endianness(&prefix);
        self.big_endian = be;
        self.offset = skip_n as u64;
        self.inner.prefix = prefix;
        self.inner.pos = skip_n;
        self.started = true;
        Ok(())
    }

    fn read_next(&mut self) -> Result<Option<CsiFrame>, FormatError> {
        if !self.started {
            self.start()?;
        }
        let start = self.offset;
        let mut lenb = [0u8; 2];
        match read_full(&mut self.inner, &mut lenb)? {
            Fill::Eof => {
                self.done = true;
                return Ok(None);
            }
            Fill::Partial(_) => {
                self.done = true;
                return Err(FormatError::Truncated { format: FORMAT, offset: start });
            }
            Fill::Full => {}
        }
        self.offset += 2;
        let field_len = if self.big_endian {
            u16::from_be_bytes(lenb)
        } else {
            u16::from_le_bytes(lenb)
        } as usize;
        if field_len < HEADER_LEN {
            let skipped = skip(&mut self.inner, field_len as u64)?;
            self.offset += skipped;
            if skipped < field_len as u64 {
                self.done = true;
                return Err(FormatError::Truncated { format: FORMAT, offset: start });
            }
            return Err(FormatError::Corrupt {
                format: FORMAT,
                offset: start,
                reason: format!("field_len {field_len} shorter than the {HEADER_LEN}-byte header"),
            });
        }
        match read_body(&mut self.inner, &mut self.buf, field_len)? {
            Fill::Full => {}
            _ => {
                self.done = true;
                return Err(FormatError::Truncated { format: FORMAT, offset: start });
            }
        }
        self.offset += field_len as u64;
        let header = AtherosHeader::parse(&self.buf, self.big_endian).ok_or(FormatError::Corrupt {
            format: FORMAT,
            offset: start,
            reason: "short header".into(),
        })?;
        let body = std::mem::take(&mut self.buf);
        let r = self.decode(start, &header, &body[HEADER_LEN..]);
        self.buf = body;
        r.map(Some)
    }

    fn decode(
        &mut self,
        start: u64,
        h: &AtherosHeader,
        after_header: &[u8],
    ) -> Result<CsiFrame, FormatError> {
        let corrupt = |reason: String| FormatError::Corrupt {
            format: FORMAT,
            offset: start,
            reason,
        };
        h.validate().map_err(&corrupt)?;
        let csi_len = h.csi_len as usize;
        let csi = after_header
            .get(..csi_len)
            .ok_or_else(|| corrupt(format!("csi_len {csi_len} exceeds record")))?;
        let (tones, nr, nc) = (h.num_tones as usize, h.nr as usize, h.nc as usize);
        let vals = unpack(csi, tones * nr * nc, self.plus_one)
            .ok_or_else(|| corrupt("CSI buffer too short".into()))?;
        let idx = tone_indices(tones);
        // File order is [tone][nc][nr]; emit [tx=nc][rx=nr][tone].
        let mut subcarriers = vec![SubcarrierData { i: 0, q: 0, index: 0 }; vals.len()];
        for (n, &(re, im)) in vals.iter().enumerate() {
            let k = n / (nr * nc);
            let rem = n % (nr * nc);
            let (tx, rx) = (rem / nr, rem % nr);
            subcarriers[(tx * nr + rx) * tones + k] = SubcarrierData {
                i: re,
                q: im,
                index: idx[k],
            };
        }
        let freq = self.freq_override.unwrap_or(match h.tx_channel as u32 {
            f if f >= 2400 => f,
            ch => channel_to_freq_mhz(ch),
        });
        let (bandwidth, bw40) = match h.bandwidth {
            0 => (Bandwidth::Bw20, false),
            1 => (Bandwidth::Bw40, true),
            2 => (Bandwidth::Bw80, false),
            _ => (Bandwidth::Bw160, false),
        };
        self.seq = self.seq.wrapping_add(1);
        Ok(CsiFrame {
            metadata: CsiMetadata {
                timestamp: device_time(h.timestamp),
                node_id: 0,
                n_antennas: (nr * nc) as u8,
                n_subcarriers: tones as u16,
                channel_freq_mhz: freq,
                rssi_dbm: if h.noise_floor == 0 {
                    0
                } else {
                    clamp_i8(h.noise_floor as f64 + h.rssi as f64)
                },
                noise_floor_dbm: h.noise_floor,
                bandwidth,
                antenna_config: AntennaConfig {
                    tx_antennas: nc as u8,
                    rx_antennas: nr as u8,
                },
                sequence: self.seq,
                ppdu_type: PpduType::HtLegacy,
                adr018_flags: Adr018Flags {
                    bw40,
                    ..Adr018Flags::default()
                },
            },
            subcarriers,
        })
    }
}

impl<R: Read> Iterator for AtherosReader<R> {
    type Item = Result<CsiFrame, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.read_next() {
            Ok(Some(f)) => Some(Ok(f)),
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

    /// Pack `(real, imag)` values (file order: tone, nc, nr) into the 10-bit stream.
    pub fn pack(vals: &[(i16, i16)]) -> Vec<u8> {
        let mut out = vec![0u8; (vals.len() * 2 * BITS).div_ceil(8)];
        let mut pos = 0usize;
        for &(re, im) in vals {
            for v in [im, re] {
                let u = (v as i32 & 0x3ff) as u32;
                for b in 0..BITS {
                    if u >> b & 1 == 1 {
                        out[(pos + b) / 8] |= 1 << ((pos + b) % 8);
                    }
                }
                pos += BITS;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        big: bool,
        ts: u64,
        channel: u16,
        noise: i8,
        bw: u8,
        tones: u8,
        nr: u8,
        nc: u8,
        rssi: u8,
        csi: &[u8],
        payload: &[u8],
    ) -> Vec<u8> {
        let w16 = |x: u16| if big { x.to_be_bytes() } else { x.to_le_bytes() };
        let mut h = Vec::new();
        h.extend_from_slice(&if big { ts.to_be_bytes() } else { ts.to_le_bytes() });
        h.extend_from_slice(&w16(csi.len() as u16));
        h.extend_from_slice(&w16(channel));
        h.extend_from_slice(&[0, noise as u8, 0x8c, bw, tones, nr, nc, rssi, rssi, rssi, rssi]);
        h.extend_from_slice(&w16(payload.len() as u16));
        assert_eq!(h.len(), HEADER_LEN);
        let field_len = HEADER_LEN + csi.len() + payload.len();
        let mut v = Vec::new();
        v.extend_from_slice(&w16(field_len as u16));
        v.extend(h);
        v.extend_from_slice(csi);
        v.extend_from_slice(payload);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;

    fn read_all(file: &[u8]) -> Vec<Result<CsiFrame, FormatError>> {
        AtherosReader::new(file, &ReaderOptions::default()).collect()
    }

    /// Hand-packed bytes: tone0 imag=-1 real=2, tone1 imag=511 real=-512.
    #[test]
    fn hand_computed_10bit_unpack() {
        let csi = [0xFF, 0x0B, 0xF0, 0x1F, 0x80];
        assert_eq!(pack(&[(2, -1), (-512, 511)]), csi);
        let v = unpack(&csi, 2, false).unwrap();
        assert_eq!(v, vec![(2, -1), (-512, 511)]);
        let v1 = unpack(&csi, 2, true).unwrap();
        assert_eq!(v1, vec![(3, 0), (-511, 512)]);
        assert!(unpack(&csi, 3, false).is_none());
    }

    #[test]
    fn reads_2x2_56_tone_frames_le() {
        let (tones, nr, nc) = (56usize, 2usize, 2usize);
        let vals: Vec<(i16, i16)> = (0..tones * nr * nc)
            .map(|n| ((n as i16 % 400) - 200, 100 - (n as i16 % 300)))
            .collect();
        let csi = pack(&vals);
        let mut file = record(false, 5_000_000, 2437, -95, 0, 56, 2, 2, 40, &csi, &[9; 17]);
        file.extend(record(false, 5_001_000, 5745, -90, 1, 56, 2, 2, 30, &csi, &[]));
        let v = read_all(&file);
        assert_eq!(v.len(), 2);
        let f = v[0].as_ref().unwrap();
        assert_eq!(f.subcarrier_count(), 56 * 4);
        assert_eq!(f.metadata.n_antennas, 4);
        assert_eq!(f.metadata.n_subcarriers, 56);
        assert_eq!(f.metadata.antenna_config.rx_antennas, 2);
        assert_eq!(f.metadata.channel_freq_mhz, 2437);
        assert_eq!(f.metadata.rssi_dbm, -55);
        assert_eq!(f.metadata.timestamp.timestamp_micros(), 5_000_000);
        // Value at file position n = ((tone*nc + tx)*nr + rx): check stream-major layout.
        for tone in [0usize, 13, 55] {
            for tx in 0..nc {
                for rx in 0..nr {
                    let n = (tone * nc + tx) * nr + rx;
                    let s = f.subcarriers[(tx * nr + rx) * tones + tone];
                    assert_eq!((s.i, s.q), vals[n]);
                }
            }
        }
        assert_eq!(f.subcarriers[0].index, -28);
        assert_eq!(f.subcarriers[27].index, -1);
        assert_eq!(f.subcarriers[28].index, 1);
        assert_eq!(v[1].as_ref().unwrap().metadata.channel_freq_mhz, 5745);
        assert_eq!(v[1].as_ref().unwrap().metadata.bandwidth, Bandwidth::Bw40);
    }

    #[test]
    fn big_endian_file_with_marker() {
        let vals: Vec<(i16, i16)> = (0..56).map(|n| (n as i16 - 20, 7)).collect();
        let csi = pack(&vals);
        let mut file = vec![0xFF];
        file.extend(record(true, 42, 6, -92, 0, 56, 1, 1, 20, &csi, &[]));
        file.extend(record(true, 43, 6, -92, 0, 56, 1, 1, 20, &csi, &[1, 2, 3]));
        let v = read_all(&file);
        assert_eq!(v.len(), 2, "{v:?}");
        let f = v[0].as_ref().unwrap();
        assert_eq!(f.subcarriers[3].i, -17);
        assert_eq!(f.metadata.channel_freq_mhz, 2437);
        assert_eq!(v[1].as_ref().unwrap().metadata.timestamp.timestamp_micros(), 43);
    }

    #[test]
    fn little_endian_file_whose_first_length_byte_is_0xff() {
        // field_len low byte 0xFF without a marker: 25 + csi + payload = 0x01FF.
        let vals: Vec<(i16, i16)> = (0..56).map(|n| (n as i16, -(n as i16))).collect();
        let csi = pack(&vals); // 140 bytes
        let payload = vec![0u8; 0x1FF - HEADER_LEN - csi.len()];
        let mut file = record(false, 1, 11, -90, 0, 56, 1, 1, 20, &csi, &payload);
        assert_eq!(file[0], 0xFF);
        file.extend(record(false, 2, 11, -90, 0, 56, 1, 1, 20, &csi, &[]));
        let v = read_all(&file);
        assert_eq!(v.len(), 2, "{v:?}");
        assert_eq!(v[0].as_ref().unwrap().subcarriers[10].i, 10);
        assert_eq!(v[1].as_ref().unwrap().metadata.timestamp.timestamp_micros(), 2);
    }

    #[test]
    fn bad_header_is_recoverable() {
        let csi = pack(&[(1, 1); 56]);
        let mut bad = record(false, 1, 6, -90, 0, 56, 1, 1, 20, &csi, &[]);
        bad[2 + 17] = 9; // nr = 9
        let good = record(false, 2, 6, -90, 0, 56, 1, 1, 20, &csi, &[]);
        let mut file = bad;
        file.extend(good);
        let v = read_all(&file);
        assert_eq!(v.len(), 2);
        assert!(matches!(v[0], Err(FormatError::Corrupt { .. })));
        assert!(v[1].is_ok());
    }

    #[test]
    fn truncated_and_short_inputs() {
        let csi = pack(&[(1, 1); 56]);
        let mut file = record(false, 1, 6, -90, 0, 56, 1, 1, 20, &csi, &[]);
        file.truncate(file.len() - 3);
        let v = read_all(&file);
        assert!(matches!(v[0], Err(FormatError::Truncated { .. })));
        assert_eq!(v.len(), 1);
        assert!(read_all(&[0x05]).len() <= 1);
        assert!(read_all(&[]).is_empty());
    }

    #[test]
    fn detection() {
        let csi = pack(&[(1, 1); 56]);
        let rec = record(false, 1, 6, -90, 0, 56, 2, 2, 20, &csi, &[]);
        // csi too small for 2x2 -> not plausible
        assert!(!looks_like_atheros(&rec));
        let csi = pack(&[(1, 1); 56 * 4]);
        let rec = record(false, 1, 6, -90, 0, 56, 2, 2, 20, &csi, &[]);
        assert!(looks_like_atheros(&rec));
        assert!(!looks_like_atheros(&[0u8; 100]));
    }
}
