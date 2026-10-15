//! Nexmon CSI (Broadcom) libpcap reader.
//!
//! nexmon_csi streams CSI as UDP packets to port 5500; capturing them with
//! `tcpdump -i wlan0 dst port 5500 -w out.pcap` yields a classic pcap file.
//! This reader walks the pcap records, extracts IPv4/UDP packets destined to
//! the CSI port and decodes the documented payload:
//!
//! ```text
//! UDP payload:
//!   0..2   u16  magic 0x1111 (bytes 11 11)
//!   2      i8   RSSI            } only in the "mzakharo" variant; stock builds
//!   3      u8   frame control   } repeat 0x11 here (-> unknown)
//!   4..10  [u8;6] source MAC
//!   10..12 u16 LE sequence number
//!   12..14 u16 LE core (bits 0-2) / spatial stream (bits 3-5)
//!   14..16 u16 LE chanspec
//!   16..18 u16 LE chip version
//!   18..   CSI, nfft (64/128/256 for 20/40/80 MHz) 32-bit little-endian words
//! ```
//!
//! CSI word layouts per chip:
//!
//! * `bcm4339`, `bcm43455c0` (Raspberry Pi 3B+/4): each word is `int16 real`
//!   then `int16 imag` ([`CsiLayout::Int16`]).
//! * `bcm4358`, `bcm4366c0`: each word is a packed floating-point value with
//!   a shared exponent: `nexp` exponent bits, then `nman` bits imag, `nman`
//!   bits real, each `nman`-bit field being sign + (`nman`-1) mantissa bits;
//!   (`nman`,`nexp`) = (9,5) for 4358 and (12,6) for 4366c0
//!   ([`CsiLayout::PackedFloat`]). Values are re-scaled per packet (see
//!   [`unpack_float_acphy`]).
//!
//! Subcarriers are emitted in the raw FFT order of the packet (DC first, then
//! positive, then negative frequencies); [`SubcarrierData::index`] carries the
//! signed subcarrier number so consumers can `fftshift` if desired.
//!
//! Each packet is one (core, spatial stream) pair and is emitted as its own
//! frame (`n_antennas == 1`); packets are not merged into MIMO frames.
//!
//! # Attribution
//!
//! The packet layout comes from the public nexmon_csi README (no licence is
//! asserted for that project, so no code was taken from it). The chip table
//! and the packed-float algorithm were cross-checked against CSIKit
//! (<https://github.com/Gi-z/CSIKit>, MIT, `read_pcap.py`, `byteops.py`) and
//! csiread (<https://github.com/citysu/csiread>, MIT, `_csiread.pyx`).

use std::io::Read;

use super::util::{read_body, read_full, wall_time, Fill};
use super::{FormatError, ReaderOptions, MAX_PCAP_PACKET_BYTES};
use crate::csi_frame::{
    Adr018Flags, AntennaConfig, Bandwidth, CsiFrame, CsiMetadata, PpduType, SubcarrierData,
};

const FORMAT: &str = "nexmon-pcap";
/// Default UDP port used by nexmon_csi.
pub const NEXMON_UDP_PORT: u16 = 5500;
/// Bytes of nexmon header preceding the CSI words.
pub const NEXMON_HEADER_LEN: usize = 18;

/// Broadcom chips with a known CSI word layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NexmonChip {
    Bcm4339,
    Bcm43455c0,
    Bcm4358,
    Bcm4366c0,
}

/// How a chip encodes each 32-bit CSI word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsiLayout {
    Int16,
    PackedFloat { nman: u32, nexp: u32 },
}

impl NexmonChip {
    pub fn layout(self) -> CsiLayout {
        match self {
            Self::Bcm4339 | Self::Bcm43455c0 => CsiLayout::Int16,
            Self::Bcm4358 => CsiLayout::PackedFloat { nman: 9, nexp: 5 },
            Self::Bcm4366c0 => CsiLayout::PackedFloat { nman: 12, nexp: 6 },
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bcm4339 => "bcm4339",
            Self::Bcm43455c0 => "bcm43455c0",
            Self::Bcm4358 => "bcm4358",
            Self::Bcm4366c0 => "bcm4366c0",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().trim_start_matches("bcm") {
            "4339" => Some(Self::Bcm4339),
            "43455c0" => Some(Self::Bcm43455c0),
            "4358" => Some(Self::Bcm4358),
            "4366c0" => Some(Self::Bcm4366c0),
            _ => None,
        }
    }

    /// Map the 2-byte chip-version field (as stored on the wire).
    pub fn from_version_bytes(b: [u8; 2]) -> Option<Self> {
        match b {
            [0x01, 0x00] => Some(Self::Bcm4339),
            [0x65, 0x00] | [0xdc, 0xa6] => Some(Self::Bcm43455c0),
            [0x03, 0x00] | [0xad, 0xde] => Some(Self::Bcm4358),
            [0x34, 0xe8] | [0x6a, 0x00] => Some(Self::Bcm4366c0),
            _ => None,
        }
    }
}

/// Decode nexmon's packed-float CSI words (`unpack_float_acphy` with
/// autoscale, `nbits = 10`).
///
/// Per word: `e = word & (2^nexp-1)` interpreted as a signed `nexp`-bit
/// exponent; `imag = (word >> nexp)` and `real = (word >> (nexp+nman))`, each
/// masked to `nman-1` mantissa bits with a separate sign bit above each field.
/// With autoscale, `e` is increased by `floor(log2(real | imag))` and the
/// maximum over the packet fixes `shift = 10 - max_e`; each component is then
/// `sign * (mantissa << (e_raw + shift))` (right-shifted when negative, and
/// zeroed below `-nman`).
pub fn unpack_float_acphy(words: &[u32], nman: u32, nexp: u32) -> Vec<(i32, i32)> {
    const NBITS: i64 = 10;
    if nman < 2 || nexp < 2 || nexp + 2 * nman > 32 {
        return vec![(0, 0); words.len()];
    }
    let iq_mask: u32 = (1u32 << (nman - 1)) - 1;
    let e_mask: u32 = (1u32 << nexp) - 1;
    let e_p: i64 = 1i64 << (nexp - 1);
    let sgnr_mask: u32 = 1u32 << (nexp + 2 * nman - 1);
    let sgni_mask: u32 = sgnr_mask >> nman;
    let e_zero: i64 = -(nman as i64);

    let mut parsed: Vec<(u32, u32, bool, bool, i64)> = Vec::with_capacity(words.len());
    let mut maxbit: i64 = -e_p;
    for &h in words {
        let vi = (h >> (nexp + nman)) & iq_mask; // real
        let vq = (h >> nexp) & iq_mask; // imag
        let mut e = (h & e_mask) as i64;
        if e >= e_p {
            e -= e_p << 1;
        }
        let raw_e = e;
        let x = vi | vq;
        if x != 0 {
            e += (31 - x.leading_zeros()) as i64;
            if e > maxbit {
                maxbit = e;
            }
        }
        parsed.push((vi, vq, h & sgnr_mask != 0, h & sgni_mask != 0, raw_e));
    }
    let shft = NBITS - maxbit;
    let apply = |mant: u32, neg: bool, raw_e: i64| -> i32 {
        let e = raw_e + shft;
        let mut v = mant as i64;
        if e < e_zero {
            v = 0;
        } else if e < 0 {
            v >>= (-e).min(63);
        } else {
            v <<= e.min(40);
        }
        if neg {
            v = -v;
        }
        v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
    };
    parsed
        .into_iter()
        .map(|(vi, vq, sr, si, re)| (apply(vi, sr, re), apply(vq, si, re)))
        .collect()
}

/// Decode a Broadcom chanspec to `(channel, freq_mhz)`.
pub fn chanspec_to_channel_freq(chanspec: u16) -> (u8, u32) {
    let channel = (chanspec & 0xff) as u8;
    let freq = match chanspec & 0xc000 {
        0xc000 => 5000 + 5 * channel as u32,
        0x0000 if (1..=14).contains(&channel) => {
            if channel == 14 {
                2484
            } else {
                2407 + 5 * channel as u32
            }
        }
        _ => 0,
    };
    (channel, freq)
}

/// A decoded nexmon payload header (before CSI words).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NexmonHeader {
    /// RSSI in dBm, `None` on stock builds that do not fill it.
    pub rssi: Option<i8>,
    pub frame_control: Option<u8>,
    pub src_mac: [u8; 6],
    pub seq: u16,
    pub core: u8,
    pub spatial_stream: u8,
    pub chanspec: u16,
    pub chip_version: [u8; 2],
}

impl NexmonHeader {
    /// Parse the first [`NEXMON_HEADER_LEN`] bytes of a UDP payload.
    pub fn parse(p: &[u8]) -> Option<Self> {
        if p.len() < NEXMON_HEADER_LEN || p[0] != 0x11 || p[1] != 0x11 {
            return None;
        }
        let stock = p[2] == 0x11 && p[3] == 0x11;
        let cs = u16::from_le_bytes([p[12], p[13]]);
        Some(Self {
            rssi: (!stock).then_some(p[2] as i8),
            frame_control: (!stock).then_some(p[3]),
            src_mac: [p[4], p[5], p[6], p[7], p[8], p[9]],
            seq: u16::from_le_bytes([p[10], p[11]]),
            core: (cs & 0x7) as u8,
            spatial_stream: ((cs >> 3) & 0x7) as u8,
            chanspec: u16::from_le_bytes([p[14], p[15]]),
            chip_version: [p[16], p[17]],
        })
    }
}

/// Pcap magic sniffing (classic pcap, both endians, micro/nanosecond).
pub(crate) fn looks_like_pcap(head: &[u8]) -> bool {
    head.len() >= 24 && PcapEndian::from_magic([head[0], head[1], head[2], head[3]]).is_some()
}

#[derive(Debug, Clone, Copy)]
struct PcapEndian {
    big: bool,
    nanos: bool,
}

impl PcapEndian {
    fn from_magic(m: [u8; 4]) -> Option<Self> {
        match m {
            [0xd4, 0xc3, 0xb2, 0xa1] => Some(Self { big: false, nanos: false }),
            [0xa1, 0xb2, 0xc3, 0xd4] => Some(Self { big: true, nanos: false }),
            [0x4d, 0x3c, 0xb2, 0xa1] => Some(Self { big: false, nanos: true }),
            [0xa1, 0xb2, 0x3c, 0x4d] => Some(Self { big: true, nanos: true }),
            _ => None,
        }
    }
    fn u32(self, b: &[u8]) -> u32 {
        let a = [b[0], b[1], b[2], b[3]];
        if self.big {
            u32::from_be_bytes(a)
        } else {
            u32::from_le_bytes(a)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkType {
    Ethernet,
    RawIp,
    LinuxSll,
}

/// Locate the UDP payload of an IPv4/UDP packet to `port`.
fn udp_payload(link: LinkType, pkt: &[u8], port: u16) -> Option<&[u8]> {
    let ip = match link {
        LinkType::Ethernet => {
            let mut off = 12;
            let mut et = u16::from_be_bytes([*pkt.get(off)?, *pkt.get(off + 1)?]);
            off += 2;
            // Skip up to two VLAN tags.
            for _ in 0..2 {
                if et == 0x8100 || et == 0x88a8 {
                    et = u16::from_be_bytes([*pkt.get(off + 2)?, *pkt.get(off + 3)?]);
                    off += 4;
                }
            }
            if et != 0x0800 {
                return None;
            }
            pkt.get(off..)?
        }
        LinkType::LinuxSll => {
            let proto = u16::from_be_bytes([*pkt.get(14)?, *pkt.get(15)?]);
            if proto != 0x0800 {
                return None;
            }
            pkt.get(16..)?
        }
        LinkType::RawIp => pkt,
    };
    if ip.len() < 20 || ip[0] >> 4 != 4 {
        return None;
    }
    let ihl = (ip[0] & 0x0f) as usize * 4;
    if ihl < 20 || ip[9] != 17 {
        return None;
    }
    let frag = u16::from_be_bytes([ip[6], ip[7]]);
    if frag & 0x1fff != 0 || frag & 0x2000 != 0 {
        return None; // fragments are not reassembled
    }
    let udp = ip.get(ihl..)?;
    if udp.len() < 8 {
        return None;
    }
    if u16::from_be_bytes([udp[2], udp[3]]) != port {
        return None;
    }
    let ulen = u16::from_be_bytes([udp[4], udp[5]]) as usize;
    let end = if ulen >= 8 { ulen.min(udp.len()) } else { udp.len() };
    udp.get(8..end)
}

/// Streaming reader for Nexmon CSI pcap files.
pub struct NexmonPcapReader<R> {
    inner: R,
    endian: PcapEndian,
    link: LinkType,
    port: u16,
    chip_override: Option<NexmonChip>,
    freq_override: Option<u32>,
    offset: u64,
    buf: Vec<u8>,
    done: bool,
}

impl<R: Read> NexmonPcapReader<R> {
    /// Read and validate the 24-byte pcap global header.
    pub fn new(mut inner: R, opts: &ReaderOptions) -> Result<Self, FormatError> {
        let mut hdr = [0u8; 24];
        match read_full(&mut inner, &mut hdr)? {
            Fill::Full => {}
            _ => {
                return Err(FormatError::BadHeader {
                    format: FORMAT,
                    reason: "file shorter than the 24-byte pcap global header".into(),
                })
            }
        }
        let endian = PcapEndian::from_magic([hdr[0], hdr[1], hdr[2], hdr[3]]).ok_or_else(|| {
            FormatError::BadHeader {
                format: FORMAT,
                reason: "not a classic pcap file (bad magic)".into(),
            }
        })?;
        let linktype = endian.u32(&hdr[20..24]) & 0x0fff_ffff;
        let link = match linktype {
            1 => LinkType::Ethernet,
            101 | 228 => LinkType::RawIp,
            113 => LinkType::LinuxSll,
            other => {
                return Err(FormatError::BadHeader {
                    format: FORMAT,
                    reason: format!(
                        "unsupported pcap link type {other} (need Ethernet=1, raw IP=101 or Linux cooked=113)"
                    ),
                })
            }
        };
        Ok(Self {
            inner,
            endian,
            link,
            port: opts.nexmon_port,
            chip_override: opts.nexmon_chip,
            freq_override: opts.channel_freq_mhz,
            offset: 24,
            buf: Vec::new(),
            done: false,
        })
    }

    fn corrupt(&self, offset: u64, reason: String) -> FormatError {
        FormatError::Corrupt {
            format: FORMAT,
            offset,
            reason,
        }
    }

    fn read_next(&mut self) -> Result<Option<CsiFrame>, FormatError> {
        loop {
            let start = self.offset;
            let mut rh = [0u8; 16];
            match read_full(&mut self.inner, &mut rh)? {
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
            self.offset += 16;
            let ts_sec = self.endian.u32(&rh[0..4]);
            let ts_frac = self.endian.u32(&rh[4..8]);
            let incl = self.endian.u32(&rh[8..12]) as usize;
            if incl > MAX_PCAP_PACKET_BYTES {
                // Framing cannot be trusted any more.
                self.done = true;
                return Err(FormatError::BadHeader {
                    format: FORMAT,
                    reason: format!(
                        "packet at offset {start} declares {incl} captured bytes (max {MAX_PCAP_PACKET_BYTES})"
                    ),
                });
            }
            match read_body(&mut self.inner, &mut self.buf, incl)? {
                Fill::Full => {}
                _ => {
                    self.done = true;
                    return Err(FormatError::Truncated {
                        format: FORMAT,
                        offset: start,
                    });
                }
            }
            self.offset += incl as u64;

            let Some(payload) = udp_payload(self.link, &self.buf, self.port) else {
                continue; // not a CSI packet
            };
            let Some(h) = NexmonHeader::parse(payload) else {
                continue;
            };
            let csi = &payload[NEXMON_HEADER_LEN..];
            if csi.len() % 4 != 0 || !matches!(csi.len() / 4, 64 | 128 | 256) {
                return Err(self.corrupt(
                    start,
                    format!("CSI payload of {} bytes is not 64/128/256 words", csi.len()),
                ));
            }
            let chip = match self
                .chip_override
                .or_else(|| NexmonChip::from_version_bytes(h.chip_version))
            {
                Some(c) => c,
                None => {
                    return Err(self.corrupt(
                        start,
                        format!(
                            "unknown chip version bytes {:02x}{:02x}; set ReaderOptions::nexmon_chip",
                            h.chip_version[0], h.chip_version[1]
                        ),
                    ))
                }
            };
            let nfft = csi.len() / 4;
            let words: Vec<u32> = csi
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let iq: Vec<(i32, i32)> = match chip.layout() {
                CsiLayout::Int16 => words
                    .iter()
                    .map(|&w| ((w & 0xffff) as u16 as i16 as i32, (w >> 16) as u16 as i16 as i32))
                    .collect(),
                CsiLayout::PackedFloat { nman, nexp } => unpack_float_acphy(&words, nman, nexp),
            };
            let half = nfft as i16 / 2;
            let subcarriers = iq
                .iter()
                .enumerate()
                .map(|(k, &(r, i))| SubcarrierData {
                    i: r.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    q: i.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    index: if (k as i16) < half { k as i16 } else { k as i16 - nfft as i16 },
                })
                .collect();
            let (_, freq) = chanspec_to_channel_freq(h.chanspec);
            let nanos = if self.endian.nanos {
                ts_frac
            } else {
                ts_frac.saturating_mul(1000)
            };
            let bandwidth = match nfft {
                64 => Bandwidth::Bw20,
                128 => Bandwidth::Bw40,
                _ => Bandwidth::Bw80,
            };
            return Ok(Some(CsiFrame {
                metadata: CsiMetadata {
                    timestamp: wall_time(ts_sec as i64, nanos),
                    node_id: h.src_mac[5],
                    n_antennas: 1,
                    n_subcarriers: nfft as u16,
                    channel_freq_mhz: self.freq_override.unwrap_or(freq),
                    rssi_dbm: h.rssi.unwrap_or(0),
                    noise_floor_dbm: 0,
                    bandwidth,
                    antenna_config: AntennaConfig::default(),
                    sequence: h.seq as u32,
                    ppdu_type: PpduType::HtLegacy,
                    adr018_flags: Adr018Flags {
                        bw40: nfft == 128,
                        ..Adr018Flags::default()
                    },
                },
                subcarriers,
            }));
        }
    }
}

impl<R: Read> Iterator for NexmonPcapReader<R> {
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

    pub fn pcap_header(nanos: bool, big: bool, linktype: u32) -> Vec<u8> {
        let magic: u32 = if nanos { 0xa1b23c4d } else { 0xa1b2c3d4 };
        let mut v = Vec::new();
        let w32 = |v: &mut Vec<u8>, x: u32| {
            if big {
                v.extend_from_slice(&x.to_be_bytes())
            } else {
                v.extend_from_slice(&x.to_le_bytes())
            }
        };
        let w16 = |v: &mut Vec<u8>, x: u16| {
            if big {
                v.extend_from_slice(&x.to_be_bytes())
            } else {
                v.extend_from_slice(&x.to_le_bytes())
            }
        };
        w32(&mut v, magic);
        w16(&mut v, 2);
        w16(&mut v, 4);
        w32(&mut v, 0);
        w32(&mut v, 0);
        w32(&mut v, 65535);
        w32(&mut v, linktype);
        v
    }

    /// Ethernet+IPv4+UDP frame carrying `payload` to `port`.
    pub fn eth_udp(port: u16, payload: &[u8]) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&[0xff; 6]);
        p.extend_from_slice(b"NEXMON");
        p.extend_from_slice(&[0x08, 0x00]);
        let total = 20 + 8 + payload.len();
        p.extend_from_slice(&[0x45, 0, (total >> 8) as u8, total as u8, 0, 0, 0, 0, 64, 17, 0, 0]);
        p.extend_from_slice(&[10, 0, 0, 1, 255, 255, 255, 255]);
        p.extend_from_slice(&[0x15, 0xb3, (port >> 8) as u8, port as u8]);
        let ulen = 8 + payload.len();
        p.extend_from_slice(&[(ulen >> 8) as u8, ulen as u8, 0, 0]);
        p.extend_from_slice(payload);
        p
    }

    pub fn record(big: bool, sec: u32, frac: u32, pkt: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        for x in [sec, frac, pkt.len() as u32, pkt.len() as u32] {
            if big {
                v.extend_from_slice(&x.to_be_bytes())
            } else {
                v.extend_from_slice(&x.to_le_bytes())
            }
        }
        v.extend_from_slice(pkt);
        v
    }

    /// Nexmon UDP payload header + CSI words.
    pub fn nexmon_payload(
        rssi: u8,
        fc: u8,
        seq: u16,
        core_ss: u16,
        chanspec: u16,
        chip: [u8; 2],
        words: &[u32],
    ) -> Vec<u8> {
        let mut p = vec![0x11, 0x11, rssi, fc, 0xde, 0xad, 0xbe, 0xef, 0x00, 0x42];
        p.extend_from_slice(&seq.to_le_bytes());
        p.extend_from_slice(&core_ss.to_le_bytes());
        p.extend_from_slice(&chanspec.to_le_bytes());
        p.extend_from_slice(&chip);
        for w in words {
            p.extend_from_slice(&w.to_le_bytes());
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;

    fn words_int16(n: usize) -> Vec<u32> {
        (0..n)
            .map(|k| {
                let re = (k as i16) - 10;
                let im = -(k as i16) * 3;
                (re as u16 as u32) | ((im as u16 as u32) << 16)
            })
            .collect()
    }

    fn read_all(file: &[u8]) -> Vec<Result<CsiFrame, FormatError>> {
        NexmonPcapReader::new(file, &ReaderOptions::default())
            .unwrap()
            .collect()
    }

    #[test]
    fn int16_43455c0_hand_values() {
        let words = words_int16(64);
        let payload = nexmon_payload(0xd6, 0x08, 0x0102, 0x0009, 0x1006, [0x65, 0x00], &words);
        let mut file = pcap_header(false, false, 1);
        file.extend(record(false, 1_700_000_000, 250_000, &eth_udp(5500, &payload)));
        let v = read_all(&file);
        assert_eq!(v.len(), 1);
        let f = v[0].as_ref().unwrap();
        assert_eq!(f.subcarrier_count(), 64);
        // k=0: re=-10, im=0; k=5: re=-5, im=-15; k=63: re=53, im=-189
        assert_eq!((f.subcarriers[0].i, f.subcarriers[0].q), (-10, 0));
        assert_eq!((f.subcarriers[5].i, f.subcarriers[5].q), (-5, -15));
        assert_eq!((f.subcarriers[63].i, f.subcarriers[63].q), (53, -189));
        assert_eq!(f.subcarriers[0].index, 0);
        assert_eq!(f.subcarriers[31].index, 31);
        assert_eq!(f.subcarriers[32].index, -32);
        let m = &f.metadata;
        assert_eq!(m.rssi_dbm, 0xd6u8 as i8);
        assert_eq!(m.sequence, 0x0102);
        assert_eq!(m.channel_freq_mhz, 2437); // chanspec 0x1006 = ch 6, 2.4 GHz
        assert_eq!(m.bandwidth, Bandwidth::Bw20);
        assert_eq!(m.node_id, 0x42);
        assert_eq!(m.timestamp.timestamp(), 1_700_000_000);
        assert_eq!(m.timestamp.timestamp_subsec_micros(), 250_000);
    }

    #[test]
    fn big_endian_nanosecond_pcap_and_5ghz_80mhz() {
        let words = words_int16(256);
        let payload = nexmon_payload(0x11, 0x11, 9, 0, 0xe02a, [0x65, 0x00], &words);
        let mut file = pcap_header(true, true, 1);
        file.extend(record(true, 5, 123_456_789, &eth_udp(5500, &payload)));
        let v = read_all(&file);
        let f = v[0].as_ref().unwrap();
        assert_eq!(f.subcarrier_count(), 256);
        assert_eq!(f.metadata.bandwidth, Bandwidth::Bw80);
        assert_eq!(f.metadata.channel_freq_mhz, 5000 + 5 * 42);
        assert_eq!(f.metadata.rssi_dbm, 0); // stock build: unknown
        assert_eq!(f.metadata.timestamp.timestamp_subsec_nanos(), 123_456_789);
    }

    /// Hand-computed packed-float vector (see doc comment on the algorithm):
    /// nman=9 nexp=5; word0: real=100 imag=-37 e=3, word1: real=-4 imag=6 e=-3.
    /// autoscale: e0=3+floor(log2(101))=9, e1=-3+2=-1 -> maxbit=9, shift=1.
    /// word0 -> (100<<4, -(37<<4)) = (1600,-592); word1 -> (-(4>>2), 6>>2) = (-1,1).
    #[test]
    fn packed_float_hand_computed() {
        let w0: u32 = 3 | (37 << 5) | (1 << 13) | (100 << 14);
        let w1: u32 = 29 | (6 << 5) | (4 << 14) | (1 << 22);
        assert_eq!(w0, 1647779);
        assert_eq!(w1, 4260061);
        let out = unpack_float_acphy(&[w0, w1], 9, 5);
        assert_eq!(out, vec![(1600, -592), (-1, 1)]);
    }

    #[test]
    fn packed_float_all_zero_and_degenerate_params() {
        assert_eq!(unpack_float_acphy(&[0, 0], 9, 5), vec![(0, 0), (0, 0)]);
        // invalid parameters must not panic
        assert_eq!(unpack_float_acphy(&[1, 2], 0, 0).len(), 2);
        assert_eq!(unpack_float_acphy(&[u32::MAX; 4], 12, 6).len(), 4);
        assert_eq!(unpack_float_acphy(&[u32::MAX; 4], 9, 5).len(), 4);
    }

    #[test]
    fn packed_float_chip_4358_through_reader() {
        let w0: u32 = 3 | (37 << 5) | (1 << 13) | (100 << 14);
        let mut words = vec![0u32; 64];
        words[0] = w0;
        let payload = nexmon_payload(0, 0, 1, 0, 0x1001, [0x03, 0x00], &words);
        let mut file = pcap_header(false, false, 1);
        file.extend(record(false, 1, 0, &eth_udp(5500, &payload)));
        let f = read_all(&file).remove(0).unwrap();
        // maxbit = 9 (only nonzero word) -> shift 1 -> (1600, -592)
        assert_eq!((f.subcarriers[0].i, f.subcarriers[0].q), (1600, -592));
    }

    #[test]
    fn skips_non_csi_traffic_and_reports_bad_sizes() {
        let good = nexmon_payload(0, 0, 1, 0, 0x1001, [0x65, 0x00], &words_int16(64));
        let short = nexmon_payload(0, 0, 2, 0, 0x1001, [0x65, 0x00], &words_int16(10));
        let unknown_chip = nexmon_payload(0, 0, 3, 0, 0x1001, [0x99, 0x99], &words_int16(64));
        let mut file = pcap_header(false, false, 1);
        file.extend(record(false, 1, 0, &eth_udp(53, b"dns")));
        file.extend(record(false, 1, 0, &eth_udp(5500, b"not nexmon")));
        file.extend(record(false, 1, 0, &eth_udp(5500, &short)));
        file.extend(record(false, 1, 0, &eth_udp(5500, &unknown_chip)));
        file.extend(record(false, 2, 0, &eth_udp(5500, &good)));
        file.extend(record(false, 2, 0, &[1, 2, 3]));
        let v = read_all(&file);
        assert_eq!(v.len(), 3);
        assert!(matches!(v[0], Err(FormatError::Corrupt { .. })));
        assert!(matches!(v[1], Err(FormatError::Corrupt { .. })));
        assert_eq!(v[2].as_ref().unwrap().metadata.sequence, 1);
    }

    #[test]
    fn chip_override_resolves_unknown_chip() {
        let p = nexmon_payload(0, 0, 3, 0, 0x1001, [0x99, 0x99], &words_int16(64));
        let mut file = pcap_header(false, false, 1);
        file.extend(record(false, 1, 0, &eth_udp(5500, &p)));
        let opts = ReaderOptions {
            nexmon_chip: Some(NexmonChip::Bcm43455c0),
            ..Default::default()
        };
        let v: Vec<_> = NexmonPcapReader::new(&file[..], &opts).unwrap().collect();
        assert!(v[0].is_ok());
    }

    #[test]
    fn raw_ip_linktype() {
        let p = nexmon_payload(0, 0, 3, 0, 0x1001, [0x65, 0x00], &words_int16(64));
        let eth = eth_udp(5500, &p);
        let mut file = pcap_header(false, false, 101);
        file.extend(record(false, 1, 0, &eth[14..]));
        assert!(read_all(&file)[0].is_ok());
    }

    #[test]
    fn header_errors() {
        assert!(NexmonPcapReader::new(&[0u8; 10][..], &ReaderOptions::default()).is_err());
        assert!(NexmonPcapReader::new(&[0u8; 24][..], &ReaderOptions::default()).is_err());
        let file = pcap_header(false, false, 147);
        assert!(NexmonPcapReader::new(&file[..], &ReaderOptions::default()).is_err());
    }

    #[test]
    fn huge_incl_len_is_fatal_not_allocated() {
        let mut file = pcap_header(false, false, 1);
        file.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0]);
        let v = read_all(&file);
        assert_eq!(v.len(), 1);
        assert!(v[0].as_ref().unwrap_err().is_fatal());
    }

    #[test]
    fn chip_table_and_chanspec() {
        assert_eq!(NexmonChip::from_version_bytes([0xdc, 0xa6]), Some(NexmonChip::Bcm43455c0));
        assert_eq!(NexmonChip::from_name("BCM4366c0"), Some(NexmonChip::Bcm4366c0));
        assert_eq!(chanspec_to_channel_freq(0xd024), (36, 5180));
        assert_eq!(chanspec_to_channel_freq(0x100e), (14, 2484));
    }
}
