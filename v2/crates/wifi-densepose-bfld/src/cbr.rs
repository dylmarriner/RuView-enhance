//! Compressed Beamforming Report (CBR) decoder for raw 802.11 beamforming
//! feedback.
//!
//! The BFLD pipeline (ADR-118/119) currently consumes the beamforming feedback
//! only as an opaque `compressed_angle_matrix` blob. This module turns a real
//! VHT (802.11ac) Compressed Beamforming Action frame body into structured
//! `Phi`/`Psi` Givens angles, per-stream average SNR, and reported dimensions,
//! so a sensing model can learn from the actual feedback rather than a
//! pre-computed proxy.
//!
//! Scope and honesty boundary (the `MEASURED`/`CLAIMED`/`SYNTHETIC` rule):
//! - **VHT SU** follows IEEE 802.11-2020 §9.4.1.51 and is exercised by
//!   round-trip and spec-table unit tests on `SYNTHETIC` frames. It has not yet
//!   been cross-checked against a real over-the-air capture or an independent
//!   decoder (`WiPiCap` / `Wi-BFI`), so decoded angle *values* are `CLAIMED`
//!   until a captured frame decodes identically under a second tool.
//! - **VHT MU** codebook-1 angle widths are not encoded here because the author
//!   could not verify them; that case returns [`CbrError::Unsupported`].
//! - **HE** and **EHT** angle bitstreams are [`CbrError::Unsupported`].

#![cfg(feature = "std")]
// `phi`/`psi` and `nr`/`nc` are the IEEE 802.11 names for these quantities;
// renaming them for lint purposes would obscure the spec mapping.
#![allow(clippy::similar_names)]

/// Errors from CBR parsing. All are recoverable — the caller drops the frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CbrError {
    /// Frame body too short for the field being read.
    Truncated {
        /// Bytes required to read the field.
        need: usize,
        /// Bytes actually present.
        have: usize,
    },
    /// Category/Action octets are not a supported beamforming report.
    NotBeamforming {
        /// Action-frame category octet.
        category: u8,
        /// Action octet within the category.
        action: u8,
    },
    /// A dimension field is outside 802.11 limits (Nr/Nc in `1..=4`, `Nc<=Nr`).
    BadDimension {
        /// Number of rows (receive chains) reported.
        nr: u8,
        /// Number of columns (space-time streams) reported.
        nc: u8,
    },
    /// A recognized-but-not-implemented format (HE/EHT angles, MU codebook 1).
    Unsupported(&'static str),
}

impl core::fmt::Display for CbrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated { need, have } => {
                write!(f, "truncated CBR: need {need} bytes, have {have}")
            }
            Self::NotBeamforming { category, action } => {
                write!(f, "not a beamforming report: category={category} action={action}")
            }
            Self::BadDimension { nr, nc } => write!(f, "bad CBR dimension nr={nr} nc={nc}"),
            Self::Unsupported(w) => write!(f, "unsupported CBR format: {w}"),
        }
    }
}

impl std::error::Error for CbrError {}

/// Channel width reported in the VHT MIMO Control field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelWidth {
    /// 20 MHz.
    W20,
    /// 40 MHz.
    W40,
    /// 80 MHz.
    W80,
    /// 160 MHz (or 80+80).
    W160,
}

impl ChannelWidth {
    fn from_bits(v: u8) -> Self {
        match v & 0x3 {
            0 => Self::W20,
            1 => Self::W40,
            2 => Self::W80,
            _ => Self::W160,
        }
    }

    /// Nominal channel width in MHz, for logging and link metadata.
    #[must_use]
    pub fn mhz(self) -> u16 {
        match self {
            Self::W20 => 20,
            Self::W40 => 40,
            Self::W80 => 80,
            Self::W160 => 160,
        }
    }
}

/// Subcarrier grouping Ng (1, 2 or 4 subcarriers per feedback point).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grouping {
    /// One feedback point per subcarrier.
    Ng1,
    /// One feedback point per two subcarriers.
    Ng2,
    /// One feedback point per four subcarriers.
    Ng4,
}

impl Grouping {
    fn from_bits(v: u8) -> Result<Self, CbrError> {
        match v & 0x3 {
            0 => Ok(Self::Ng1),
            1 => Ok(Self::Ng2),
            2 => Ok(Self::Ng4),
            _ => Err(CbrError::Unsupported("reserved grouping Ng=3")),
        }
    }
}

/// Parsed VHT MIMO Control field (802.11-2020 §9.4.1.50): 24 bits / 3 octets,
/// packed least-significant-bit first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VhtMimoControl {
    /// Number of columns Nc (space-time streams fed back), `1..=4`.
    pub nc: u8,
    /// Number of rows Nr (beamformee receive chains), `1..=4`.
    pub nr: u8,
    /// Reported channel width.
    pub width: ChannelWidth,
    /// Subcarrier grouping Ng.
    pub grouping: Grouping,
    /// Codebook-information bit (selects the angle quantization widths).
    pub codebook: bool,
    /// Feedback type: `false` = SU, `true` = MU.
    pub mu: bool,
    /// Remaining feedback segments (segmentation, 802.11 §9.4.1.50).
    pub remaining_segments: u8,
    /// First-feedback-segment flag.
    pub first_segment: bool,
    /// Sounding dialog token this report answers.
    pub sounding_token: u8,
}

fn take_u8(v: u32, shift: u32, width: u32) -> u8 {
    let mask = (1u32 << width) - 1;
    u8::try_from((v >> shift) & mask).expect("field width <= 8 bits")
}

impl VhtMimoControl {
    fn parse(b: &[u8]) -> Result<Self, CbrError> {
        if b.len() < 3 {
            return Err(CbrError::Truncated { need: 3, have: b.len() });
        }
        let v = u32::from(b[0]) | (u32::from(b[1]) << 8) | (u32::from(b[2]) << 16);
        let nc = take_u8(v, 0, 3) + 1;
        let nr = take_u8(v, 3, 3) + 1;
        if !(1..=4).contains(&nr) || !(1..=4).contains(&nc) || nc > nr {
            return Err(CbrError::BadDimension { nr, nc });
        }
        Ok(Self {
            nc,
            nr,
            width: ChannelWidth::from_bits(take_u8(v, 6, 2)),
            grouping: Grouping::from_bits(take_u8(v, 8, 2))?,
            codebook: take_u8(v, 10, 1) == 1,
            mu: take_u8(v, 11, 1) == 1,
            remaining_segments: take_u8(v, 12, 3),
            first_segment: take_u8(v, 15, 1) == 1,
            sounding_token: take_u8(v, 18, 6),
        })
    }

    /// `(psi_bits, phi_bits)` per 802.11-2020 Table 9-92, verified rows only.
    fn angle_bits(self) -> Result<(u32, u32), CbrError> {
        Ok(match (self.mu, self.codebook) {
            (false, false) => (5, 7), // SU codebook 0
            // SU codebook 1 and MU codebook 0 share the (7, 9) widths.
            (false, true) | (true, false) => (7, 9),
            (true, true) => {
                return Err(CbrError::Unsupported("MU codebook 1 angle widths unverified"))
            }
        })
    }
}

/// Number of `Phi` angles (equal to the number of `Psi` angles) per subcarrier
/// for an `(Nr, Nc)` feedback matrix.
///
/// From the ordered Givens decomposition (802.11-2020 §19.3.12.3.6):
/// `sum_{i=1}^{min(Nc, Nr-1)} (Nr - i)`. Verified against the standard's angle
/// table in unit tests.
#[must_use]
pub fn angles_per_subcarrier(nr: u8, nc: u8) -> usize {
    let last = core::cmp::min(nc, nr.saturating_sub(1));
    (1..=last).map(|i| usize::from(nr - i)).sum()
}

/// Number of subcarriers Ns carrying angles, from bandwidth and grouping
/// (802.11-2020 Table 9-91, VHT).
fn num_subcarriers(width: ChannelWidth, ng: Grouping) -> usize {
    use ChannelWidth::{W160, W20, W40, W80};
    use Grouping::{Ng1, Ng2, Ng4};
    // Verbatim Table 9-91; distinct (width, grouping) rows are kept separate for
    // legibility even where two happen to share a subcarrier count.
    #[allow(clippy::match_same_arms)]
    match (width, ng) {
        (W20, Ng1) => 52,
        (W20, Ng2) => 30,
        (W20, Ng4) => 16,
        (W40, Ng1) => 108,
        (W40, Ng2) => 58,
        (W40, Ng4) => 30,
        (W80, Ng1) => 234,
        (W80, Ng2) => 122,
        (W80, Ng4) => 62,
        (W160, Ng1) => 468,
        (W160, Ng2) => 244,
        (W160, Ng4) => 124,
    }
}

/// A fully decoded VHT compressed beamforming report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VhtBeamform {
    /// Parsed MIMO Control field.
    pub control: VhtMimoControl,
    /// Average SNR per space-time stream, raw `i8` (0.25 dB units per spec).
    pub avg_snr: Vec<i8>,
    /// Number of subcarriers carrying angles.
    pub num_subcarriers: usize,
    /// `Phi` (= `Psi`) angle count per subcarrier.
    pub angles_per_sub: usize,
    /// Bit width of each `Psi` code.
    pub psi_bits: u32,
    /// Bit width of each `Phi` code.
    pub phi_bits: u32,
    /// `Phi` codes, flattened as `phi[sub * angles_per_sub + k]`.
    pub phi: Vec<u16>,
    /// `Psi` codes, same layout as [`Self::phi`].
    pub psi: Vec<u16>,
}

impl VhtBeamform {
    /// Dequantize a `Phi` code to radians: `phi = (k + 1/2) * pi / 2^(bphi-1)`,
    /// spanning `(0, 2*pi)`.
    #[must_use]
    pub fn phi_radians(&self, code: u16) -> f64 {
        (f64::from(code) + 0.5) * core::f64::consts::PI / f64::from(1u32 << (self.phi_bits - 1))
    }

    /// Dequantize a `Psi` code to radians: `psi = (k + 1/2) * pi / 2^(bpsi+1)`,
    /// spanning `(0, pi/2)`.
    #[must_use]
    pub fn psi_radians(&self, code: u16) -> f64 {
        (f64::from(code) + 0.5) * core::f64::consts::PI / f64::from(1u32 << (self.psi_bits + 1))
    }
}

/// Least-significant-bit-first bit reader (802.11 on-air order: B0 = LSB of
/// octet 0). Reads at most 16 bits per call.
struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read(&mut self, count: u32) -> Result<u16, CbrError> {
        let mut out = 0u16;
        for i in 0..count {
            let byte_idx = self.pos / 8;
            if byte_idx >= self.bytes.len() {
                return Err(CbrError::Truncated { need: byte_idx + 1, have: self.bytes.len() });
            }
            let one_bit = (self.bytes[byte_idx] >> (self.pos % 8)) & 1;
            out |= u16::from(one_bit) << i;
            self.pos += 1;
        }
        Ok(out)
    }
}

/// Parse a VHT Compressed Beamforming Action frame body.
///
/// `body` starts at the Category octet:
/// `[Category=21][VHTAction=0][MIMO ctl x3][avg SNR xNc][angle bitstream…]`.
///
/// # Errors
/// Returns [`CbrError`] when `body` is truncated, is not a VHT compressed
/// beamforming report, carries an out-of-range dimension, or uses a format that
/// is recognized but not yet decoded (MU codebook 1).
pub fn parse_vht_action(body: &[u8]) -> Result<VhtBeamform, CbrError> {
    if body.len() < 2 {
        return Err(CbrError::Truncated { need: 2, have: body.len() });
    }
    let (category, action) = (body[0], body[1]);
    // Category 21 = VHT, VHT Action 0 = Compressed Beamforming.
    if category != 21 || action != 0 {
        return Err(CbrError::NotBeamforming { category, action });
    }
    parse_vht_report(&body[2..])
}

/// Parse the VHT report starting at the MIMO Control field (no Category/Action).
///
/// # Errors
/// Same conditions as [`parse_vht_action`].
pub fn parse_vht_report(rep: &[u8]) -> Result<VhtBeamform, CbrError> {
    let control = VhtMimoControl::parse(rep)?;
    if control.mu && control.codebook {
        return Err(CbrError::Unsupported("MU codebook 1 angle widths unverified"));
    }
    let (psi_bits, phi_bits) = control.angle_bits()?;
    let nc = usize::from(control.nc);

    let snr_start = 3;
    let snr_end = snr_start + nc;
    if rep.len() < snr_end {
        return Err(CbrError::Truncated { need: snr_end, have: rep.len() });
    }
    let avg_snr: Vec<i8> = rep[snr_start..snr_end].iter().map(|&b| b.cast_signed()).collect();

    let ns = num_subcarriers(control.width, control.grouping);
    let per = angles_per_subcarrier(control.nr, control.nc);
    let mut reader = BitReader::new(&rep[snr_end..]);

    let mut phi = Vec::with_capacity(ns * per);
    let mut psi = Vec::with_capacity(ns * per);
    // Per subcarrier the bitstream is grouped by Givens level i=1..min(Nc,Nr-1):
    // (Nr-i) Phi codes then (Nr-i) Psi codes. Flattened here into phi[]/psi[]
    // in that order; (nr,nc) recovers the level structure.
    let last = core::cmp::min(control.nc, control.nr.saturating_sub(1));
    for _ in 0..ns {
        for i in 1..=last {
            let level_count = control.nr - i;
            for _ in 0..level_count {
                phi.push(reader.read(phi_bits)?);
            }
            for _ in 0..level_count {
                psi.push(reader.read(psi_bits)?);
            }
        }
    }

    debug_assert_eq!(phi.len(), ns * per);
    debug_assert_eq!(psi.len(), ns * per);

    Ok(VhtBeamform {
        control,
        avg_snr,
        num_subcarriers: ns,
        angles_per_sub: per,
        psi_bits,
        phi_bits,
        phi,
        psi,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        angles_per_subcarrier, parse_vht_action, CbrError, ChannelWidth, Grouping, VhtMimoControl,
    };

    /// Build a VHT MIMO Control (3 octets, LSB-first) from fields.
    fn mk_ctl(nc: u8, nr: u8, width: u8, ng: u8, codebook: u8, mu: u8, token: u8) -> [u8; 3] {
        let v: u32 = u32::from(nc - 1)
            | (u32::from(nr - 1) << 3)
            | (u32::from(width) << 6)
            | (u32::from(ng) << 8)
            | (u32::from(codebook) << 10)
            | (u32::from(mu) << 11)
            | (u32::from(token) << 18);
        v.to_le_bytes()[..3].try_into().expect("3 bytes")
    }

    /// LSB-first bit packer mirroring the decoder, for building test frames.
    #[derive(Default)]
    struct BitPacker {
        out: Vec<u8>,
        acc: u32,
        held: u32,
    }
    impl BitPacker {
        fn push(&mut self, value: u16, width: u32) {
            let mask = (1u32 << width) - 1;
            self.acc |= (u32::from(value) & mask) << self.held;
            self.held += width;
            while self.held >= 8 {
                self.out.push(u8::try_from(self.acc & 0xff).expect("masked"));
                self.acc >>= 8;
                self.held -= 8;
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.held > 0 {
                self.out.push(u8::try_from(self.acc & 0xff).expect("masked"));
            }
            self.out
        }
    }

    #[test]
    fn angle_count_table_matches_spec() {
        // 802.11-2020 Table 9-92 Na/2 (Phi count) for each (Nr, Nc).
        let cases = [
            ((2, 1), 1),
            ((2, 2), 1),
            ((3, 1), 2),
            ((3, 2), 3),
            ((3, 3), 3),
            ((4, 1), 3),
            ((4, 2), 5),
            ((4, 3), 6),
            ((4, 4), 6),
        ];
        for ((nr, nc), expect) in cases {
            assert_eq!(angles_per_subcarrier(nr, nc), expect, "Nr{nr}xNc{nc}");
        }
    }

    #[test]
    fn mimo_control_roundtrip() {
        let raw = mk_ctl(2, 2, 2, 0, 0, 0, 42); // 2x2, 80MHz, Ng1, SU cb0
        let c = VhtMimoControl::parse(&raw).unwrap();
        assert_eq!(c.nc, 2);
        assert_eq!(c.nr, 2);
        assert_eq!(c.width, ChannelWidth::W80);
        assert_eq!(c.grouping, Grouping::Ng1);
        assert!(!c.codebook && !c.mu);
        assert_eq!(c.sounding_token, 42);
        assert_eq!(c.angle_bits().unwrap(), (5, 7));
    }

    #[test]
    fn rejects_non_beamforming() {
        assert!(matches!(
            parse_vht_action(&[0x00, 0x00, 0, 0, 0]),
            Err(CbrError::NotBeamforming { .. })
        ));
    }

    #[test]
    fn mu_codebook1_unsupported_not_guessed() {
        let raw = mk_ctl(2, 2, 2, 0, 1, 1, 0); // MU + codebook1
        let mut body = vec![21u8, 0];
        body.extend_from_slice(&raw);
        body.extend_from_slice(&[0; 64]);
        assert_eq!(
            parse_vht_action(&body),
            Err(CbrError::Unsupported("MU codebook 1 angle widths unverified"))
        );
    }

    /// End-to-end: synthesize a 2x2/80MHz/Ng1/SU-cb0 report with known angle
    /// codes, decode it, and confirm every code and the SNR round-trips.
    #[test]
    fn vht_2x2_roundtrip() {
        let (nr, nc) = (2u8, 2u8);
        let ctl = mk_ctl(nc, nr, 2, 0, 0, 0, 7); // 80MHz Ng1
        let ns = 234usize;
        let per = angles_per_subcarrier(nr, nc); // 1
        let (psi_bits, phi_bits) = (5u32, 7u32);

        let phi_codes: Vec<u16> =
            (0..ns * per).map(|i| u16::try_from(i * 3 % (1 << phi_bits)).unwrap()).collect();
        let psi_codes: Vec<u16> =
            (0..ns * per).map(|i| u16::try_from(i * 5 % (1 << psi_bits)).unwrap()).collect();

        let mut packer = BitPacker::default();
        for s in 0..ns {
            for k in 0..per {
                packer.push(phi_codes[s * per + k], phi_bits);
            }
            for k in 0..per {
                packer.push(psi_codes[s * per + k], psi_bits);
            }
        }

        let mut body = vec![21u8, 0];
        body.extend_from_slice(&ctl);
        body.extend_from_slice(&[10i8.cast_unsigned(), (-4i8).cast_unsigned()]); // avg SNR
        body.extend_from_slice(&packer.finish());

        let r = parse_vht_action(&body).unwrap();
        assert_eq!(r.num_subcarriers, ns);
        assert_eq!(r.angles_per_sub, per);
        assert_eq!(r.avg_snr, vec![10, -4]);
        assert_eq!(r.phi, phi_codes);
        assert_eq!(r.psi, psi_codes);

        let two_pi = 2.0 * core::f64::consts::PI;
        for &c in &r.phi {
            let a = r.phi_radians(c);
            assert!(a > 0.0 && a < two_pi, "phi {a} out of (0,2pi)");
        }
        for &c in &r.psi {
            let a = r.psi_radians(c);
            assert!(a > 0.0 && a < core::f64::consts::FRAC_PI_2, "psi {a} out of (0,pi/2)");
        }
    }

    #[test]
    fn truncated_stream_errs() {
        let ctl = mk_ctl(2, 2, 0, 0, 0, 0, 0); // 20MHz Ng1 -> 52 subs
        let mut body = vec![21u8, 0];
        body.extend_from_slice(&ctl);
        body.extend_from_slice(&[0, 0]); // SNR
        body.extend_from_slice(&[0u8; 4]); // far too few angle bytes
        assert!(matches!(parse_vht_action(&body), Err(CbrError::Truncated { .. })));
    }
}
