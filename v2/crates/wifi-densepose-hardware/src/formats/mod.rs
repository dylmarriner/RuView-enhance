//! Multi-format CSI file ingest.
//!
//! Reads CSI captured by other open-source tools and maps it onto RuView's
//! existing [`CsiFrame`] / [`CsiMetadata`] types so public datasets and other
//! hardware can feed the same pipeline as the ESP32 nodes.
//!
//! | Format | Reader | Status |
//! |--------|--------|--------|
//! | Intel 5300 Linux 802.11n CSI Tool `.dat` | [`intel5300::Intel5300Reader`] | supported |
//! | ESP32-CSI-Tool / Espressif esp-csi console CSV | [`esp32_csv::Esp32CsvReader`] | supported |
//! | Nexmon CSI `.pcap` (int16 and packed-float) | [`nexmon::NexmonPcapReader`] | supported |
//! | Atheros CSI Tool `.dat` | [`atheros::AtherosReader`] | supported |
//! | FeitCSI `.dat`, PicoScenes `.csi` | -- | detection only ([`FormatError::UnsupportedFormat`]) |
//!
//! # Usage
//!
//! ```no_run
//! use wifi_densepose_hardware::formats::{open_path, ReaderOptions};
//!
//! // Auto-detects the format from magic bytes / structure (extension as a
//! // fallback) and streams frames one at a time: memory use is bounded by a
//! // single record, never by the file size.
//! let (format, frames) = open_path("capture.dat", &ReaderOptions::default())?;
//! for frame in frames {
//!     match frame {
//!         Ok(f) => println!("{format}: {} subcarriers", f.subcarrier_count()),
//!         Err(e) if e.is_fatal() => { eprintln!("stopping: {e}"); break; }
//!         Err(e) => eprintln!("skipping bad record: {e}"),
//!     }
//! }
//! # Ok::<(), wifi_densepose_hardware::formats::FormatError>(())
//! ```
//!
//! To convert a file to RuView's `.csi.jsonl` recording format use the
//! `csi-convert` binary (see `docs/csi-formats.md`).
//!
//! # Frame conventions
//!
//! * `CsiFrame::subcarriers` is stream-major: stream `s = tx * n_rx + rx`, and
//!   `n_subcarriers` values per stream. `metadata.n_antennas` is the number of
//!   streams so that `n_antennas * n_subcarriers == subcarriers.len()`
//!   (matching the antenna-major layout `CsiData` expects); the true
//!   TX/RX counts are in `metadata.antenna_config`.
//! * I/Q are the raw integer values from the file (no AGC/RSSI scaling); the
//!   Intel 5300 scaling to absolute units is available as an explicit helper
//!   ([`intel5300::BfeeRecord::scaled_csi`]).
//! * Timestamps are wall-clock for pcap input; for formats that only carry a
//!   device clock (Intel 5300, Atheros, ESP32) the timestamp is the device
//!   clock in microseconds expressed as an offset from 1970-01-01. See
//!   [`CsiFormat::has_wall_clock`].
//! * The `hardware_label` of each format uses the names of
//!   `wifi_densepose_signal::hardware_norm::HardwareType` (`Intel5300`,
//!   `Atheros`, `Esp32S3`, `Generic`); the signal crate decides the actual
//!   normalisation from the subcarrier count (30 -> Intel5300, 56 -> Atheros,
//!   64 -> Esp32S3, otherwise Generic), which this module's outputs are
//!   consistent with.
//!
//! # Safety / robustness
//!
//! Every length taken from a file header is validated against a hard maximum
//! before any allocation, parsers never index without bounds checks, and
//! corrupt or truncated input yields a [`FormatError`] rather than a panic.
//! Fuzz-style tests (random bytes, every truncation point, byte mutation of
//! valid fixtures) assert this for each reader.
//!
//! # Provenance and licensing
//!
//! Format knowledge for Intel 5300, Atheros and Nexmon was cross-checked
//! against CSIKit (<https://github.com/Gi-z/CSIKit>, MIT) and csiread
//! (<https://github.com/citysu/csiread>, MIT); attribution is repeated in each
//! reader's module header. nexmon_csi has no clear licence, so only its
//! publicly documented packet layout was used and the code is original.

pub mod atheros;
pub mod esp32_csv;
pub mod intel5300;
pub mod nexmon;
pub mod recording;
mod util;

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::csi_frame::CsiFrame;

pub use util::freq_mhz_to_channel;

/// Hard cap on subcarrier values (all streams together) in one frame.
pub const MAX_VALUES_PER_FRAME: usize = 4096;
/// Maximum accepted length of one text line (ESP32 CSV).
pub const MAX_LINE_BYTES: usize = 64 * 1024;
/// Maximum accepted pcap packet capture length (classic libpcap snaplen cap).
pub const MAX_PCAP_PACKET_BYTES: usize = 262_144;
/// Number of leading bytes inspected by [`detect_format_bytes`].
pub const DETECT_PREFIX_BYTES: usize = 4096;

/// Errors produced while ingesting a CSI file.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    /// Underlying I/O failure. Fatal.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The stream ends in the middle of a record. Fatal (end of data).
    #[error("{format}: truncated record at byte offset {offset}")]
    Truncated { format: &'static str, offset: u64 },

    /// A single record is malformed but the framing is intact; the reader
    /// has already advanced past it and the next record can be read.
    #[error("{format}: corrupt record at byte offset {offset}: {reason}")]
    Corrupt {
        format: &'static str,
        offset: u64,
        reason: String,
    },

    /// The file/stream header is invalid so nothing can be read. Fatal.
    #[error("{format}: invalid file header: {reason}")]
    BadHeader { format: &'static str, reason: String },

    /// A recognised format that this crate deliberately does not decode.
    #[error("unsupported format {format}: {reason}")]
    UnsupportedFormat { format: String, reason: String },

    /// No format matched the content or the file extension.
    #[error("could not determine CSI file format: {0}")]
    UnknownFormat(String),
}

impl FormatError {
    /// `true` when the reader cannot continue after this error. After a
    /// non-fatal error the iterator may be polled again.
    pub fn is_fatal(&self) -> bool {
        !matches!(self, FormatError::Corrupt { .. })
    }
}

/// CSI file formats known to the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CsiFormat {
    /// Intel 5300 Linux 802.11n CSI Tool `.dat`.
    Intel5300,
    /// Atheros CSI Tool `.dat`.
    Atheros,
    /// ESP32-CSI-Tool or Espressif esp-csi console CSV.
    Esp32Csv,
    /// Nexmon CSI libpcap capture.
    NexmonPcap,
    /// FeitCSI (Intel AX200/AX210) -- detection only.
    FeitCsi,
    /// PicoScenes `.csi` -- detection only.
    PicoScenes,
}

impl CsiFormat {
    /// Short stable identifier (CLI `--format` value, `source_format` field).
    pub fn id(self) -> &'static str {
        match self {
            Self::Intel5300 => "intel5300",
            Self::Atheros => "atheros",
            Self::Esp32Csv => "esp32-csv",
            Self::NexmonPcap => "nexmon-pcap",
            Self::FeitCsi => "feitcsi",
            Self::PicoScenes => "picoscenes",
        }
    }

    /// Parse an identifier produced by [`CsiFormat::id`].
    pub fn from_id(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "intel5300" | "intel" | "iwl5300" => Self::Intel5300,
            "atheros" | "ath" => Self::Atheros,
            "esp32-csv" | "esp32" | "esp32csv" => Self::Esp32Csv,
            "nexmon-pcap" | "nexmon" | "pcap" => Self::NexmonPcap,
            "feitcsi" => Self::FeitCsi,
            "picoscenes" => Self::PicoScenes,
            _ => return None,
        })
    }

    /// `true` when frames can actually be decoded.
    pub fn is_supported(self) -> bool {
        !matches!(self, Self::FeitCsi | Self::PicoScenes)
    }

    /// `true` when the format carries absolute (wall-clock) timestamps.
    pub fn has_wall_clock(self) -> bool {
        matches!(self, Self::NexmonPcap)
    }

    /// Name of the closest `HardwareType` variant in
    /// `wifi_densepose_signal::hardware_norm` (informational).
    pub fn hardware_label(self) -> &'static str {
        match self {
            Self::Intel5300 => "Intel5300",
            Self::Atheros => "Atheros",
            Self::Esp32Csv => "Esp32S3",
            _ => "Generic",
        }
    }

    fn unsupported(self) -> FormatError {
        let reason = match self {
            Self::FeitCsi => {
                "FeitCSI's on-disk layout is not specified in public documentation \
                 precisely enough to implement a parser without guessing; convert it \
                 with FeitCSI's own exporter or the PicoScenes/CSIKit tooling first"
            }
            Self::PicoScenes => {
                "the PicoScenes .csi container is a multi-segment, versioned binary \
                 format (RxSBasic/CSI/ExtraInfo segments) whose specification is tied \
                 to the PicoScenes SDK; not implemented here"
            }
            _ => "no decoder",
        };
        FormatError::UnsupportedFormat {
            format: self.id().to_string(),
            reason: reason.to_string(),
        }
    }
}

impl fmt::Display for CsiFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Reader tuning knobs shared by all formats. Defaults are sensible for
/// unmodified captures.
#[derive(Debug, Clone)]
pub struct ReaderOptions {
    /// Centre frequency (MHz) to stamp on frames from formats that do not
    /// carry it (Intel 5300). `None` leaves it at 0 (unknown).
    pub channel_freq_mhz: Option<u32>,
    /// Nexmon: override the chip inferred from the packet header.
    pub nexmon_chip: Option<nexmon::NexmonChip>,
    /// Nexmon: UDP destination port carrying CSI (default 5500).
    pub nexmon_port: u16,
    /// ESP32 CSV: interleave order of the CSI byte buffer.
    pub esp32_iq_order: esp32_csv::IqOrder,
    /// Atheros: add the `+1` offset some tools apply after sign conversion.
    pub atheros_plus_one: bool,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        Self {
            channel_freq_mhz: None,
            nexmon_chip: None,
            nexmon_port: nexmon::NEXMON_UDP_PORT,
            esp32_iq_order: esp32_csv::IqOrder::ImagFirst,
            atheros_plus_one: false,
        }
    }
}

/// Boxed streaming frame source returned by [`open_reader`] / [`open_path`].
pub type FrameIter<'a> = Box<dyn Iterator<Item = Result<CsiFrame, FormatError>> + 'a>;

/// Detect the format from the first bytes of a file (up to
/// [`DETECT_PREFIX_BYTES`] are inspected). Returns `Ok(None)` when nothing
/// matches; pcapng (not supported) is reported as an error so users get a
/// precise message.
pub fn detect_format_bytes(head: &[u8]) -> Result<Option<CsiFormat>, FormatError> {
    let head = &head[..head.len().min(DETECT_PREFIX_BYTES)];
    if head.len() >= 4 && head[..4] == [0x0A, 0x0D, 0x0D, 0x0A] {
        return Err(FormatError::UnsupportedFormat {
            format: "pcapng".into(),
            reason: "convert to classic pcap first (e.g. `editcap -F pcap in.pcapng out.pcap`)"
                .into(),
        });
    }
    if nexmon::looks_like_pcap(head) {
        return Ok(Some(CsiFormat::NexmonPcap));
    }
    if esp32_csv::looks_like_esp32_csv(head) {
        return Ok(Some(CsiFormat::Esp32Csv));
    }
    if intel5300::looks_like_intel5300(head) {
        return Ok(Some(CsiFormat::Intel5300));
    }
    if atheros::looks_like_atheros(head) {
        return Ok(Some(CsiFormat::Atheros));
    }
    Ok(None)
}

/// Extension / file-name based fallback used when content sniffing fails.
pub fn detect_format_from_name(path: &Path) -> Option<CsiFormat> {
    let name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    let ext = path.extension()?.to_string_lossy().to_ascii_lowercase();
    match ext.as_str() {
        "pcap" => Some(CsiFormat::NexmonPcap),
        "csv" => Some(CsiFormat::Esp32Csv),
        "csi" => Some(CsiFormat::PicoScenes),
        // `.dat` is shared by Intel 5300, Atheros and FeitCSI: only a file
        // name hint can disambiguate without content.
        "dat" if name.contains("feitcsi") || name.contains("feit") => Some(CsiFormat::FeitCsi),
        "dat" if name.contains("intel") || name.contains("5300") => Some(CsiFormat::Intel5300),
        "dat" if name.contains("atheros") || name.contains("ath") => Some(CsiFormat::Atheros),
        _ => None,
    }
}

/// Detect a file's format: magic/structure first, extension second.
pub fn detect_format(path: impl AsRef<Path>) -> Result<CsiFormat, FormatError> {
    let path = path.as_ref();
    let mut f = File::open(path)?;
    let mut head = vec![0u8; DETECT_PREFIX_BYTES];
    let n = read_prefix(&mut f, &mut head)?;
    head.truncate(n);
    match detect_format_bytes(&head)? {
        Some(fmt) => Ok(fmt),
        None => detect_format_from_name(path).ok_or_else(|| {
            FormatError::UnknownFormat(format!(
                "{}: content did not match any known format and the extension is ambiguous",
                path.display()
            ))
        }),
    }
}

fn read_prefix<R: Read>(r: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

/// Open a streaming reader of a known `format` over any byte source.
///
/// Detection-only formats return [`FormatError::UnsupportedFormat`].
pub fn open_reader<'a, R: Read + 'a>(
    format: CsiFormat,
    reader: R,
    opts: &ReaderOptions,
) -> Result<FrameIter<'a>, FormatError> {
    Ok(match format {
        CsiFormat::Intel5300 => Box::new(intel5300::Intel5300Reader::new(reader, opts)),
        CsiFormat::Atheros => Box::new(atheros::AtherosReader::new(reader, opts)),
        CsiFormat::Esp32Csv => Box::new(esp32_csv::Esp32CsvReader::new(
            BufReader::new(reader),
            opts,
        )),
        CsiFormat::NexmonPcap => Box::new(nexmon::NexmonPcapReader::new(reader, opts)?),
        CsiFormat::FeitCsi | CsiFormat::PicoScenes => return Err(format.unsupported()),
    })
}

/// Detect the format of the file at `path` and open a streaming reader.
pub fn open_path(
    path: impl AsRef<Path>,
    opts: &ReaderOptions,
) -> Result<(CsiFormat, FrameIter<'static>), FormatError> {
    let format = detect_format(path.as_ref())?;
    let it = open_path_as(path, format, opts)?;
    Ok((format, it))
}

/// Open the file at `path` as an explicitly chosen `format`.
pub fn open_path_as(
    path: impl AsRef<Path>,
    format: CsiFormat,
    opts: &ReaderOptions,
) -> Result<FrameIter<'static>, FormatError> {
    if !format.is_supported() {
        return Err(format.unsupported());
    }
    let mut f = File::open(path.as_ref())?;
    f.seek(SeekFrom::Start(0))?;
    open_reader(format, BufReader::with_capacity(64 * 1024, f), opts)
}

#[cfg(test)]
mod tests;
