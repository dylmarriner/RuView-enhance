//! ESP32 CSV readers: ESP32-CSI-Tool and Espressif esp-csi console output.
//!
//! Both emit one line per packet beginning with `CSI_DATA` whose last column
//! is a bracketed list of signed integers, the raw CSI buffer:
//!
//! ```text
//! type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data
//! CSI_DATA,AP,24:6f:28:aa:bb:cc,-55,11,1,0,0,1,1,1,0,0,0,-93,0,6,0,1234567,0,100,0,8,0,"[1,2,3,4,5,6,7,8]"
//! ```
//!
//! The Espressif `csi_recv` console example prints the same columns with
//! `id` (a packet counter) in place of `role`. When a header line starting
//! with `type,` is present its column names are used; otherwise the standard
//! 24-column layout above is assumed. Extra, missing or reordered columns are
//! tolerated; only `data` is mandatory.
//!
//! The CSI buffer is a sequence of signed bytes, two per subcarrier. Per the
//! ESP-IDF documentation the first byte of each pair is the **imaginary**
//! part and the second the real part ([`IqOrder::ImagFirst`], the default).
//!
//! No third-party code is used; the formats are documented in the
//! ESP32-CSI-Tool README and the ESP-IDF `wifi_csi_info_t` reference.

use std::io::BufRead;

use super::util::{channel_to_freq_mhz, device_time, read_bounded_line, Line};
use super::{FormatError, ReaderOptions, MAX_LINE_BYTES, MAX_VALUES_PER_FRAME};
use crate::csi_frame::{
    Adr018Flags, AntennaConfig, Bandwidth, CsiFrame, CsiMetadata, PpduType, SubcarrierData,
};

const FORMAT: &str = "esp32-csv";

const DEFAULT_COLUMNS: [&str; 24] = [
    "type",
    "role",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
];

/// Byte order within each I/Q pair of the raw buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IqOrder {
    /// `[imag, real]` per subcarrier (ESP-IDF documented order).
    ImagFirst,
    /// `[real, imag]` per subcarrier.
    RealFirst,
}

/// Content sniffing for [`super::detect_format_bytes`].
pub(crate) fn looks_like_esp32_csv(head: &[u8]) -> bool {
    // Pure printable ASCII/UTF-8 text with a CSI_DATA marker or csv header.
    let probe = &head[..head.len().min(2048)];
    if probe.is_empty() || probe.iter().any(|&b| b == 0 || (b < 0x09) || b == 0x7F) {
        return false;
    }
    let text = String::from_utf8_lossy(probe);
    text.lines().take(16).any(|l| {
        let l = l.trim_start();
        l.starts_with("CSI_DATA,")
            || ((l.starts_with("type,role,") || l.starts_with("type,id,"))
                && l.contains("mac")
                && l.contains("data"))
    })
}

/// Streaming reader. Lines that do not contain a `CSI_DATA,` record (log
/// noise, blank lines, headers) are skipped silently.
pub struct Esp32CsvReader<R> {
    inner: R,
    line_no: u64,
    frame_no: u32,
    buf: Vec<u8>,
    columns: Vec<String>,
    iq: IqOrder,
    freq_override: Option<u32>,
    done: bool,
}

impl<R: BufRead> Esp32CsvReader<R> {
    pub fn new(inner: R, opts: &ReaderOptions) -> Self {
        Self {
            inner,
            line_no: 0,
            frame_no: 0,
            buf: Vec::new(),
            columns: DEFAULT_COLUMNS.iter().map(|s| s.to_string()).collect(),
            iq: opts.esp32_iq_order,
            freq_override: opts.channel_freq_mhz,
            done: false,
        }
    }

    fn parse_line(&mut self, line: &str) -> Result<CsiFrame, String> {
        let open = line.find('[').ok_or("no '[' CSI data list")?;
        let close = line.rfind(']').ok_or("no closing ']' for CSI data list")?;
        if close <= open {
            return Err("malformed CSI data list".into());
        }
        let meta_part = &line[..open];
        let fields: Vec<&str> = meta_part
            .trim_end_matches(|c: char| c == ',' || c == '"' || c.is_whitespace())
            .split(',')
            .map(|s| s.trim().trim_matches('"'))
            .collect();

        let get = |name: &str| -> Option<&str> {
            let idx = self.columns.iter().position(|c| c == name)?;
            fields.get(idx).copied().filter(|s| !s.is_empty())
        };
        let get_i = |name: &str| -> Option<i64> { get(name).and_then(|s| s.parse::<i64>().ok()) };

        // Raw buffer.
        let list = &line[open + 1..close];
        let mut raw: Vec<i16> = Vec::new();
        for tok in list.split(|c: char| c == ',' || c.is_whitespace()) {
            if tok.is_empty() {
                continue;
            }
            if raw.len() >= MAX_VALUES_PER_FRAME * 2 {
                return Err(format!(
                    "CSI list longer than {} values",
                    MAX_VALUES_PER_FRAME * 2
                ));
            }
            let v: i32 = tok
                .parse()
                .map_err(|_| format!("non-integer CSI value {tok:?}"))?;
            if !(i16::MIN as i32..=i16::MAX as i32).contains(&v) {
                return Err(format!("CSI value {v} out of i16 range"));
            }
            raw.push(v as i16);
        }
        if raw.is_empty() {
            return Err("empty CSI data list".into());
        }
        if raw.len() % 2 != 0 {
            return Err(format!("odd number of CSI values ({})", raw.len()));
        }
        let n_sc = raw.len() / 2;
        let half = n_sc as i16 / 2;
        let subcarriers: Vec<SubcarrierData> = raw
            .chunks_exact(2)
            .enumerate()
            .map(|(k, p)| {
                let (i, q) = match self.iq {
                    IqOrder::ImagFirst => (p[1], p[0]),
                    IqOrder::RealFirst => (p[0], p[1]),
                };
                let index = if (k as i16) < half {
                    -(half - k as i16)
                } else {
                    k as i16 - half + 1
                };
                SubcarrierData { i, q, index }
            })
            .collect();

        let mac_last = get("mac")
            .and_then(|m| m.rsplit(':').next())
            .and_then(|b| u8::from_str_radix(b, 16).ok())
            .unwrap_or(0);
        let channel = get_i("channel").unwrap_or(0).clamp(0, 255) as u32;
        let freq = self
            .freq_override
            .unwrap_or_else(|| channel_to_freq_mhz(channel));
        let wide = get_i("bandwidth").unwrap_or(0) == 1 || n_sc > 64;
        let sig_mode = get_i("sig_mode").unwrap_or(0);
        let ts_us = get_i("local_timestamp").unwrap_or(0).max(0) as u64;
        let clamp8 = |v: i64| v.clamp(i8::MIN as i64, i8::MAX as i64) as i8;
        let seq = self.frame_no.wrapping_add(1);

        let frame = CsiFrame {
            metadata: CsiMetadata {
                timestamp: device_time(ts_us),
                node_id: mac_last,
                n_antennas: 1,
                n_subcarriers: n_sc.min(u16::MAX as usize) as u16,
                channel_freq_mhz: freq,
                rssi_dbm: clamp8(get_i("rssi").unwrap_or(0)),
                noise_floor_dbm: clamp8(get_i("noise_floor").unwrap_or(0)),
                bandwidth: if wide { Bandwidth::Bw40 } else { Bandwidth::Bw20 },
                antenna_config: AntennaConfig::default(),
                sequence: seq,
                // sig_mode: 0 = non-HT, 1 = HT, 3 = VHT; all in the HT/legacy bucket.
                ppdu_type: PpduType::HtLegacy,
                adr018_flags: Adr018Flags {
                    bw40: wide,
                    stbc: get_i("stbc").unwrap_or(0) != 0 && sig_mode != 0,
                    ..Adr018Flags::default()
                },
            },
            subcarriers,
        };
        self.frame_no = seq;
        Ok(frame)
    }

    fn read_next(&mut self) -> Result<Option<CsiFrame>, FormatError> {
        loop {
            match read_bounded_line(&mut self.inner, &mut self.buf, MAX_LINE_BYTES)? {
                Line::Eof => {
                    self.done = true;
                    return Ok(None);
                }
                Line::TooLong => {
                    self.line_no += 1;
                    return Err(FormatError::Corrupt {
                        format: FORMAT,
                        offset: self.line_no,
                        reason: format!("line longer than {MAX_LINE_BYTES} bytes"),
                    });
                }
                Line::Ok => {}
            }
            self.line_no += 1;
            let owned = String::from_utf8_lossy(&self.buf).into_owned();
            let line = owned.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with("type,") {
                let cols: Vec<String> = line
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').to_string())
                    .collect();
                if cols.iter().any(|c| c == "data") {
                    self.columns = cols;
                }
                continue;
            }
            let Some(start) = line.find("CSI_DATA,") else {
                continue;
            };
            return match self.parse_line(&line[start..]) {
                Ok(f) => Ok(Some(f)),
                Err(reason) => Err(FormatError::Corrupt {
                    format: FORMAT,
                    // For text formats the "offset" is the 1-based line number.
                    offset: self.line_no,
                    reason,
                }),
            };
        }
    }
}

impl<R: BufRead> Iterator for Esp32CsvReader<R> {
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
mod tests {
    use super::*;

    const HEADER: &str = "type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data";
    const ROW: &str = "CSI_DATA,AP,24:6f:28:aa:bb:cc,-55,11,1,0,0,1,1,1,0,0,0,-93,0,6,0,1234567,0,100,0,8,0,\"[1,2,3,4,5,6,7,-8]\"";

    fn read(s: &str) -> Vec<Result<CsiFrame, FormatError>> {
        Esp32CsvReader::new(s.as_bytes(), &ReaderOptions::default()).collect()
    }

    #[test]
    fn parses_esp32_csi_tool_row_imag_first() {
        let v = read(&format!("{HEADER}\n{ROW}\n"));
        assert_eq!(v.len(), 1);
        let f = v[0].as_ref().unwrap();
        assert_eq!(f.subcarrier_count(), 4);
        // pairs are (imag, real): (1,2) (3,4) (5,6) (7,-8)
        assert_eq!((f.subcarriers[0].q, f.subcarriers[0].i), (1, 2));
        assert_eq!((f.subcarriers[3].q, f.subcarriers[3].i), (7, -8));
        let m = &f.metadata;
        assert_eq!(m.rssi_dbm, -55);
        assert_eq!(m.noise_floor_dbm, -93);
        assert_eq!(m.channel_freq_mhz, 2437);
        assert_eq!(m.node_id, 0xcc);
        assert_eq!(m.timestamp.timestamp_micros(), 1_234_567);
        assert_eq!(m.bandwidth, Bandwidth::Bw20);
    }

    #[test]
    fn parses_without_header_and_ignores_log_noise() {
        let text = format!("I (123) wifi: connected\n\n{ROW}\nrandom text\n{ROW}\n");
        let v = read(&text);
        assert_eq!(v.len(), 2);
        assert_eq!(v[1].as_ref().unwrap().metadata.sequence, 2);
    }

    #[test]
    fn real_first_option() {
        let opts = ReaderOptions {
            esp32_iq_order: IqOrder::RealFirst,
            ..Default::default()
        };
        let v: Vec<_> = Esp32CsvReader::new(ROW.as_bytes(), &opts).collect();
        let f = v[0].as_ref().unwrap();
        assert_eq!((f.subcarriers[0].i, f.subcarriers[0].q), (1, 2));
    }

    #[test]
    fn esp_csi_id_header_variant() {
        let header = HEADER.replace("type,role,", "type,id,");
        let row = ROW.replace("CSI_DATA,AP,", "CSI_DATA,17,");
        let v = read(&format!("{header}\n{row}\n"));
        assert!(v[0].is_ok());
    }

    #[test]
    fn corrupt_rows_are_recoverable() {
        let text = format!(
            "CSI_DATA,AP,aa,-1,1,1,0,0,0,0,0,0,0,0,-90,0,1,0,5,0,1,0,3,0,\"[1,2,3]\"\n{ROW}\nCSI_DATA,x,\"[1,zz]\"\nCSI_DATA,y,\"[99999,1]\"\nCSI_DATA,z,no brackets\n"
        );
        let v = read(&text);
        assert_eq!(v.len(), 5);
        assert!(matches!(v[0], Err(FormatError::Corrupt { .. }))); // odd count
        assert!(v[1].is_ok());
        assert!(v[2].is_err() && v[3].is_err() && v[4].is_err());
        assert!(v.iter().filter_map(|r| r.as_ref().err()).all(|e| !e.is_fatal()));
    }

    #[test]
    fn overlong_line_is_bounded() {
        let mut text = String::from("CSI_DATA,AP,\"[");
        text.push_str(&"1,".repeat(MAX_LINE_BYTES));
        text.push_str("1]\"\n");
        text.push_str(ROW);
        text.push('\n');
        let v = read(&text);
        assert_eq!(v.len(), 2);
        assert!(v[0].is_err());
        assert!(v[1].is_ok());
    }

    #[test]
    fn too_many_values_rejected() {
        let vals = vec!["1"; MAX_VALUES_PER_FRAME * 2 + 2].join(",");
        let v = read(&format!("CSI_DATA,AP,a,\"[{vals}]\"\n"));
        assert!(v[0].is_err());
    }

    #[test]
    fn detection() {
        assert!(looks_like_esp32_csv(ROW.as_bytes()));
        assert!(looks_like_esp32_csv(HEADER.as_bytes()));
        assert!(!looks_like_esp32_csv(&[0u8, 1, 2, 3]));
        assert!(!looks_like_esp32_csv(b"hello,world\n1,2\n"));
    }
}
