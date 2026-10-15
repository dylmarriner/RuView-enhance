//! Cross-format tests: detection, dispatch, streaming and fuzz-style
//! robustness (random / truncated / mutated input must never panic and every
//! reader must terminate).

use super::*;
use std::io::Cursor;

fn intel_fixture() -> Vec<u8> {
    let csi: Vec<(i8, i8)> = (0..30 * 6).map(|n| (n as i8, (n as i8).wrapping_neg())).collect();
    let mut f = Vec::new();
    for n in 0..4u32 {
        f.extend(intel5300::testutil::encode_record(
            n * 1000, n as u16, 3, 2, [40, 41, 42], -95, 30, 0b100100, 0, &csi,
        ));
    }
    f
}

fn atheros_fixture(big: bool) -> Vec<u8> {
    let vals: Vec<(i16, i16)> = (0..56 * 2 * 2).map(|n| (n as i16 - 100, 50 - n as i16)).collect();
    let csi = atheros::testutil::pack(&vals);
    let mut f = if big { vec![0xFF] } else { Vec::new() };
    for n in 0..3u64 {
        f.extend(atheros::testutil::record(big, n * 100, 2437, -95, 0, 56, 2, 2, 30, &csi, &[7; 9]));
    }
    f
}

fn nexmon_fixture() -> Vec<u8> {
    use nexmon::testutil::*;
    let words: Vec<u32> = (0..64u32).map(|k| k | (k << 16)).collect();
    let mut f = pcap_header(false, false, 1);
    for n in 0..3u16 {
        let p = nexmon_payload(0xd0, 8, n, 0, 0x1006, [0x65, 0x00], &words);
        f.extend(record(false, 1000 + n as u32, 0, &eth_udp(5500, &p)));
    }
    f
}

fn esp32_fixture() -> Vec<u8> {
    let row = "CSI_DATA,AP,24:6f:28:aa:bb:cc,-55,11,1,0,0,1,1,1,0,0,0,-93,0,6,0,1234567,0,100,0,8,0,\"[1,2,3,4,5,6,7,8]\"\n";
    row.repeat(3).into_bytes()
}

fn collect(format: CsiFormat, data: &[u8]) -> Vec<Result<crate::CsiFrame, FormatError>> {
    match open_reader(format, Cursor::new(data.to_vec()), &ReaderOptions::default()) {
        Ok(it) => it.take(10_000).collect(),
        Err(e) => vec![Err(e)],
    }
}

#[test]
fn detection_by_content() {
    let cases: Vec<(Vec<u8>, CsiFormat)> = vec![
        (intel_fixture(), CsiFormat::Intel5300),
        (atheros_fixture(false), CsiFormat::Atheros),
        (atheros_fixture(true), CsiFormat::Atheros),
        (nexmon_fixture(), CsiFormat::NexmonPcap),
        (esp32_fixture(), CsiFormat::Esp32Csv),
    ];
    for (bytes, want) in cases {
        assert_eq!(detect_format_bytes(&bytes).unwrap(), Some(want), "{want}");
        // And each decodes at least one frame through the dispatcher.
        let v = collect(want, &bytes);
        assert!(v.iter().any(|r| r.is_ok()), "{want}: {v:?}");
    }
    assert_eq!(detect_format_bytes(&[]).unwrap(), None);
    assert_eq!(detect_format_bytes(&[0u8; 200]).unwrap(), None);
}

#[test]
fn full_decode_counts_match_fixtures() {
    assert_eq!(collect(CsiFormat::Intel5300, &intel_fixture()).len(), 4);
    assert_eq!(collect(CsiFormat::Atheros, &atheros_fixture(false)).len(), 3);
    assert_eq!(collect(CsiFormat::Atheros, &atheros_fixture(true)).len(), 3);
    assert_eq!(collect(CsiFormat::NexmonPcap, &nexmon_fixture()).len(), 3);
    assert_eq!(collect(CsiFormat::Esp32Csv, &esp32_fixture()).len(), 3);
    for fmt in [CsiFormat::Intel5300, CsiFormat::Atheros, CsiFormat::NexmonPcap, CsiFormat::Esp32Csv] {
        let bytes = match fmt {
            CsiFormat::Intel5300 => intel_fixture(),
            CsiFormat::Atheros => atheros_fixture(false),
            CsiFormat::NexmonPcap => nexmon_fixture(),
            _ => esp32_fixture(),
        };
        assert!(collect(fmt, &bytes).iter().all(|r| r.is_ok()), "{fmt}");
    }
}

#[test]
fn pcapng_and_unsupported_formats_give_clear_errors() {
    let e = detect_format_bytes(&[0x0A, 0x0D, 0x0D, 0x0A, 0, 0, 0, 0]).unwrap_err();
    assert!(e.to_string().contains("pcapng"));
    for f in [CsiFormat::FeitCsi, CsiFormat::PicoScenes] {
        let e = match open_reader(f, Cursor::new(vec![0u8; 4]), &ReaderOptions::default()) {
            Err(e) => e,
            Ok(_) => panic!("{f} should be unsupported"),
        };
        assert!(matches!(e, FormatError::UnsupportedFormat { .. }));
        assert!(e.to_string().contains(f.id()));
        assert!(e.is_fatal());
    }
}

#[test]
fn extension_fallback_and_ambiguous_dat() {
    use std::path::Path;
    assert_eq!(detect_format_from_name(Path::new("a.pcap")), Some(CsiFormat::NexmonPcap));
    assert_eq!(detect_format_from_name(Path::new("a.csv")), Some(CsiFormat::Esp32Csv));
    assert_eq!(detect_format_from_name(Path::new("run.csi")), Some(CsiFormat::PicoScenes));
    assert_eq!(detect_format_from_name(Path::new("feitcsi_001.dat")), Some(CsiFormat::FeitCsi));
    assert_eq!(detect_format_from_name(Path::new("log.dat")), None);
    assert_eq!(detect_format_from_name(Path::new("noext")), None);
}

#[test]
fn open_path_roundtrip_on_disk() {
    let dir = std::env::temp_dir().join(format!("ruview-formats-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("capture.dat");
    std::fs::write(&p, intel_fixture()).unwrap();
    let (fmt, it) = open_path(&p, &ReaderOptions::default()).unwrap();
    assert_eq!(fmt, CsiFormat::Intel5300);
    assert_eq!(it.filter(|r| r.is_ok()).count(), 4);
    assert_eq!(detect_format(&p).unwrap(), CsiFormat::Intel5300);

    let unknown = dir.join("mystery.dat");
    std::fs::write(&unknown, [0u8; 64]).unwrap();
    assert!(matches!(detect_format(&unknown), Err(FormatError::UnknownFormat(_))));

    let feit = dir.join("feitcsi_x.dat");
    std::fs::write(&feit, [0u8; 64]).unwrap();
    assert!(matches!(
        open_path(&feit, &ReaderOptions::default()),
        Err(FormatError::UnsupportedFormat { .. })
    ));
    assert!(matches!(
        open_path(dir.join("missing.dat"), &ReaderOptions::default()),
        Err(FormatError::Io(_))
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn streaming_does_not_buffer_whole_input() {
    // A reader that yields a long synthetic Intel 5300 stream lazily: if the
    // implementation tried to read-to-end this would allocate ~GBs / never end.
    struct Endless {
        rec: Vec<u8>,
        pos: usize,
        left: usize,
    }
    impl std::io::Read for Endless {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.left == 0 {
                return Ok(0);
            }
            let n = out.len().min(self.rec.len() - self.pos);
            out[..n].copy_from_slice(&self.rec[self.pos..self.pos + n]);
            self.pos += n;
            if self.pos == self.rec.len() {
                self.pos = 0;
                self.left -= 1;
            }
            Ok(n)
        }
    }
    let csi = vec![(1i8, 2i8); 30];
    let rec = intel5300::testutil::encode_record(1, 1, 1, 1, [30, 0, 0], -92, 10, 0, 0, &csi);
    let src = Endless { rec, pos: 0, left: 200_000 };
    let mut it = open_reader(CsiFormat::Intel5300, src, &ReaderOptions::default()).unwrap();
    // Pull a few frames only; must return immediately.
    for _ in 0..5 {
        assert!(it.next().unwrap().is_ok());
    }
}

// ---- fuzz-style robustness -------------------------------------------------

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

const ALL: [CsiFormat; 4] = [
    CsiFormat::Intel5300,
    CsiFormat::Atheros,
    CsiFormat::NexmonPcap,
    CsiFormat::Esp32Csv,
];

fn fixtures() -> Vec<(CsiFormat, Vec<u8>)> {
    vec![
        (CsiFormat::Intel5300, intel_fixture()),
        (CsiFormat::Atheros, atheros_fixture(false)),
        (CsiFormat::Atheros, atheros_fixture(true)),
        (CsiFormat::NexmonPcap, nexmon_fixture()),
        (CsiFormat::Esp32Csv, esp32_fixture()),
    ]
}

#[test]
fn fuzz_random_bytes_never_panic() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for fmt in ALL {
        for round in 0..400 {
            let len = (rng.next() % 3000) as usize;
            let data = rng.bytes(len);
            let _ = collect(fmt, &data);
            let _ = detect_format_bytes(&data);
            if round % 50 == 0 {
                // Biased starts that pass each format's first sanity gate.
                let mut d = match fmt {
                    CsiFormat::NexmonPcap => nexmon_fixture()[..24].to_vec(),
                    CsiFormat::Esp32Csv => b"CSI_DATA,AP,aa:bb,\"[".to_vec(),
                    CsiFormat::Intel5300 => vec![0, 0x80, 0xBB],
                    _ => vec![0xFF, 0x00, 0x20],
                };
                d.extend(rng.bytes(len));
                let _ = collect(fmt, &d);
            }
        }
    }
}

#[test]
fn fuzz_every_truncation_point_never_panics_and_terminates() {
    for (fmt, bytes) in fixtures() {
        for cut in 0..=bytes.len() {
            let v = collect(fmt, &bytes[..cut]);
            assert!(v.len() < 10_000, "{fmt} cut {cut} did not terminate");
            // Detection on a prefix must also be safe.
            let _ = detect_format_bytes(&bytes[..cut]);
        }
    }
}

#[test]
fn fuzz_mutated_fixtures_never_panic() {
    let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
    for (fmt, bytes) in fixtures() {
        for _ in 0..600 {
            let mut m = bytes.clone();
            let flips = 1 + (rng.next() % 8) as usize;
            for _ in 0..flips {
                let i = (rng.next() as usize) % m.len();
                m[i] = rng.next() as u8;
            }
            if rng.next() % 4 == 0 {
                let n = (rng.next() as usize) % m.len();
                m.truncate(n);
            }
            let v = collect(fmt, &m);
            assert!(v.len() < 10_000);
            // Fatal errors must be final: nothing may follow one.
            if let Some(pos) = v.iter().position(|r| matches!(r, Err(e) if e.is_fatal())) {
                assert_eq!(pos, v.len() - 1, "{fmt}: items after fatal error");
            }
        }
    }
}

#[test]
fn recoverable_errors_always_make_progress() {
    // A stream of all-zero bytes must terminate quickly for every format.
    for fmt in ALL {
        let data = vec![0u8; 8_000];
        let v = collect(fmt, &data);
        assert!(v.len() < 10_000, "{fmt}");
    }
}

#[test]
fn frame_invariants_hold_on_valid_fixtures() {
    for (fmt, bytes) in fixtures() {
        for f in collect(fmt, &bytes).into_iter().flatten() {
            let m = &f.metadata;
            assert_eq!(
                m.n_antennas as usize * m.n_subcarriers as usize,
                f.subcarriers.len(),
                "{fmt}"
            );
            assert!(f.subcarrier_count() <= MAX_VALUES_PER_FRAME);
        }
    }
}
