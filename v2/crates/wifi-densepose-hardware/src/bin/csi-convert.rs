//! `csi-convert`: convert third-party CSI captures into RuView's `.csi.jsonl`
//! recording format (the one written by the sensing server's recorder and by
//! `scripts/collect-training-data.py`).
//!
//! Supported inputs: Intel 5300 `.dat`, Atheros CSI Tool `.dat`, Nexmon CSI
//! `.pcap`, ESP32-CSI-Tool / esp-csi CSV. The format is auto-detected from the
//! file content (extension as a fallback) unless `--format` is given.
//!
//! ```text
//! csi-convert capture.dat out.csi.jsonl
//! csi-convert --detect-only capture.pcap
//! csi-convert --format nexmon --nexmon-chip bcm4366c0 in.pcap out.csi.jsonl
//! csi-convert --intel-scale --freq 5180 --start-time 1700000000 in.dat out.csi.jsonl
//! ```
//!
//! The input is streamed; memory use does not depend on the file size.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use wifi_densepose_hardware::formats::{
    detect_format, esp32_csv::IqOrder, intel5300::Intel5300Reader, nexmon::NexmonChip,
    open_path_as, recording, CsiFormat, FormatError, ReaderOptions,
};
use wifi_densepose_hardware::CsiFrame;

#[derive(Parser, Debug)]
#[command(
    name = "csi-convert",
    about = "Convert Intel 5300 / Atheros / Nexmon / ESP32 CSI captures to RuView .csi.jsonl",
    version
)]
struct Cli {
    /// Input capture file.
    input: PathBuf,

    /// Output `.csi.jsonl` path, or `-` for stdout.
    output: Option<PathBuf>,

    /// Input format: intel5300, atheros, esp32-csv, nexmon-pcap (default: auto-detect).
    #[arg(long)]
    format: Option<String>,

    /// Only print the detected format and exit.
    #[arg(long)]
    detect_only: bool,

    /// Channel centre frequency (MHz) for formats that do not carry one.
    #[arg(long)]
    freq: Option<u32>,

    /// Rebase device-clock timestamps (Intel 5300, Atheros, ESP32) so the
    /// first frame is at this Unix time (seconds). Ignored for pcap input.
    #[arg(long)]
    start_time: Option<f64>,

    /// Stop after N frames.
    #[arg(long)]
    limit: Option<u64>,

    /// Do not write the `features.phase` array.
    #[arg(long)]
    no_phase: bool,

    /// Intel 5300: write amplitudes scaled to absolute units (RSSI/AGC/noise).
    #[arg(long)]
    intel_scale: bool,

    /// Nexmon: chip override (bcm4339, bcm43455c0, bcm4358, bcm4366c0).
    #[arg(long)]
    nexmon_chip: Option<String>,

    /// Nexmon: UDP port carrying CSI.
    #[arg(long, default_value_t = 5500)]
    nexmon_port: u16,

    /// ESP32 CSV: treat each buffer pair as [real, imag] instead of [imag, real].
    #[arg(long)]
    esp32_real_first: bool,

    /// Atheros: add +1 after 10-bit sign conversion (CSIKit behaviour).
    #[arg(long)]
    atheros_plus_one: bool,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    force: bool,

    /// Exit non-zero if any record was skipped as corrupt or the file is truncated.
    #[arg(long)]
    strict: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("csi-convert: error: {e}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode, Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let format = match &cli.format {
        Some(s) => CsiFormat::from_id(s).ok_or_else(|| format!("unknown --format {s:?}"))?,
        None => detect_format(&cli.input)?,
    };
    if cli.detect_only {
        println!("{}", format.id());
        return Ok(if format.is_supported() {
            ExitCode::SUCCESS
        } else {
            eprintln!("note: {format} is recognised but not supported for decoding");
            ExitCode::from(2)
        });
    }

    let output = cli
        .output
        .clone()
        .ok_or("an output path (or `-`) is required unless --detect-only is given")?;

    let mut opts = ReaderOptions {
        channel_freq_mhz: cli.freq,
        nexmon_port: cli.nexmon_port,
        atheros_plus_one: cli.atheros_plus_one,
        ..ReaderOptions::default()
    };
    if cli.esp32_real_first {
        opts.esp32_iq_order = IqOrder::RealFirst;
    }
    if let Some(name) = &cli.nexmon_chip {
        opts.nexmon_chip =
            Some(NexmonChip::from_name(name).ok_or_else(|| format!("unknown chip {name:?}"))?);
    }

    let out = open_output(&cli.input, &output, cli.force)?;
    let mut sink = Sink {
        out,
        format,
        rec_opts: recording::RecordingOptions {
            include_phase: !cli.no_phase,
            time_offset_secs: 0.0,
        },
        rebase: if format.has_wall_clock() { None } else { cli.start_time },
        stats: Stats::default(),
    };

    if cli.intel_scale {
        if format != CsiFormat::Intel5300 {
            return Err("--intel-scale only applies to Intel 5300 input".into());
        }
        let file = BufReader::with_capacity(64 * 1024, File::open(&cli.input)?);
        let mut reader = Intel5300Reader::new(file, &opts);
        loop {
            if cli.limit.is_some_and(|l| sink.stats.frames >= l) {
                break;
            }
            match reader.next_record() {
                Ok(Some((rec, ts))) => {
                    let frame = rec.to_frame(ts, cli.freq.unwrap_or(0));
                    let amp_phase = match rec.scaled_csi() {
                        Some(s) => Some((
                            s.iter().map(|&(r, i)| (r * r + i * i).sqrt()).collect(),
                            s.iter().map(|&(r, i)| i.atan2(r)).collect(),
                        )),
                        None => {
                            sink.stats.unscaled += 1;
                            None
                        }
                    };
                    sink.emit(&frame, amp_phase)?;
                }
                Ok(None) => break,
                Err(e) => {
                    let fatal = e.is_fatal();
                    sink.stats.note(&e);
                    if fatal {
                        break;
                    }
                }
            }
        }
    } else {
        let frames = open_path_as(&cli.input, format, &opts)?;
        for item in frames {
            if cli.limit.is_some_and(|l| sink.stats.frames >= l) {
                break;
            }
            match item {
                Ok(frame) => sink.emit(&frame, None)?,
                Err(e) => {
                    let fatal = e.is_fatal();
                    sink.stats.note(&e);
                    if fatal {
                        break;
                    }
                }
            }
        }
    }
    sink.out.flush()?;

    eprintln!(
        "csi-convert: {} -> {} frames written ({} corrupt records skipped{}{})",
        format,
        sink.stats.frames,
        sink.stats.corrupt,
        if sink.stats.truncated { ", input truncated" } else { "" },
        if sink.stats.unscaled > 0 {
            format!(", {} frames could not be scaled", sink.stats.unscaled)
        } else {
            String::new()
        }
    );
    if !format.has_wall_clock() && cli.start_time.is_none() {
        eprintln!(
            "csi-convert: note: timestamps are the capture device's clock (offset from 1970), \
             not wall-clock time; pass --start-time to rebase"
        );
    }
    if sink.stats.frames == 0 {
        return Err("no frames could be decoded from the input".into());
    }
    if cli.strict && (sink.stats.corrupt > 0 || sink.stats.truncated || sink.stats.fatal) {
        return Ok(ExitCode::from(3));
    }
    Ok(ExitCode::SUCCESS)
}

struct Sink {
    out: Box<dyn Write>,
    format: CsiFormat,
    rec_opts: recording::RecordingOptions,
    rebase: Option<f64>,
    stats: Stats,
}

impl Sink {
    fn emit(&mut self, frame: &CsiFrame, amp_phase: Option<(Vec<f64>, Vec<f64>)>) -> io::Result<()> {
        if self.stats.frames == 0 {
            if let Some(t0) = self.rebase {
                let first = frame.metadata.timestamp.timestamp() as f64
                    + frame.metadata.timestamp.timestamp_subsec_nanos() as f64 / 1e9;
                self.rec_opts.time_offset_secs = t0 - first;
            }
        }
        recording::write_frame(&mut self.out, frame, self.format, &self.rec_opts, amp_phase)?;
        self.stats.frames += 1;
        Ok(())
    }
}

#[derive(Default)]
struct Stats {
    frames: u64,
    corrupt: u64,
    unscaled: u64,
    truncated: bool,
    fatal: bool,
}

impl Stats {
    fn note(&mut self, e: &FormatError) {
        eprintln!("csi-convert: warning: {e}");
        match e {
            FormatError::Corrupt { .. } => self.corrupt += 1,
            FormatError::Truncated { .. } => self.truncated = true,
            _ => self.fatal = true,
        }
    }
}

fn open_output(input: &Path, output: &Path, force: bool) -> Result<Box<dyn Write>, Box<dyn std::error::Error>> {
    if output == Path::new("-") {
        return Ok(Box::new(BufWriter::new(io::stdout().lock())));
    }
    if let (Ok(a), Ok(b)) = (input.canonicalize(), output.canonicalize()) {
        if a == b {
            return Err("input and output must be different files".into());
        }
    }
    let mut oo = OpenOptions::new();
    oo.write(true);
    if force {
        oo.create(true).truncate(true);
    } else {
        oo.create_new(true);
    }
    let f = oo
        .open(output)
        .map_err(|e| format!("cannot create {} ({e}); use --force to overwrite", output.display()))?;
    Ok(Box::new(BufWriter::with_capacity(64 * 1024, f)))
}
