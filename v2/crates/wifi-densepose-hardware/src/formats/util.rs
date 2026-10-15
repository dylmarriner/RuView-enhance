//! Small bounded-I/O helpers shared by the format readers.
//!
//! Everything here is written so that a hostile or truncated file can never
//! cause an unbounded allocation or a panic.

use chrono::{DateTime, Utc};
use std::io::{self, BufRead, Read};

/// Result of trying to fill a fixed-size buffer from a reader.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Fill {
    /// Zero bytes were available (clean end of stream).
    Eof,
    /// Buffer completely filled.
    Full,
    /// Stream ended after this many (non-zero) bytes: a truncated record.
    Partial(usize),
}

/// Fill `buf` completely if possible, distinguishing clean EOF from truncation.
pub(crate) fn read_full<R: Read + ?Sized>(r: &mut R, buf: &mut [u8]) -> io::Result<Fill> {
    let mut got = 0usize;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(if got == buf.len() {
        Fill::Full
    } else if got == 0 {
        Fill::Eof
    } else {
        Fill::Partial(got)
    })
}

/// Read exactly `len` bytes into `buf` (resized to `len`). `len` must already
/// have been validated against a format-specific maximum by the caller.
/// A zero-byte read is reported as a truncation, not a clean EOF.
pub(crate) fn read_body<R: Read + ?Sized>(
    r: &mut R,
    buf: &mut Vec<u8>,
    len: usize,
) -> io::Result<Fill> {
    buf.clear();
    buf.resize(len, 0);
    if len == 0 {
        return Ok(Fill::Full);
    }
    match read_full(r, &mut buf[..])? {
        Fill::Eof => Ok(Fill::Partial(0)),
        other => Ok(other),
    }
}

/// Discard up to `n` bytes without buffering them. Returns bytes skipped.
pub(crate) fn skip<R: Read + ?Sized>(r: &mut R, n: u64) -> io::Result<u64> {
    io::copy(&mut r.take(n), &mut io::sink())
}

/// Outcome of a bounded line read.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Line {
    Eof,
    /// A complete (or final unterminated) line is in the buffer.
    Ok,
    /// The line exceeded the limit; it was consumed and discarded.
    TooLong,
}

/// Read one `\n`-terminated line into `out` holding at most `max` bytes.
/// Longer lines are drained (not stored) so memory stays bounded.
pub(crate) fn read_bounded_line<R: BufRead + ?Sized>(
    r: &mut R,
    out: &mut Vec<u8>,
    max: usize,
) -> io::Result<Line> {
    out.clear();
    let mut overlong = false;
    let mut any = false;
    loop {
        let (consumed, done) = {
            let chunk = match r.fill_buf() {
                Ok(c) => c,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            if chunk.is_empty() {
                break;
            }
            any = true;
            match chunk.iter().position(|&b| b == b'\n') {
                Some(pos) => {
                    if !overlong {
                        if out.len() + pos > max {
                            overlong = true;
                        } else {
                            out.extend_from_slice(&chunk[..pos]);
                        }
                    }
                    (pos + 1, true)
                }
                None => {
                    if !overlong {
                        if out.len() + chunk.len() > max {
                            overlong = true;
                        } else {
                            out.extend_from_slice(chunk);
                        }
                    }
                    (chunk.len(), false)
                }
            }
        };
        r.consume(consumed);
        if done {
            break;
        }
    }
    if overlong {
        out.clear();
        return Ok(Line::TooLong);
    }
    if !any {
        return Ok(Line::Eof);
    }
    Ok(Line::Ok)
}

/// Map a device clock (microseconds since an arbitrary epoch such as NIC
/// boot) onto a `DateTime<Utc>` as an offset from 1970-01-01. This is **not**
/// wall-clock time; consumers needing absolute time must add an offset.
pub(crate) fn device_time(us: u64) -> DateTime<Utc> {
    let secs = (us / 1_000_000).min(i64::MAX as u64 / 2) as i64;
    let nanos = ((us % 1_000_000) * 1_000) as u32;
    DateTime::from_timestamp(secs, nanos).unwrap_or(DateTime::UNIX_EPOCH)
}

/// Wall-clock time from seconds + nanoseconds (pcap), clamped to a valid range.
pub(crate) fn wall_time(secs: i64, nanos: u32) -> DateTime<Utc> {
    DateTime::from_timestamp(secs, nanos.min(999_999_999)).unwrap_or(DateTime::UNIX_EPOCH)
}

pub(crate) fn clamp_i8(v: f64) -> i8 {
    if v.is_nan() {
        0
    } else {
        v.round().clamp(i8::MIN as f64, i8::MAX as f64) as i8
    }
}

/// Channel number to centre frequency (MHz); 0 when unknown.
pub(crate) fn channel_to_freq_mhz(channel: u32) -> u32 {
    match channel {
        1..=13 => 2407 + 5 * channel,
        14 => 2484,
        32..=196 => 5000 + 5 * channel,
        _ => 0,
    }
}

/// Centre frequency to channel number (0 when unknown); used for reporting.
pub fn freq_mhz_to_channel(freq: u32) -> u32 {
    match freq {
        2484 => 14,
        2412..=2472 => (freq - 2407) / 5,
        5160..=5985 => (freq - 5000) / 5,
        _ => 0,
    }
}
