# CSI file formats RuView can ingest

The `wifi_densepose_hardware::formats` module (crate
`v2/crates/wifi-densepose-hardware`) reads CSI captured by other open-source
tools and maps it onto RuView's `CsiFrame` / `CsiMetadata` types. The
`csi-convert` binary writes RuView's existing `.csi.jsonl` recording format.

Evidence tags used below: **SYNTHETIC** = tested on fixtures generated in test
code (bytes built by hand from the published layout, plus an independent
hand-computed vector where noted). **MEASURED** = tested on a real capture
(none so far; no real sample under 50 KB with a permissive licence was
found, and no datasets are committed). Nothing here is a claim about sensing
accuracy; it is only about parsing.

## Support matrix

| Format | Status | Evidence |
|--------|--------|----------|
| Intel 5300 Linux 802.11n CSI Tool `.dat` | supported (+ optional absolute scaling) | SYNTHETIC; bit layout hand-checked, scaling checked against an independent Python calculation |
| ESP32-CSI-Tool CSV / Espressif esp-csi `csi_recv` CSV | supported | SYNTHETIC |
| Nexmon CSI `.pcap`, int16 chips (bcm4339, bcm43455c0) | supported | SYNTHETIC |
| Nexmon CSI `.pcap`, packed-float chips (bcm4358, bcm4366c0) | supported | SYNTHETIC; algorithm hand-computed on one vector |
| Atheros CSI Tool `.dat` | supported | SYNTHETIC; two reference implementations disagree on details, see below |
| FeitCSI `.dat` | detection by file name only; decoding returns `UnsupportedFormat` | n/a |
| PicoScenes `.csi` | detection by extension only; decoding returns `UnsupportedFormat` | n/a |
| pcapng | detected, returns `UnsupportedFormat` (convert with `editcap -F pcap`) | n/a |

FeitCSI and PicoScenes are not decoded because their on-disk layouts are not
specified in public documentation in enough detail to implement without
guessing, and a wrong decoder is worse than a clear error.

## Usage

```bash
cd v2
# detect only
cargo run -p wifi-densepose-hardware --bin csi-convert -- --detect-only capture.dat
# convert (format auto-detected)
cargo run -p wifi-densepose-hardware --bin csi-convert -- capture.dat out.csi.jsonl
# Intel 5300 with scaling to absolute units, anchored to a wall-clock start
cargo run -p wifi-densepose-hardware --bin csi-convert -- \
    --intel-scale --freq 5180 --start-time 1700000000 capture.dat out.csi.jsonl
# Nexmon capture from a bcm4366c0 router
cargo run -p wifi-densepose-hardware --bin csi-convert -- \
    --format nexmon-pcap --nexmon-chip bcm4366c0 capture.pcap out.csi.jsonl
```

Library:

```rust
use wifi_densepose_hardware::formats::{open_path, ReaderOptions};
let (format, frames) = open_path("capture.dat", &ReaderOptions::default())?;
for frame in frames { /* Result<CsiFrame, FormatError> */ }
```

Readers are iterators over any `Read`; memory use is bounded by one record
(<= 64 KiB; pcap <= 256 KiB; CSV lines <= 64 KiB), never by the file size.
`FormatError::is_fatal()` distinguishes a skippable corrupt record from the
end of usable data.

## Output: `.csi.jsonl`

One JSON object per line, the same shape as the sensing server's
`RecordedFrame` and `scripts/collect-training-data.py`'s `CsiRecorder`:

```json
{"timestamp": 1700000000.25, "subcarriers": [/* amplitudes */], "rssi": -55.0,
 "noise_floor": -92.0, "features": {"node_id": 0, "n_antennas": 1,
 "n_subcarriers": 30, "freq_mhz": 5180, "channel": 36, "seq": 7,
 "bandwidth_mhz": 20, "antenna_tx": 1, "antenna_rx": 1,
 "source_format": "intel5300", "phase": [/* radians */]}}
```

`subcarriers` is stream-major (`n_antennas * n_subcarriers` amplitudes, stream
`= tx * n_rx + rx`). `--no-phase` omits `features.phase`.

## Per-format notes and known limitations

**Timestamps.** Only pcap carries wall-clock time. Intel 5300 (32-bit µs NIC
clock, unwrapped across rollovers), Atheros (TSF µs) and ESP32
(`local_timestamp`, µs since boot) give a device clock, emitted as an offset
from 1970-01-01. Use `--start-time` to rebase.

**Intel 5300.** No channel information is stored in the file; pass `--freq`.
`rssi_dbm` is the total RSS from the per-chain RSSI/AGC (0 if unavailable).
Raw I/Q are written as the integers in the file; `--intel-scale` writes
amplitudes scaled per the tool's `get_scaled_csi` algorithm (returns no
scaling, and falls back to raw, when RSSI or CSI power is zero). The receive
antenna permutation is applied only when `Nrx == 3` and `antenna_sel` is a
valid permutation. 40 MHz is flagged from bit 11 of `fake_rate_n_flags`.

**ESP32 CSV.** Per the ESP-IDF documentation each pair is `[imag, real]`;
`--esp32-real-first` flips this. `node_id` is the last MAC byte. Rows without
a bracketed data list are skipped; malformed rows are reported and skipped.

**Nexmon.** Each UDP packet is one (core, spatial stream) and becomes its own
frame (not merged into MIMO frames). Subcarriers keep FFT order (DC, positive,
negative) with signed `index`. The chip is inferred from the header's chip
version bytes (table taken from CSIKit/csiread); override with
`--nexmon-chip` for unknown values. Note: CSIKit and csiread both decode
bcm43455c0 (Raspberry Pi 3B+/4) as plain int16 I/Q, and use the packed-float
layout for bcm4358 and bcm4366c0; this implementation follows them. Only
IPv4/UDP over Ethernet, raw-IP or Linux-cooked pcap link types are handled;
IP fragments are skipped. `noise_floor` is not provided by Nexmon (0).

**Atheros.** Reference implementations differ on (a) antenna loop order for
multi-antenna captures and (b) an optional `+1` after sign conversion
(`--atheros-plus-one`). The defaults (CSIKit's loop order, no `+1`) are
unverified against real captures. `rssi_dbm` is `noise_floor + rssi`
(CLAIMED: the tool's `rssi` is dB above noise).

## Robustness

All length fields are validated against fixed maxima before allocation; no
parser indexes unchecked; truncated or corrupt input yields `FormatError`.
The module tests include fuzz-style cases (random bytes, every truncation
point of every fixture, byte-mutated fixtures, zero streams) that assert no
panic and guaranteed termination.

## Attribution

Intel 5300, Atheros and Nexmon details were cross-checked against
[CSIKit](https://github.com/Gi-z/CSIKit) (MIT) and
[csiread](https://github.com/citysu/csiread) (MIT); each reader's module header
repeats the attribution. [nexmon_csi](https://github.com/seemoo-lab/nexmon_csi)
has no clear licence, so only its documented packet layout was used and the
code is original.
