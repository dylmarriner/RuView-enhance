# ADR-419: Integrate external WiFi-sensing techniques into the peer-link pipeline

- Status: proposed
- Date: 2026-10-15
- Related: ADR-018 (ESP32 frame layout), ADR-050 (OTA PSK), ADR-292 (wideband ingest)
- Promotion: fork branch `phase3-frankenstein` (RuView-enhance); not proposed upstream

## Context

The 4-node ESP32-S3 peer-link mesh (12 directed links, about 6-9 usable)
measured 0/30 empty-room false alarms but detected a still person only ~40%
of the time, and RTI localisation always peaked on a node (MEASURED,
2026-10-07, see CHANGELOG). Several open projects solve parts of this. Their
licences differ, so each technique is either reused with attribution or
reimplemented from its published description.

## Decision

Add five independent pieces, each off or advisory by default:

| Piece | Source and licence | Where | Default |
|---|---|---|---|
| Multi-format CSI ingest (Intel 5300, Atheros, Nexmon pcap, ESP32 CSV) and `csi-convert` | Format documentation; clean-room readers | `wifi-densepose-hardware::formats` | CLI only |
| Phase-coherent AoA (Bartlett/MVDR/MUSIC, SpotFi-style AoA/ToF, multi-array fusion) | Published methods; clean-room | `wifi-densepose-signal::aoa` | Library only. The single-antenna boards cannot use it. |
| Webcam-taught activity k-NN with held-out test sessions | wifi-sensing-kit (MIT), method adapted | `activity.rs`, `/ui/activity.html` | Off until used |
| Still-person features: NBVI tone selection, moving-variance segmentation, baseline shift with an absolute floor, breathing cue | ESPectre (GPL-3.0) public docs; clean-room, no code copied | `presence_features.rs` | `--peer-presence-mode advisory` (reported, not used) |
| Receiver gain lock | `espressif/esp_csi_gain_ctrl` (Apache-2.0) component | firmware `csi_gain_lock.*` | `CONFIG_CSI_GAIN_LOCK=n` |
| Room occupancy latch | Standard occupancy-sensor practice | `occupancy_latch.rs` | `--occupancy-hold-secs 0` (off) |

The occupancy latch enters on room motion or a room-level still vote. A
still cue on a single link only refreshes an occupancy that already exists.
The latch releases after the hold time with no evidence. While held, an
`absent` room is reported as `present_still`; `unavailable` is never
overridden. By default only channel-shape wander may refresh it. The advisory
still and breathing cues refresh it only with `--peer-presence-mode active`,
because they drift into false alarms hours after calibration (SYNTHETIC).

Upstream changes carried in with the ingest branch (2026-10-12 releases) are
kept. For `provision.py`, upstream's validated OTA PSK, which is never
cached in state, replaces the fork's version.

## Consequences

- Nothing changes on the installed boards until an operator opts in: the gain
  lock needs a firmware build with the option enabled, and the presence vote
  and latch need server flags.
- All accuracy figures for the new pieces are SYNTHETIC (simulator in
  `presence_eval.rs`) or unit-level. Hardware evidence needs labelled
  empty/occupied periods on the real mesh, scored with
  `scripts/eval_peer_links.py`.
- The latch trades detection for a release tail. After a person leaves, the
  room stays occupied for up to the hold time.
- RTI localisation is limited by link geometry with about 6 usable links.
  Position fingerprinting through the activity teacher (zone labels instead
  of movement labels) is the documented alternative. It needs no new code.

## Verification

- Firmware host tests, including `test_csi_gain_lock`: all pass.
- Provisioning tests (72): pass after the conflict resolution.
- `cargo test --no-default-features` for the hardware, signal,
  sensing-server, cli and mtk-bridge crates, including the latch unit and
  integration tests.
- Not verified on hardware. No board was flashed with this branch.
