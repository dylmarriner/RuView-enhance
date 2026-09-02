# ADR 351: Near-real-time generative room visualization via fal.ai

## Status

Accepted. Implemented as `wifi-densepose-room-viz` (`v2/crates/wifi-densepose-room-viz`).
Real, live-data end-to-end validation performed: two real captures (architectural
and abstract style presets) against the live household sensing feed on
`ruv-mac-mini`, real generative calls to fal.ai `fast-sdxl`, real assembled
video files. Cinematic and minimal presets are designed and prompt-templated
below but were not captured to real video in this validation pass (cost/time
tradeoff — noted honestly, not run).

## Context

RuView's default posture, applied consistently everywhere else in this
project, is that real sensor data stays local — never sent to a third-party
hosted service. This ADR documents a **deliberate, explicit exception** to
that default, made by direct user decision, not a change to the project's
default policy.

The user asked whether fal.ai (a third-party generative-inference hosting
platform) could be combined with live RuView sensing data to produce a
near-real-time visualization of a room. Two real constraints were confirmed
before building anything:

1. fal.ai's product is generative image/video model hosting, optimized for
   low latency but not literally real-time frame-by-frame rendering. The
   `fast-sdxl` model (<https://fal.ai/models/fal-ai/fast-sdxl>) generates a
   single image in well under one second (measured this session: 0.485s for
   a real test call), which supports a "near-real-time" cadence of one
   generated frame every few seconds — not video-framerate generation.
2. Sending real household presence/motion data to fal.ai is a real,
   consequential change from this project's normal stance. The user
   explicitly authorized it for this specific use case, on the same
   already-established informal-consent basis as this session's other real
   RuView data work on this household (`cohen`'s home, RuView collaborator).

**What actually crosses the local/third-party boundary**: only the already
-classified, coarse scene state returned by the sensing server's
`/api/v1/sensing/latest` endpoint — `classification.presence` (bool),
`classification.motion_level` (string), `estimated_persons` (integer) — is
read and mapped into a short English text prompt. Raw CSI, skeleton
keypoints, per-node RSSI/signal-field data, and vital-signs values returned
by the same endpoint are read locally but are **not** included in anything
sent to fal.ai.

## Decision

### Data source and transport

The live sensing server on `ruv-mac-mini` is reachable only via its own
loopback (`http://127.0.0.1:3000`), not directly over Tailscale from other
hosts (confirmed this session). `wifi-densepose-room-viz` polls it by
shelling out to `ssh <host> curl ... 127.0.0.1:3000/api/v1/sensing/latest`
once per capture interval (default 2s), matching the SSH-loopback access
pattern already used throughout this project for that host. A native
WebSocket stream is advertised on the sensing server's own root page
(`ws://localhost:8765/ws/sensing`) but was found not to be actually
listening when tested this session — REST polling is the real, working
path, not a fallback of convenience.

### Sensing-to-scene mapping (shared across every style)

A single, auditable mapping function (`map_scene_state`) converts the raw
classification into a coarse `SceneState { activity, multiple_occupants }`:

- `activity = Vacant` iff `presence == false`.
- `activity = OccupiedMoving` iff `presence == true` and `motion_level`
  contains the substring `"moving"`.
- `activity = OccupiedStill` iff `presence == true` and `motion_level` does
  not contain `"moving"`.
- `multiple_occupants = estimated_persons >= 2`.

Every style preset's prompt builder consumes this same `SceneState` — only
the stylistic language differs between presets, never the underlying
classification logic. This keeps "what the visualization is honestly
claiming about the room" auditable in one place, independent of which style
a caller picks.

### Style presets

Four genuinely distinct, hand-written prompt templates, selectable via
`--style` (not a shared base prompt with a style-name suffix):

**Architectural / editorial** — literal room depiction, high-end interior-
design-magazine aesthetic. Example (vacant): *"professional interior design
magazine photograph of a minimalist living room, perfectly styled and
unoccupied, soft diffused daylight through sheer curtains, calm and still
atmosphere, architectural digest style, considered symmetrical composition,
neutral sophisticated color palette of warm greige and soft white, natural
light study, shot on medium format camera, ultra sharp focus, 8k editorial
quality."*

**Cinematic / moody** — literal room depiction, dramatic film-still
aesthetic; occupied states read naturally warmer/more dramatic than empty
ones. Example (occupied, moving): *"cinematic film still of a living room
with a figure caught mid-movement, motion blur trailing, dramatic side
lighting, high contrast, sense of restless energy, shot on anamorphic lens,
shallow depth of field, dramatic chiaroscuro lighting, rich teal and amber
color grade, 35mm film grain, A24 movie aesthetic, moody atmospheric
interior."*

**Abstract / data-art** — deliberately **never depicts a literal human
figure**. This is a real, meaningful privacy property of this preset, not
only an aesthetic choice: an occupied room becomes a glowing form of light,
not a rendered person. Example (occupied, still): *"abstract generative art
visualization of a stable occupied space, one warm glowing orb of soft amber
light, gently pulsing, centered and steady, soft volumetric light, glowing
particle field, dark charcoal background, ambient occlusion, high-end data
visualization aesthetic, generative light installation style, no human
figures, pure light and form."*

**Minimal / technical** — literal room depiction abstracted into an
occupancy-status diagram, not a photoreal scene; reads as a monitoring
dashboard rather than generated art. Example (vacant): *"minimalist
technical line-art rendering of an empty room floor plan, all zones shown
inactive, cool grey tones, status: vacant, monochrome architectural
wireframe, clean vector linework, muted slate and white palette, dashboard
UI aesthetic, isometric technical diagram style, precise and clinical."*

Full per-activity prompt text for all four styles is in
`v2/crates/wifi-densepose-room-viz/src/main.rs::build_prompt`.

### Generation and assembly

Each frame: build the styled prompt from the current real `SceneState` →
`POST https://fal.run/fal-ai/fast-sdxl` with `Authorization: Key $FAL_KEY`
(confirmed this session: `Key`, not `Bearer` — fal.ai's own scheme) → save
the returned image → repeat at the configured interval → assemble the saved
frames into an H.264 MP4 via `ffmpeg` at capture end. A fixed `--seed` is
required per capture session (not sourced from wall-clock time) for visual
consistency across a session's frames and for reproducibility.

### Credential handling

`FAL_KEY` is read from the process environment only, fetched by the caller
via `gcloud secrets versions access latest --secret=MINIMAX_MUSIC_FAL_KEY
--project=cognitum-20260110` at the point of use. The binary never writes
the key to disk, never logs it, and it is never committed. (This is the only
fal.ai credential in the project's GCP Secret Manager; it was provisioned
for an earlier, unrelated MiniMax music-generation use and is reused here as
a general fal.ai account key — a real, deliberate, user-confirmed decision,
not an assumption.)

## Privacy and data-handling notes

- Real biometric-adjacent data (presence, motion classification) about a
  real, currently-occupied household leaves the local network boundary for
  the first time in this project, by explicit user authorization scoped to
  this feature.
- Only coarse classification fields cross the boundary — never raw CSI,
  skeleton/pose keypoints, per-node RF features, or vital-signs values, even
  though the same sensing endpoint returns all of them.
- The generated output is a **depiction**, not the raw feed — but it is
  still derived from real presence data about a real occupied home, and
  should be handled with the same care as any other artifact revealing
  "someone was home and moving at this time." Generated videos are treated
  as build artifacts (gitignored), not committed to the repository.
- The abstract/data-art preset is recommended as the privacy-preferred
  default for any future non-demonstration use, since it structurally
  cannot render a human likeness regardless of prompt drift.

## Consequences

- A new, isolated crate (`wifi-densepose-room-viz`) with no callers from the
  rest of the workspace — it does not change any existing service's default
  behavior or data-handling posture.
- Real fal.ai spend occurs on every run; the CLI's `--frames`/`--interval-secs`
  keep a single capture session small and bounded by design (no unbounded
  loop).
- This ADR's exception is scoped to this one feature. It does not authorize
  sending any other RuView data category to fal.ai or any other third-party
  service without its own explicit decision and ADR.

## Real validation performed this session

- Live fal.ai connectivity: real `POST` to `fast-sdxl`, HTTP 200, real image
  returned in 0.485s.
- Live sensing connectivity: real poll of `ruv-mac-mini`'s
  `/api/v1/sensing/latest` via SSH loopback from `ruvultra`, confirmed
  returning genuinely live-changing data (a capture session's scene state
  transitioned from `OccupiedStill` to `OccupiedMoving` mid-capture, proving
  it is not a cached/static response).
- Two real 8-frame captures against the live feed, one per style
  (architectural, abstract), each assembled into a real, `ffprobe`-verified
  H.264 MP4 (1024x576, 8 frames, ~8s, non-corrupt).
- `cargo check`/`cargo clippy --all-targets -- -D warnings` both clean for
  the new crate.
