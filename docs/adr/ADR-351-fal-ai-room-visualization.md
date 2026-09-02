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

**Branded / RuView tech-overlay** — image-to-image against RuView's own
marketing hero graphic (`assets/ruview-small-gemini.jpg`), not a pure
text-prompt style like the other four. **This style deliberately depicts a
literal human silhouette with a glowing pose-skeleton overlay** (joint
markers, WiFi signal arcs, floating vital-sign readout panels), matching the
reference image's own sci-fi data-overlay aesthetic. This is a materially
weaker privacy posture than `abstract` and should be understood as such by
any caller: the figure is fal.ai's own synthetic generation conditioned on a
real product-marketing image, not a reproduction of any real person's
likeness, but it is the only preset among the five that renders anything
resembling a human form. Callers who want to avoid human-figure depiction
should use `abstract` instead. Example (occupied, moving, single occupant):
*"a room with one glowing cyan pose-skeleton human figure mid-stride in
motion, joint markers with motion trails, WiFi signal arcs radiating from a
router, floating vital-sign readout panels, status overlay reading OCCUPIED
- ACTIVE, dark navy-blue background, glowing cyan and white sci-fi data
overlay aesthetic, futuristic WiFi sensing visualization, technical HUD
readout style, high contrast glow, RuView WiFi-DensePose branding
aesthetic."* Implementation: the reference image is base64-encoded and
passed directly as a `data:image/jpeg;base64,...` URI in the `image_url`
field of `POST https://fal.run/fal-ai/fast-sdxl/image-to-image` (confirmed
this session: no separate fal.ai storage-upload call is required, a data URI
is accepted directly), with `strength` (default 0.55) controlling how much
the output favors the reference image's composition versus the per-frame
text prompt.

Full per-activity prompt text for all five styles is in
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
  cannot render a human likeness regardless of prompt drift. Conversely,
  **`branded` is the weakest privacy posture of the five presets** and
  should only be used for deliberate, explicitly-authorized branding/
  demonstration purposes, exactly as this session's use was.

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
- **Branded style (added in a follow-up session)**: real image-to-image
  connectivity confirmed via a minimal test call to
  `fast-sdxl/image-to-image` with a base64 data-URI `image_url` (HTTP 200,
  ~0.48s inference) before any code was written, establishing that no
  separate fal.ai storage-upload step is needed. A real 8-frame capture
  against the live household feed produced a real, `ffprobe`-verified H.264
  MP4 (1024x1024, 8 frames, 8s, non-corrupt). All 8 frames are byte-
  identical (same MD5) -- checked honestly, not assumed a bug: the raw
  sensing feature values (breathing/motion band power, per-node RSSI)
  genuinely varied between live polls during the capture window (confirmed
  via direct polls of the live endpoint), but the coarse `Activity` bucket
  that `build_prompt` actually consumes stayed at `OccupiedMoving` (single
  occupant) for the entire ~16s window, and a fixed seed + fixed prompt +
  fixed reference image + fixed strength legitimately produces deterministic
  output from `fast-sdxl`. This is expected model behavior given a real,
  genuinely-unchanging coarse state, not fabricated or cached data. Real
  total spend for this style's validation + capture: 9 real API calls (1
  connectivity test + 8 capture frames) at fal.ai's `fast-sdxl`-class
  per-image pricing (order of $0.02-0.03/image per public pricing pages),
  well under the $20 budget set for this work.
