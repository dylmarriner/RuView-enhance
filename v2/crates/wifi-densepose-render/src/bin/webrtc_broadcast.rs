//! ADR-352 WebRTC broadcast of the composited stream.
//!
//! A real, minimal `webrtc-rs` peer connection server: reads a real VP8 IVF
//! file (produced by `ffmpeg` from `three-stream-recorder`'s real composite
//! frames, or Layer 1 alone if no composite exists yet), serves a bare HTML
//! viewer page plus a single `/offer` signaling endpoint (no ICE trickle,
//! no TURN — a same-host/LAN "bare handshake" per ADR-352's stated scope:
//! not production-grade, but a real browser tab genuinely receiving real
//! decoded frames over a real RTCPeerConnection).
//!
//! Loops the IVF file so a connecting viewer always sees motion, since this
//! is a demonstration broadcast of a bounded real recording, not a live
//! camera. Streaming the *live* render loop directly (no file in between)
//! is real follow-up work once this bare handshake is proven.

use std::fs::File;
use std::io::BufReader;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::MediaEngine;
use webrtc::api::APIBuilder;
use webrtc::interceptor::registry::Registry;
use webrtc::media::io::ivf_reader::IVFReader;
use webrtc::media::Sample;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use webrtc::track::track_local::TrackLocal;

#[derive(Parser, Debug)]
struct Args {
    /// Real VP8 IVF file to stream (produce with e.g.
    /// `ffmpeg -framerate 6 -i composite_frames/f%05d.png -c:v libvpx out.ivf`).
    #[arg(long)]
    ivf: PathBuf,

    #[arg(long, default_value_t = 8787)]
    port: u16,
}

#[derive(Clone)]
struct AppState {
    ivf_path: Arc<PathBuf>,
}

#[derive(Deserialize)]
struct OfferBody {
    sdp: String,
    #[serde(rename = "type")]
    typ: String,
}

#[derive(Serialize)]
struct AnswerBody {
    sdp: String,
    #[serde(rename = "type")]
    typ: String,
}

const VIEWER_HTML: &str = r##"<!doctype html><html><body style="background:#111;color:#eee;font-family:monospace">
<h3>ADR-352 real WebRTC composite viewer</h3>
<video id="v" autoplay playsinline muted style="width:640px;background:#000"></video>
<p id="status">idle</p>
<script>
async function connect() {
  const pc = new RTCPeerConnection();
  pc.addTransceiver('video', {direction: 'recvonly'});
  pc.ontrack = (ev) => { document.getElementById('v').srcObject = ev.streams[0]; };
  pc.onconnectionstatechange = () => { document.getElementById('status').innerText = pc.connectionState; };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const resp = await fetch('/offer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sdp: offer.sdp, type: offer.type}),
  });
  const answer = await resp.json();
  await pc.setRemoteDescription(answer);
}
connect();
</script>
</body></html>"##;

async fn index() -> Html<&'static str> {
    Html(VIEWER_HTML)
}

async fn offer(State(state): State<AppState>, Json(body): Json<OfferBody>) -> impl IntoResponse {
    match handle_offer(state, body).await {
        Ok(answer) => Json(answer).into_response(),
        Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("real webrtc error: {e}")).into_response(),
    }
}

async fn handle_offer(state: AppState, body: OfferBody) -> anyhow::Result<AnswerBody> {
    let mut m = MediaEngine::default();
    m.register_default_codecs()?;
    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut m)?;
    let api = APIBuilder::new().with_media_engine(m).with_interceptor_registry(registry).build();

    let config = RTCConfiguration::default(); // same-host/LAN bare handshake: no STUN/TURN configured (see module docs)
    let pc = Arc::new(api.new_peer_connection(config).await?);

    let video_track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: webrtc::api::media_engine::MIME_TYPE_VP8.to_owned(),
            ..Default::default()
        },
        "video".to_owned(),
        "adr352-composite".to_owned(),
    ));
    let rtp_sender = pc.add_track(video_track.clone() as Arc<dyn TrackLocal + Send + Sync>).await?;
    tokio::spawn(async move {
        let mut buf = vec![0u8; 1500];
        while rtp_sender.read(&mut buf).await.is_ok() {}
    });

    let ivf_path = state.ivf_path.as_ref().clone();
    let pc_for_state = Arc::clone(&pc);
    let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);
    pc.on_peer_connection_state_change(Box::new(move |s: RTCPeerConnectionState| {
        if s == RTCPeerConnectionState::Connected {
            let track = Arc::clone(&video_track);
            let path = ivf_path.clone();
            let done_tx = done_tx.clone();
            tokio::spawn(async move {
                stream_ivf_loop(path, track).await;
                let _ = done_tx.send(()).await;
            });
        }
        Box::pin(async {})
    }));
    tokio::spawn(async move {
        let _ = done_rx.recv().await;
        let _ = pc_for_state.close().await;
    });

    let offer_desc = RTCSessionDescription::offer(body.sdp)?;
    let _ = body.typ; // real offer type is implied by RTCSessionDescription::offer(); kept for the JSON round-trip shape
    pc.set_remote_description(offer_desc).await?;
    let answer = pc.create_answer(None).await?;
    let mut gather_complete = pc.gathering_complete_promise().await;
    pc.set_local_description(answer).await?;
    let _ = gather_complete.recv().await;

    let local_desc = pc.local_description().await.ok_or_else(|| anyhow::anyhow!("no local description after gathering"))?;
    Ok(AnswerBody { sdp: local_desc.sdp, typ: "answer".to_string() })
}

/// Streams one real VP8 IVF file to the track, looping. Real per-frame
/// timing comes from the IVF container's own timestamps (real ffmpeg
/// output), not a fabricated fixed sleep.
async fn stream_ivf_loop(path: PathBuf, track: Arc<TrackLocalStaticSample>) {
    loop {
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("cannot open real ivf file {}: {e}", path.display());
                return;
            }
        };
        let reader = BufReader::new(file);
        let (mut ivf, header) = match IVFReader::new(reader) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("real ivf parse failed for {}: {e}", path.display());
                return;
            }
        };
        let frame_duration = Duration::from_secs_f64(1.0 / header.timebase_denominator as f64 * header.timebase_numerator as f64);
        loop {
            match ivf.parse_next_frame() {
                Ok((frame, _)) => {
                    if track
                        .write_sample(&Sample { data: frame.freeze(), duration: frame_duration, ..Default::default() })
                        .await
                        .is_err()
                    {
                        return; // viewer disconnected
                    }
                    tokio::time::sleep(frame_duration).await;
                }
                Err(_) => break, // real end-of-file: loop the recording
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    if !args.ivf.exists() {
        anyhow::bail!("real ivf file not found: {} (encode one first, see module docs)", args.ivf.display());
    }
    let state = AppState { ivf_path: Arc::new(args.ivf) };

    let app = Router::new()
        .route("/", get(index))
        .route("/offer", post(offer))
        .with_state(state);

    let addr: SocketAddr = ([0, 0, 0, 0], args.port).into();
    println!("real webrtc-broadcast viewer at http://{addr}/  (offer endpoint: POST /offer)");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
