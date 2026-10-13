//! HTTP surface for webcam-supervised activity recognition (see `activity.rs`).
//!
//! Adapted from the approach of wifi-sensing-kit (MIT) —
//! https://github.com/andrisgauracs/wifi-sensing-kit
//!
//! Every route lives under `/api/v1/activity/`, so the opt-in bearer-token
//! gate on `/api/v1/*` covers it with no extra wiring. Only label strings
//! arrive from the browser; video never reaches the server.
//!
//! - `GET    /api/v1/activity/status`          links, sessions, recording, model
//! - `POST   /api/v1/activity/session/start`   `{ "kind": "train" | "test" }`
//! - `POST   /api/v1/activity/session/stop`
//! - `DELETE /api/v1/activity/sessions/{id}`
//! - `POST   /api/v1/activity/label`           `{ "label": "walking" }`
//! - `POST   /api/v1/activity/train`           `{ "k": 7 }` (train sessions only)
//! - `POST   /api/v1/activity/evaluate`        `{ "k": 7 }` (test sessions only)
//! - `GET    /api/v1/activity/live`            smoothed WiFi-only prediction

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use serde::Deserialize;
use serde_json::json;

use crate::activity::{self, ActivityError, SessionKind};
use crate::path_safety::safe_id;
use crate::training_api::AppState;
use wifi_densepose_sensing_server::error_response;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/activity/status", get(status))
        .route("/api/v1/activity/session/start", post(session_start))
        .route("/api/v1/activity/session/stop", post(session_stop))
        .route("/api/v1/activity/sessions/:id", delete(session_delete))
        .route("/api/v1/activity/label", post(post_label))
        .route("/api/v1/activity/train", post(train))
        .route("/api/v1/activity/evaluate", post(evaluate))
        .route("/api/v1/activity/live", get(live))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Map an [`ActivityError`] to a response. I/O detail is logged, never sent.
fn error(e: ActivityError) -> Response {
    let (code, msg) = match e {
        ActivityError::Invalid(m) => (StatusCode::BAD_REQUEST, m),
        ActivityError::Conflict(m) => (StatusCode::CONFLICT, m),
        ActivityError::NotFound(m) => (StatusCode::NOT_FOUND, m),
        ActivityError::Io(d) => return error_response::internal_error("activity storage", d).into_response(),
    };
    (code, Json(json!({ "success": false, "error": msg, "message": msg }))).into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StartBody {
    kind: SessionKind,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LabelBody {
    label: String,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct KBody {
    k: Option<usize>,
}

async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    let a = &s.activity;
    let model = a.model().map(|m| {
        json!({
            "k": m.k, "labels": m.labels, "link_ids": m.link_ids, "trained_unix": m.trained_unix,
            "train_sessions": m.train_sessions, "n_train_windows": m.n_train_windows,
        })
    });
    Json(json!({
        "success": true,
        "links": a.live_link_summary(std::time::Instant::now()),
        "min_link_rate_hz": activity::MIN_LINK_RATE_HZ,
        "recording": a.recording(),
        "sessions": a.sessions().iter().map(|s| s.summary()).collect::<Vec<_>>(),
        "model": model,
        "limits": {
            "max_samples_per_session": activity::MAX_SAMPLES_PER_SESSION,
            "max_sessions": activity::MAX_SESSIONS,
            "max_k": activity::MAX_K,
        },
    }))
}

async fn session_start(State(state): State<AppState>, Json(body): Json<StartBody>) -> Response {
    let mut s = state.write().await;
    match s.activity.start_session(body.kind, std::time::Instant::now(), unix_now()) {
        Ok(p) => Json(json!({ "success": true, "session": p })).into_response(),
        Err(e) => error(e),
    }
}

async fn session_stop(State(state): State<AppState>) -> Response {
    let mut s = state.write().await;
    match s.activity.stop_session() {
        Ok(p) => Json(json!({ "success": true, "session": p })).into_response(),
        Err(e) => error(e),
    }
}

async fn session_delete(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    if let Err(e) = safe_id(&id) {
        return error(ActivityError::Invalid(format!("invalid session id: {e}")));
    }
    let mut s = state.write().await;
    match s.activity.delete_session(&id) {
        Ok(()) => Json(json!({ "success": true })).into_response(),
        Err(e) => error(e),
    }
}

/// The browser posts the camera's current movement label a few times a second.
async fn post_label(State(state): State<AppState>, Json(body): Json<LabelBody>) -> Response {
    // "unsure" is how the browser says "not certain": accepted, never stored.
    if body.label == "unsure" {
        return Json(json!({ "success": true, "stored": false })).into_response();
    }
    let mut s = state.write().await;
    match s.activity.add_label(&body.label, std::time::Instant::now()) {
        Ok(()) => Json(json!({ "success": true, "stored": true })).into_response(),
        Err(e) => error(e),
    }
}

fn k_of(body: Option<Json<KBody>>) -> usize {
    body.and_then(|Json(b)| b.k).unwrap_or(activity::DEFAULT_K)
}

async fn train(State(state): State<AppState>, body: Option<Json<KBody>>) -> Response {
    let k = k_of(body);
    let sessions = state.read().await.activity.sessions().to_vec();
    let unix = unix_now();
    let model = match tokio::task::spawn_blocking(move || activity::train_model(&sessions, k, unix)).await {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => return error(e),
        Err(e) => return error_response::internal_error("activity train", e).into_response(),
    };
    let summary = json!({
        "k": model.k, "labels": model.labels, "link_ids": model.link_ids,
        "train_sessions": model.train_sessions, "n_train_windows": model.n_train_windows,
    });
    let mut s = state.write().await;
    match s.activity.set_model(model) {
        Ok(()) => Json(json!({ "success": true, "model": summary })).into_response(),
        Err(e) => error(e),
    }
}

async fn evaluate(State(state): State<AppState>, body: Option<Json<KBody>>) -> Response {
    let k = k_of(body);
    let sessions = state.read().await.activity.sessions().to_vec();
    match tokio::task::spawn_blocking(move || activity::evaluate(&sessions, k)).await {
        Ok(Ok(ev)) => Json(json!({ "success": true, "evaluation": ev })).into_response(),
        Ok(Err(e)) => error(e),
        Err(e) => error_response::internal_error("activity evaluate", e).into_response(),
    }
}

async fn live(State(state): State<AppState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    Json(json!({ "success": true, "prediction": s.activity.live() }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use tower::ServiceExt;

    fn app() -> (Router, AppState, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let mut inner = crate::AppStateInner::minimal();
        inner.activity = activity::ActivityEngine::new(dir.path().to_path_buf());
        let state: AppState = Arc::new(tokio::sync::RwLock::new(inner));
        (routes().with_state(state.clone()), state, dir)
    }

    async fn call(app: &Router, method: &str, uri: &str, body: &str) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 22).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
    }

    #[tokio::test]
    async fn rejects_bad_input() {
        let (app, _s, _d) = app();
        assert_eq!(call(&app, "POST", "/api/v1/activity/label", r#"{"label":"Bad Label"}"#).await.0, StatusCode::BAD_REQUEST);
        assert_eq!(call(&app, "POST", "/api/v1/activity/label", r#"{"label":"x","extra":1}"#).await.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(call(&app, "POST", "/api/v1/activity/session/start", r#"{"kind":"validation"}"#).await.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(call(&app, "DELETE", "/api/v1/activity/sessions/.hidden", "").await.0, StatusCode::BAD_REQUEST);
        assert_eq!(call(&app, "DELETE", "/api/v1/activity/sessions/nope", "").await.0, StatusCode::NOT_FOUND);
        assert_eq!(call(&app, "POST", "/api/v1/activity/train", r#"{"k":0}"#).await.0, StatusCode::BAD_REQUEST);
        assert_eq!(call(&app, "POST", "/api/v1/activity/train", r#"{"k":9999}"#).await.0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unsure_is_accepted_but_not_stored() {
        let (app, _s, _d) = app();
        let (c, v) = call(&app, "POST", "/api/v1/activity/label", r#"{"label":"unsure"}"#).await;
        assert_eq!(c, StatusCode::OK);
        assert_eq!(v["stored"], false);
        let (_, v) = call(&app, "POST", "/api/v1/activity/label", r#"{"label":"walking"}"#).await;
        assert_eq!(v["stored"], true);
    }

    #[tokio::test]
    async fn lifecycle_without_links_is_a_clean_conflict() {
        let (app, _s, _d) = app();
        assert_eq!(call(&app, "POST", "/api/v1/activity/session/start", r#"{"kind":"train"}"#).await.0, StatusCode::CONFLICT);
        assert_eq!(call(&app, "POST", "/api/v1/activity/session/stop", "").await.0, StatusCode::CONFLICT);
        assert_eq!(call(&app, "POST", "/api/v1/activity/evaluate", "").await.0, StatusCode::CONFLICT);
        let (c, v) = call(&app, "GET", "/api/v1/activity/live", "").await;
        assert_eq!(c, StatusCode::OK);
        assert_eq!(v["prediction"]["status"], "no_model");
        let (c, v) = call(&app, "GET", "/api/v1/activity/status", "").await;
        assert_eq!(c, StatusCode::OK);
        assert!(v["sessions"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn end_to_end_record_train_evaluate_with_synthetic_frames() {
        let (app, state, _d) = app();
        // SYNTHETIC frames: two classes with different spectral shapes.
        let frame = |class: usize, i: usize| -> Vec<f64> {
            (0..activity::N_TONES)
                .map(|k| 10.0 + 4.0 * ((k as f64 * 0.3) * (class as f64 + 1.0)).sin() + 0.2 * (((i * 31 + k * 7) % 11) as f64 / 11.0))
                .collect()
        };
        let feed = |class: usize, label: &'static str, base: f64, n: usize| {
            let state = state.clone();
            async move {
                for i in 0..n {
                    let t = base + i as f64 * 0.1;
                    let mut s = state.write().await;
                    let _ = s.activity.add_label_at(label, t);
                    s.activity.observe_at("R>1", &frame(class, i), t);
                }
            }
        };
        for (kind, base) in [("train", 100.0), ("test", 400.0)] {
            // Warm the link (>= 5 frames/s) so the session can pick it up.
            feed(0, "sitting", base - 3.0, 30).await;
            {
                let mut s = state.write().await;
                s.activity.start_session_at(if kind == "train" { SessionKind::Train } else { SessionKind::Test }, base, 1).unwrap();
            }
            feed(0, "sitting", base, 120).await;
            feed(1, "walking", base + 12.0, 120).await;
            let mut s = state.write().await;
            s.activity.stop_session().unwrap();
        }
        let (c, v) = call(&app, "POST", "/api/v1/activity/train", "").await;
        assert_eq!(c, StatusCode::OK, "{v}");
        let (c, v) = call(&app, "POST", "/api/v1/activity/evaluate", r#"{"k":5}"#).await;
        assert_eq!(c, StatusCode::OK, "{v}");
        let ev = &v["evaluation"];
        assert_eq!(ev["evidence_tag"], "MEASURED");
        assert!(ev["n_test"].as_u64().unwrap() > 0);
        assert!((ev["chance_baseline"].as_f64().unwrap() - 0.5).abs() < 1e-9);
        // SYNTHETIC data, trivially separable.
        assert!(ev["accuracy"].as_f64().unwrap() > 0.9, "{ev}");
    }
}
