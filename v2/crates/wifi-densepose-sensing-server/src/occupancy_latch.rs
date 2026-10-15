//! Room occupancy latch over the peer-link mesh.
//!
//! A person who walks in and sits still stops perturbing the motion detectors
//! within seconds, and the still-body cues (channel-shape wander, baseline
//! shift, breathing) are weaker and intermittent. Judged frame by frame, the
//! room flickers back to "absent". Occupancy sensors solve this with a latch:
//! motion enters OCCUPIED, any later still-body evidence (even from a single
//! link) keeps it, and it releases only after `hold` with no evidence at all.
//!
//! Entering needs motion or a room-level still vote; one link's still cue may
//! only *refresh* an existing occupancy, so a single noisy link cannot create
//! one. The cost is a tail of up to `hold` after the last evidence when a
//! person has left. `hold == 0` disables the latch.

use serde::Serialize;
use std::time::{Duration, Instant};

/// Evidence for one room update.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Evidence {
    /// Room-level motion vote.
    pub motion: bool,
    /// Room-level still-presence vote (enough links agree).
    pub still_room: bool,
    /// Still-body or breathing cue on at least one link.
    pub still_any_link: bool,
}

#[derive(Debug, Clone)]
pub struct OccupancyLatch {
    hold: Duration,
    occupied: bool,
    entered_at: Option<Instant>,
    last_evidence: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LatchStatus {
    pub enabled: bool,
    pub occupied: bool,
    pub hold_s: f64,
    /// Seconds since the latch entered OCCUPIED.
    pub occupied_for_s: Option<f64>,
    /// Seconds since the last motion or still-body evidence.
    pub since_evidence_s: Option<f64>,
}

impl OccupancyLatch {
    pub fn new(hold: Duration) -> Self {
        Self { hold, occupied: false, entered_at: None, last_evidence: None }
    }

    pub fn enabled(&self) -> bool {
        !self.hold.is_zero()
    }

    pub fn occupied(&self) -> bool {
        self.enabled() && self.occupied
    }

    /// Feed one room update; returns whether the room is held occupied.
    pub fn observe(&mut self, e: Evidence, now: Instant) -> bool {
        if !self.enabled() {
            return false;
        }
        let enters = e.motion || e.still_room;
        if enters || (self.occupied && e.still_any_link) {
            self.last_evidence = Some(now);
            if !self.occupied {
                self.occupied = true;
                self.entered_at = Some(now);
            }
        } else if self.occupied
            && self.last_evidence.is_none_or(|t| now.saturating_duration_since(t) >= self.hold)
        {
            self.occupied = false;
            self.entered_at = None;
        }
        self.occupied
    }

    /// Forget all state (e.g. when the empty-room calibration is redone).
    pub fn reset(&mut self) {
        *self = Self::new(self.hold);
    }

    pub fn status(&self, now: Instant) -> LatchStatus {
        let ago = |t: Option<Instant>| t.map(|t| now.saturating_duration_since(t).as_secs_f64());
        LatchStatus {
            enabled: self.enabled(),
            occupied: self.occupied(),
            hold_s: self.hold.as_secs_f64(),
            occupied_for_s: if self.occupied { ago(self.entered_at) } else { None },
            since_evidence_s: ago(self.last_evidence),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOLD: Duration = Duration::from_secs(300);
    const MOTION: Evidence = Evidence { motion: true, still_room: false, still_any_link: false };
    const ONE_LINK: Evidence = Evidence { motion: false, still_room: false, still_any_link: true };
    const NOTHING: Evidence = Evidence { motion: false, still_room: false, still_any_link: false };

    fn at(t0: Instant, s: u64) -> Instant {
        t0 + Duration::from_secs(s)
    }

    #[test]
    fn motion_enters_and_quiet_releases_after_hold() {
        let t0 = Instant::now();
        let mut l = OccupancyLatch::new(HOLD);
        assert!(l.observe(MOTION, t0));
        assert!(l.observe(NOTHING, at(t0, 299)), "held within the hold time");
        assert!(!l.observe(NOTHING, at(t0, 300)), "released at the hold time");
    }

    #[test]
    fn single_link_still_cue_keeps_but_cannot_create_occupancy() {
        let t0 = Instant::now();
        let mut l = OccupancyLatch::new(HOLD);
        assert!(!l.observe(ONE_LINK, t0), "one link must not enter");
        l.observe(MOTION, t0);
        // A person sits still for 20 minutes; one link sees them every 4 min.
        for m in (4..=20).step_by(4) {
            assert!(l.observe(NOTHING, at(t0, m * 60 - 1)));
            assert!(l.observe(ONE_LINK, at(t0, m * 60)));
        }
        assert!(l.status(at(t0, 1200)).occupied_for_s.unwrap() >= 1200.0);
        assert!(!l.observe(NOTHING, at(t0, 1200 + 300)));
    }

    #[test]
    fn room_still_vote_enters_without_motion() {
        // Someone already sitting when the server starts.
        let t0 = Instant::now();
        let mut l = OccupancyLatch::new(HOLD);
        let e = Evidence { still_room: true, ..NOTHING };
        assert!(l.observe(e, t0));
    }

    #[test]
    fn zero_hold_disables() {
        let t0 = Instant::now();
        let mut l = OccupancyLatch::new(Duration::ZERO);
        assert!(!l.observe(MOTION, t0));
        assert!(!l.status(t0).enabled);
    }

    #[test]
    fn reset_clears_occupancy() {
        let t0 = Instant::now();
        let mut l = OccupancyLatch::new(HOLD);
        l.observe(MOTION, t0);
        l.reset();
        assert!(!l.occupied());
        assert_eq!(l.status(t0).since_evidence_s, None);
    }
}
