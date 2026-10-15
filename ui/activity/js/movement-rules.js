// Camera-teacher rules: MediaPipe Pose landmarks -> one movement label.
//
// Adapted from wifi-sensing-kit by Andris Gauracs (MIT licence)
// https://github.com/andrisgauracs/wifi-sensing-kit  (kit/web/movement.html)
// The rule set and its thresholds are the kit's; the code is restructured into a
// pure, DOM-free module so it can be unit-tested. The kit validated the rules
// offline on its author's own recording (CLAIMED by that project, not measured here).
//
// Pure computation only: landmarks in, label string out. Nothing here touches
// the network, so video frames and landmarks never leave the browser.

export const MOVES = [
  { label: 'nobody', name: 'Nobody', hint: 'Step out of the camera view' },
  { label: 'standing', name: 'Standing', hint: 'Face the camera, whole body in view' },
  { label: 'walking', name: 'Walking', hint: 'Back and forth across the room' },
  { label: 'squatting', name: 'Squatting', hint: 'Steady squats, whole body in view' },
  { label: 'jumping', name: 'Jumping', hint: 'Jumping jacks or small jumps' },
  { label: 'spinning', name: 'Spinning', hint: 'Turn in place, one turn every 2-3 s' },
  { label: 'sitting', name: 'Sitting', hint: 'On a chair, still in view' },
  { label: 'waving', name: 'Waving', hint: 'Standing, hands above your shoulders' },
];

export const UNSURE = 'unsure';
// MediaPipe Pose landmark indices.
const P = { LS: 11, RS: 12, LW: 15, RW: 16, LH: 23, RH: 24, LK: 25, RK: 26, LA: 27, RA: 28 };
export const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28]];
export const JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
// Samples are taken at ~8/s, as in the kit's validation; the spin rule counts swings.
export const SAMPLE_MS = 120;

const vis = (l, i, th = 0.5) => (l[i].visibility ?? 1) > th;
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const span = (a) => (a.length ? Math.max(...a) - Math.min(...a) : 0);
const std = (a) => {
  if (!a.length) return 0;
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length);
};
const median = (a) => {
  const b = [...a].sort((x, y) => x - y);
  return b.length ? b[b.length >> 1] : 0;
};
function angle(a, b, c) {
  const v1 = [a.x - b.x, a.y - b.y];
  const v2 = [c.x - b.x, c.y - b.y];
  const d = Math.hypot(...v1) * Math.hypot(...v2) || 1;
  return (Math.acos(Math.max(-1, Math.min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / d))) * 180) / Math.PI;
}

export class MovementClassifier {
  constructor() {
    this.hist = []; // recent samples
    this.thighBase = null; // the user's standing thigh length (vertical / torso), learned while standing
    this.lastSample = -Infinity;
  }

  /** Feed one pose result (`lm` = landmark array or null). Returns a label when a new sample was taken, else null. */
  push(lm, now) {
    if (now - this.lastSample < SAMPLE_MS) return null;
    this.lastSample = now;
    if (!lm) {
      this.hist.push({ t: now, present: false, present2: false });
    } else {
      const present = vis(lm, P.LS) && vis(lm, P.RS) && vis(lm, P.LH) && vis(lm, P.RH);
      const present2 = (vis(lm, P.LS, 0.4) || vis(lm, P.RS, 0.4)) && (vis(lm, P.LH, 0.4) || vis(lm, P.RH, 0.4));
      const sh = mid(lm[P.LS], lm[P.RS]);
      const hip = mid(lm[P.LH], lm[P.RH]);
      const torso = Math.hypot(sh.x - hip.x, sh.y - hip.y) || 1e-3;
      // Legs one side at a time: a hip and its knee are enough (ankles are often hidden).
      const legs = [[P.LH, P.LK, P.LA], [P.RH, P.RK, P.RA]].filter(([h, k]) => vis(lm, h, 0.4) && vis(lm, k, 0.25));
      this.hist.push({
        t: now, present, present2, hip, torso,
        knee2: legs.length ? Math.min(...legs.map(([h, k, a]) => angle(lm[h], lm[k], lm[a]))) : null,
        thigh: legs.length ? Math.min(...legs.map(([h, k]) => Math.abs(lm[k].y - lm[h].y))) / torso : null,
        shW: Math.abs(lm[P.LS].x - lm[P.RS].x) / torso,
        hipPx: Math.abs(lm[P.LH].x - lm[P.RH].x) + 1e-4,
        wristsUp: (vis(lm, P.LW) && lm[P.LW].y < lm[P.LS].y - 0.1 * torso)
          || (vis(lm, P.RW) && lm[P.RW].y < lm[P.RS].y - 0.1 * torso),
      });
    }
    while (this.hist.length && now - this.hist[0].t > 3000) this.hist.shift();
    return this.classify(now);
  }

  classify(now) {
    const H = this.hist.filter((h) => now - h.t < 1560);
    const HL = this.hist.filter((h) => now - h.t < 2560);
    if (H.length < 8) return UNSURE;
    const seen = H.filter((h) => h.present);
    const P2 = H.filter((h) => h.present2);
    if (seen.length < H.length * 0.3) return 'nobody';
    if (seen.length < H.length * 0.7) return UNSURE; // walking in or out of view
    if (P2.length < H.length * 0.6) return UNSURE;
    const torso = median(P2.map((h) => h.torso));
    const xs = P2.map((h) => h.hip.x);
    const ys = P2.map((h) => h.hip.y);
    const dt = (P2[P2.length - 1].t - P2[0].t) / 1000 || 1.5;
    const hSpeed = Math.abs(xs[xs.length - 1] - xs[0]) / torso / dt + (span(xs) / torso) * 0.5;
    const scaleChange = span(P2.map((h) => h.torso)) / torso;
    const vStd = std(ys) / torso;
    const vSpan = span(ys) / torso;
    const th = P2.map((h) => h.thigh).filter((v) => v != null);
    const k2 = P2.map((h) => h.knee2).filter((v) => v != null);
    const comp = th.length && this.thighBase ? Math.min(...th) / this.thighBase : null; // 1 = standing
    const armsUp = P2.filter((h) => h.wristsUp).length / P2.length;
    const L2 = HL.filter((h) => h.present2);
    const sw = L2.map((h) => h.shW);
    const swMed = median(sw);
    let crossings = 0;
    for (let i = 1; i < sw.length; i++) if (Math.sign(sw[i] - swMed) !== Math.sign(sw[i - 1] - swMed)) crossings++;
    const stillPlace = L2.length > 6 && span(L2.map((h) => h.hip.x)) / torso < 1.0
      && span(L2.map((h) => h.torso)) / median(L2.map((h) => h.torso)) < 0.25;
    const hp = P2.map((h) => h.hipPx);
    const approach = span(hp) / median(hp) > 0.3;
    const legsStraight = comp == null || comp > 0.8;
    let lab = UNSURE;
    if (hSpeed > 0.9 || (approach && scaleChange > 0.2) || (scaleChange > 0.35 && legsStraight)) lab = 'walking';
    else if (sw.length > 6 && span(sw) > 0.3 && crossings >= 3 && stillPlace) lab = 'spinning';
    else if (comp != null && comp < 0.75 && vSpan > 0.12 && !approach) lab = 'squatting';
    else if (comp != null && comp < 0.65 && vSpan <= 0.12) lab = 'sitting';
    else if (k2.length && Math.min(...k2) < 110 && vStd < 0.05) lab = 'sitting';
    else if (vSpan > 0.2) lab = comp != null && comp < 0.85 ? 'squatting' : 'jumping';
    else if (vStd > 0.08) lab = 'jumping';
    else if (armsUp > 0.4) lab = 'waving';
    else if (vStd < 0.05 && k2.length >= P2.length * 0.5) lab = 'standing';
    if (lab === 'standing' && th.length) {
      const m = median(th);
      this.thighBase = this.thighBase == null ? m : 0.95 * this.thighBase + 0.05 * m;
    }
    return lab;
  }
}
