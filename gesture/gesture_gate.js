// 作答闸门（gesture_gate.py 的 JS 版）：动作激活 + 保持确认 + 每题一次。
//
// 为什么：手部关键点看不到身体，静止垂在体侧的手（食指自然朝下）和「向下指」
// 在几何上几乎一样。必须要求「手先动过」才开始作答，并且每题只接受一次。
//
// 用法：
//   const gate = new AnswerGate();
//   const ev = gate.update({ landmarks, gesture, confidence }, performance.now());
//   if (ev.answer) { ... }

export const GATE_DEFAULTS = {
  activateMotion: 0.02,   // 0.5s 窗口内腕点位移 / 画面短边
  motionWindowMs: 500,
  holdMs: 400,
  minConf: 0.45,
  releaseMs: 600,
  noneReleaseMs: 800,
  cooldownMs: 800,
};

export class AnswerGate {
  constructor(opts = {}) {
    this.o = { ...GATE_DEFAULTS, ...opts };
    this.reset();
  }

  reset() {
    this.history = [];
    this.active = false;
    this.armed = true;
    this.holdStart = null;
    this.holdGesture = 'none';
    this.lastAnswerMs = -1e9;
    this.lastHandMs = null;
    this.noneSince = null;
  }

  /** 每题开始时调用：要求重新做动作激活。 */
  resetQuestion() {
    this.armed = true;
    this.active = false;
    this.holdStart = null;
    this.holdGesture = 'none';
  }

  update(out, nowMs = performance.now()) {
    const now = Number(nowMs);
    const landmarks = out && out.landmarks ? out.landmarks : null;
    const hasHand = !!(landmarks && landmarks.length);
    if (hasHand) {
      const wrist = landmarks[0];
      this.history.push([now, wrist.x, wrist.y]);
      this.lastHandMs = now;
      this.noneSince = null;
    }
    while (this.history.length && now - this.history[0][0] > this.o.motionWindowMs) {
      this.history.shift();
    }

    let motion = 0;
    if (this.history.length >= 2) {
      const [x0, y0] = [this.history[0][1], this.history[0][2]];
      for (const [, x, y] of this.history) {
        motion = Math.max(motion, Math.hypot(x - x0, y - y0));
      }
    }

    if (!hasHand) {
      if (this.lastHandMs !== null && now - this.lastHandMs >= this.o.releaseMs) {
        this.active = false;
        this.armed = true;
        this.holdStart = null;
        this.holdGesture = 'none';
      }
      return this._event(false, out, motion, false);
    }

    const gesture = (out && out.gesture) || 'none';
    const confidence = Number((out && out.confidence) || 0);

    if (gesture === 'none') {
      if (this.noneSince === null) this.noneSince = now;
      else if (now - this.noneSince >= this.o.noneReleaseMs) {
        this.active = false;
        this.armed = true;
        this.holdStart = null;
        this.holdGesture = 'none';
      }
    } else {
      this.noneSince = null;
    }

    if (!this.active && motion >= this.o.activateMotion) this.active = true;

    let answer = false;
    if (this.active && this.armed && gesture !== 'none' && confidence >= this.o.minConf) {
      if (this.holdGesture !== gesture) {
        this.holdGesture = gesture;
        this.holdStart = now;
      } else if (this.holdStart !== null && now - this.holdStart >= this.o.holdMs) {
        if (now - this.lastAnswerMs >= this.o.cooldownMs) {
          answer = true;
          this.lastAnswerMs = now;
          this.armed = false;
          this.active = false;
          this.holdStart = null;
          this.holdGesture = 'none';
        }
      }
    } else {
      this.holdStart = null;
      this.holdGesture = 'none';
    }

    return this._event(answer, out, motion, true);
  }

  _event(answer, out, motion, hasHand) {
    return {
      answer: !!answer,
      gesture: hasHand ? ((out && out.gesture) || 'none') : 'none',
      confidence: hasHand ? Number((out && out.confidence) || 0) : 0,
      active: this.active,
      armed: this.armed,
      motion: Math.round(motion * 10000) / 10000,
    };
  }
}
