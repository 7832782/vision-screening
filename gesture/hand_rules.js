// 食指指向 4 方向判定（hand_engine.py 的 JS 移植，浏览器端用）。
// 输入：MediaPipe HandLandmarker 的 21 个 normalized landmarks（[{x,y,z}]），以及图像宽高。
// 输出：{ gesture, confidence, debug }；与 Python 版 hand_engine.classify_hand_landmarks 对齐。
//
// 坐标：x/y 归一化到 [0,1]（画面左上角原点，y 向下）。mirror=true 时先 x -> 1-x，
// 与自拍镜像预览一致（前端 CSS scaleX(-1)）。

export const GESTURES = ["up", "down", "left", "right", "none"];

export const CONST = {
  MIN_HAND_SCALE_RATIO: 0.025, // 手部尺度 / min(宽,高)；分辨率无关
  MIN_INDEX_EXT: 0.60,
  MIN_INDEX_DOMINANCE: 0.04,
  MIN_FINGER_SPAN: 0.25,       // 食指 2D 长度 / 手掌尺度；朝镜头戳时塌缩
  MIN_AXIS_RATIO: 0.78,
  AXIS_CONF_FLOOR: 0.70,
};

const WRIST = 0;
const THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8;
const MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12;
const RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16;
const PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20;

const CHAINS = {
  thumb: [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
  index: [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
  middle: [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
  ring: [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
  pinky: [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
};

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const dist2d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function fingerExtension(pts, [mcp, pip, dip, tip]) {
  const straight = dist2d(pts[tip], pts[mcp]);
  const chain = dist2d(pts[pip], pts[mcp]) + dist2d(pts[dip], pts[pip]) + dist2d(pts[tip], pts[dip]);
  if (chain < 1e-6) return 0;
  return straight / chain;
}

export function classifyHand(landmarks, width, height, mirror = true) {
  const debug = {
    hand: null, dx: 0, dy: 0, angle: 0, hand_px: 0, hand_ratio: 0, finger_span: 0, ext: {},
  };
  if (!landmarks || landmarks.length < 21 || width <= 0 || height <= 0) {
    return { gesture: "none", confidence: 0, debug };
  }
  const pts = landmarks.slice(0, 21).map((p) => {
    const x = mirror ? 1 - p.x : p.x;
    return [x * width, p.y * height, p.z || 0];
  });
  if (!pts.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))) {
    return { gesture: "none", confidence: 0, debug };
  }

  const handPx = dist2d(pts[MIDDLE_MCP], pts[WRIST]);
  const handRatio = handPx / Math.min(width, height);
  debug.hand_px = handPx;
  debug.hand_ratio = handRatio;
  if (handRatio < CONST.MIN_HAND_SCALE_RATIO) return { gesture: "none", confidence: 0, debug };

  const ext = {};
  for (const [name, chain] of Object.entries(CHAINS)) ext[name] = fingerExtension(pts, chain);
  debug.ext = ext;

  const indexExt = ext.index;
  const others = [ext.middle, ext.ring, ext.pinky];
  const dominance = indexExt - Math.max(...others);
  if (indexExt < CONST.MIN_INDEX_EXT || dominance < CONST.MIN_INDEX_DOMINANCE) {
    return { gesture: "none", confidence: 0, debug };
  }

  const vx = pts[INDEX_TIP][0] - pts[INDEX_MCP][0];
  const vy = pts[INDEX_TIP][1] - pts[INDEX_MCP][1];
  const norm = Math.hypot(vx, vy);
  if (norm < 1e-6) return { gesture: "none", confidence: 0, debug };
  const fingerSpan = norm / handPx;
  debug.finger_span = fingerSpan;
  if (fingerSpan < CONST.MIN_FINGER_SPAN) return { gesture: "none", confidence: 0, debug };

  const theta = (Math.atan2(-vy, vx) * 180) / Math.PI;
  let gesture;
  if (theta >= -45 && theta < 45) gesture = "right";
  else if (theta >= 45 && theta < 135) gesture = "up";
  else if (theta >= 135 || theta < -135) gesture = "left";
  else gesture = "down";

  const axisRatio = Math.max(Math.abs(vx), Math.abs(vy)) / norm;
  debug.dx = vx / handPx;
  debug.dy = vy / handPx;
  debug.angle = theta;
  if (axisRatio < CONST.MIN_AXIS_RATIO) return { gesture: "none", confidence: 0, debug };

  const extScore = clamp01((indexExt - 0.70) / 0.25);
  const domScore = clamp01((dominance - 0.02) / 0.20);
  const axisScore = clamp01((axisRatio - CONST.AXIS_CONF_FLOOR) / 0.30);
  const sizeScore = clamp01((handRatio - CONST.MIN_HAND_SCALE_RATIO) / 0.10);
  const confidence = clamp01(0.4 * extScore + 0.25 * domScore + 0.25 * axisScore + 0.1 * sizeScore);
  return { gesture, confidence, debug };
}
