/* 离线手势作答引擎（MediaPipe Hand Landmarker + 食指规则）。
 *
 * 独立于页面：负责加载模型、开摄像头、输出“上/下/左/右”的作答事件。
 * 用法：
 *   const session = await GestureEngine.start(videoEl, canvasEl, onAnswer, onStatus);
 *   session.stop();
 */
import { FilesetResolver, HandLandmarker } from './vendor/vision_bundle.mjs';
import { classifyHand } from './hand_rules.js';
import { AnswerGate } from './gesture_gate.js';

let vision = null;
let landmarker = null;
let loading = null;
let delegate = null;

export async function load(onStatus) {
  if (landmarker) return { landmarker, delegate };
  if (loading) return loading;
  loading = (async () => {
    if (onStatus) onStatus('正在加载手势模型（约 17 MB）…');
    vision = await FilesetResolver.forVisionTasks('./gesture/vendor/wasm');
    const make = (d) => HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: './gesture/models/hand_landmarker.task', delegate: d },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    try {
      landmarker = await make('GPU');
      delegate = 'GPU';
    } catch (e) {
      landmarker = await make('CPU');
      delegate = 'CPU';
    }
    if (onStatus) onStatus(`手势模型就绪（${delegate}）`);
    return { landmarker, delegate };
  })().finally(() => { loading = null; });
  return loading;
}

function bestOf(hands, w, h) {
  if (!hands || !hands.length) return { gesture: 'none', confidence: 0, landmarks: null, debug: null };
  const rows = hands.map((lms) => ({ ...classifyHand(lms, w, h, true), landmarks: lms }));
  rows.sort((a, b) => (b.gesture !== 'none') - (a.gesture !== 'none') || b.confidence - a.confidence);
  return rows[0];
}

const EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

function draw(canvas, best, w, h) {
  if (!canvas) return;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  if (!best.landmarks) return;
  const pts = best.landmarks.map((p) => [(1 - p.x) * w, p.y * h]); // 镜像绘制
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 3;
  for (const [a, b] of EDGES) {
    ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
  }
  ctx.fillStyle = '#f97316';
  for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); }
  ctx.strokeStyle = '#facc15'; ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(pts[5][0], pts[5][1]); ctx.lineTo(pts[8][0], pts[8][1]); ctx.stroke();
}

export async function start(video, canvas, onAnswer, onStatus) {
  await load(onStatus);
  const media = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    audio: false,
  });
  video.srcObject = media;
  await video.play();
  const gate = new AnswerGate();
  let active = true;
  let last = 0;

  const loop = (now) => {
    if (!active) return;
    requestAnimationFrame(loop);
    if (now - last < 66) return;
    last = now;
    if (video.readyState < 2) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;
    let result;
    try { result = landmarker.detectForVideo(video, now); } catch (e) { return; }
    const best = bestOf(result.landmarks || [], w, h);
    const ev = gate.update(best, now);
    draw(canvas, best, w, h);
    if (onStatus) onStatus(best, ev);
    if (ev.answer && ev.gesture !== 'none' && ev.confidence >= 0.45) {
      onAnswer(ev.gesture, ev.confidence);
      gate.resetQuestion();
    }
  };
  requestAnimationFrame(loop);

  return {
    stop() {
      active = false;
      try { media.getTracks().forEach((t) => t.stop()); } catch (e) {}
      video.srcObject = null;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    },
    resetQuestion() { gate.resetQuestion(); },
  };
}

export function resetQuestion() { /* 当前会话由 start 返回的句柄管理 */ }
