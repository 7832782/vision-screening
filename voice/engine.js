/* 离线语音作答引擎（Vosk WASM，四字硬语法）。
 *
 * 独立于页面：只负责“加载模型 + 听麦克风 + 输出方向”，页面决定怎么作答。
 * 用法：
 *   const listener = VoiceEngine.createListener({ onResult, onStatus });
 *   await listener.start();   // 首次会加载约 44 MB 模型
 *   listener.stop();
 */
(function () {
  'use strict';

  const SR = 16000;
  const GRAMMAR = JSON.stringify(['上', '下', '左', '右', '看不清', '看不见', '不知道', '[unk]']);
  const DIR = { '上': 'up', '下': 'down', '左': 'left', '右': 'right' };
  let model = null;
  let modelPromise = null;
  let modelMs = 0;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      if (window.Vosk) return resolve();
      const s = document.createElement('script');
      s.src = src;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('无法加载 ' + src));
      document.head.appendChild(s);
    });
  }

  async function load(onStatus) {
    if (model) return model;
    if (modelPromise) return modelPromise;
    modelPromise = (async () => {
      const t0 = performance.now();
      if (onStatus) onStatus('正在加载离线语音运行时…');
      await loadScript('voice/vosk.js');
      if (onStatus) onStatus('正在加载语音模型（约 44 MB，首次较慢）…');
      // 必须用绝对 URL：Vosk 的 Blob Worker 里 location 是 blob: 地址，
      // 相对路径在 GitHub Pages 的子目录部署下会被解析到站点根目录。
      const modelUrl = new URL('voice/model.tar.gz', window.location.href).href;
      model = await window.Vosk.createModel(modelUrl);
      modelMs = performance.now() - t0;
      if (onStatus) onStatus(`语音模型就绪（${(modelMs / 1000).toFixed(1)}s）`);
      return model;
    })().finally(() => { modelPromise = null; });
    return modelPromise;
  }

  function mapText(text) {
    const t = String(text || '');
    if (/看不清|看不见|不知道|不清楚|看不到/.test(t)) return 'unsure';
    for (const [cn, dir] of Object.entries(DIR)) if (t.includes(cn)) return dir;
    return null;
  }

  async function recognize(float32) {
    if (!model) await load();
    const t0 = performance.now();
    const recognizer = new model.KaldiRecognizer(SR, GRAMMAR);
    let text = '';
    let err = '';
    let results = 0;
    recognizer.on('result', (msg) => {
      results += 1;
      const t = msg && msg.result && msg.result.text ? String(msg.result.text).trim() : '';
      if (t) text = t;
    });
    recognizer.on('error', (msg) => { err = (msg && msg.error) || 'unknown'; });
    const block = 3200; // 0.2s
    for (let i = 0; i < float32.length; i += block) {
      recognizer.acceptWaveformFloat(float32.slice(i, i + block), SR);
    }
    recognizer.retrieveFinalResult();
    await new Promise((resolve) => {
      const started = performance.now();
      const timer = setInterval(() => {
        if (results > 0 || err || performance.now() - started > 5000) {
          clearInterval(timer);
          resolve();
        }
      }, 25);
    });
    try { recognizer.remove(); } catch (e) {}
    if (err) throw new Error(err);
    return { text, dir: mapText(text), ms: Math.round(performance.now() - t0) };
  }

  function resample(samples, fromRate) {
    if (Math.abs(fromRate - SR) < 20) return samples;
    const outLen = Math.max(1, Math.round(samples.length * SR / fromRate));
    const out = new Float32Array(outLen);
    const step = fromRate / SR;
    for (let i = 0; i < outLen; i++) {
      const pos = i * step;
      const i0 = Math.floor(pos);
      const i1 = Math.min(samples.length - 1, i0 + 1);
      const f = pos - i0;
      out[i] = samples[i0] * (1 - f) + samples[i1] * f;
    }
    return out;
  }

  function concat(parts) {
    let n = 0;
    for (const p of parts) n += p.length;
    const out = new Float32Array(n);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  function createListener(opts) {
    const onResult = opts && opts.onResult ? opts.onResult : function () {};
    const onStatus = opts && opts.onStatus ? opts.onStatus : function () {};
    let ctx = null;
    let stream = null;
    let node = null;
    let source = null;
    let sink = null;
    let active = false;
    let busy = false;
    let speech = false;
    let silence = 0;
    let pre = [];
    let frames = [];
    let cooldownUntil = 0;

    function rms(frame) {
      let s = 0;
      for (let i = 0; i < frame.length; i++) s += frame[i] * frame[i];
      return Math.sqrt(s / frame.length);
    }

    function stopTracks() {
      try { source && source.disconnect(); } catch (e) {}
      try { node && node.disconnect(); } catch (e) {}
      try { sink && sink.disconnect(); } catch (e) {}
      try { stream && stream.getTracks().forEach((t) => t.stop()); } catch (e) {}
      try { ctx && ctx.close(); } catch (e) {}
      stream = null; node = null; source = null; sink = null; ctx = null;
    }

    async function finalize() {
      if (!frames.length) { speech = false; pre = []; return; }
      const all = concat(frames);
      frames = []; pre = []; speech = false; silence = 0;
      busy = true;
      try {
        onStatus('正在识别…');
        const x = resample(all, ctx ? ctx.sampleRate : SR);
        if (x.length < 1600) return;
        const r = await recognize(x);
        onResult(r);
      } catch (e) {
        onStatus('识别失败：' + (e.message || e));
      } finally {
        busy = false;
        cooldownUntil = performance.now() + 1000;
      }
    }

    async function start() {
      if (active) return;
      await load(onStatus);
      const media = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      stream = media;
      let ac;
      try { ac = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR }); }
      catch (e) { ac = new (window.AudioContext || window.webkitAudioContext)(); }
      if (ac.state === 'suspended') { try { await ac.resume(); } catch (e) {} }
      ctx = ac;
      source = ac.createMediaStreamSource(stream);
      node = ac.createScriptProcessor(2048, 1, 1);
      sink = ac.createGain();
      sink.gain.value = 0;
      node.onaudioprocess = (ev) => {
        if (!active || busy) return;
        if (performance.now() < cooldownUntil) return;
        if (performance.now() < (window.__answerMuteUntil || 0)) return;
        const d = ev.inputBuffer.getChannelData(0);
        const f = new Float32Array(d.length);
        f.set(d);
        const level = rms(f);
        if (!speech) {
          pre.push(f);
          if (pre.length > 6) pre.shift();
          if (level > 0.018) {
            speech = true;
            frames = pre.slice();
            silence = 0;
          }
          return;
        }
        frames.push(f);
        if (level < 0.012) silence += 1; else silence = 0;
        // 2048/16k = 128ms；5 帧 ≈ 0.64s 静音收句
        if (silence >= 5 || frames.length > 150) finalize();
      };
      source.connect(node);
      node.connect(sink);
      sink.connect(ac.destination);
      active = true;
      onStatus('正在听…说「上 / 下 / 左 / 右」');
    }

    function stop() {
      active = false;
      stopTracks();
      onStatus('已停止');
    }

    return { start, stop };
  }

  window.VoiceEngine = { load, createListener };
})();
