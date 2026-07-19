// Web Worker that runs the language model in-browser via transformers.js.
// Mirrors the Stage-1 server API: each request returns a "snapshot" with the
// text so far and the top-k candidates (raw + temperature-adjusted probs).

import { AutoTokenizer, AutoModelForCausalLM, Tensor, env } from "./vendor/transformers.min.js";

env.allowLocalModels = false;
// Serve the ONNX runtime .mjs/.wasm from our vendored copy, never a CDN.
env.backends.onnx.wasm.wasmPaths = new URL("vendor/", self.location.href).href;
// Multithreaded WASM needs SharedArrayBuffer, which needs COOP/COEP headers.
// Without them, force single-thread instead of letting the runtime abort.
if (!self.crossOriginIsolated) {
  env.backends.onnx.wasm.numThreads = 1;
}

const TOP_K = 20;
const GREEDY_THRESHOLD = 0.05;

// Dtype ladders per backend, tried in order until one loads AND produces
// sane logits. fp16 compute (q4f16/fp16) silently returns garbage on some
// mobile GPUs (Adreno), so Android prefers fp32-compute variants first.
// The WASM ladder avoids q4 (its file for Qwen is ~900 MB and overflows the
// 4 GB WASM heap) and fp16 (unsupported there).
const isAndroid = /Android/i.test(self.navigator?.userAgent ?? "");
const MODELS = {
  "Qwen3-0.6B": {
    repo: "onnx-community/Qwen3-0.6B-ONNX",
    webgpu: isAndroid ? ["q4"] : ["q4f16", "q4"],
    wasm: ["q8"],
  },
  "GPT-2": {
    repo: "onnx-community/gpt2-ONNX",
    webgpu: isAndroid ? ["fp32"] : ["fp16", "fp32"],
    wasm: ["q8"],
  },
  "SmolLM2-135M": {
    repo: "HuggingFaceTB/SmolLM2-135M-Instruct",
    webgpu: isAndroid ? ["q4"] : ["q4f16", "q4"],
    wasm: ["q8"],
  },
  "SmolLM2-360M": {
    repo: "HuggingFaceTB/SmolLM2-360M-Instruct",
    webgpu: isAndroid ? ["q4"] : ["q4f16", "q4"],
    wasm: ["q8"],
  },
};
// Phones start on the smallest model; desktop can afford the better one.
const DEFAULT_MODEL = isAndroid ? "SmolLM2-135M" : "Qwen3-0.6B";

let tokenizer = null;
let model = null;
let currentKey = null;
let currentDtype = null;
let device = null;

let ids = [];
let promptLen = 0;
let lastLogits = null; // Float32Array over the vocab
let eosIds = new Set();

// ---------- math over the logits ----------

function maxOf(arr) {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
  return m;
}

function argmaxOf(arr) {
  let m = -Infinity, mi = 0;
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) { m = arr[i]; mi = i; }
  return mi;
}

function sumExp(arr, max, t) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += Math.exp((arr[i] - max) / t);
  return s;
}

function topKIndices(arr, k) {
  const idx = []; // indices sorted by logit, descending
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (idx.length < k || v > arr[idx[idx.length - 1]]) {
      let lo = 0, hi = idx.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[idx[mid]] < v) hi = mid; else lo = mid + 1;
      }
      idx.splice(lo, 0, i);
      if (idx.length > k) idx.pop();
    }
  }
  return idx;
}

function sampleTempered(arr, temperature) {
  if (temperature < GREEDY_THRESHOLD) return argmaxOf(arr);
  const max = maxOf(arr);
  const total = sumExp(arr, max, temperature);
  let r = Math.random() * total;
  for (let i = 0; i < arr.length; i++) {
    r -= Math.exp((arr[i] - max) / temperature);
    if (r <= 0) return i;
  }
  return argmaxOf(arr);
}

// ---------- model plumbing ----------

function collectEosIds() {
  eosIds = new Set();
  for (const source of [model?.config?.eos_token_id, model?.generation_config?.eos_token_id]) {
    if (Array.isArray(source)) source.forEach((t) => eosIds.add(t));
    else if (typeof source === "number") eosIds.add(source);
  }
}

async function pickDevice() {
  // navigator.gpu existing is not enough (e.g. Linux Chrome ships WebGPU
  // disabled): only trust it if an adapter is actually granted.
  if (self.navigator?.gpu) {
    try {
      if (await self.navigator.gpu.requestAdapter()) return "webgpu";
    } catch {}
  }
  return "wasm";
}

async function sanityCheckLogits() {
  // Run one tiny forward pass and reject degenerate output (NaN/Inf or a
  // flat distribution) — e.g. broken fp16 compute on some mobile GPUs
  // silently returns all-zero logits.
  ids = Array.from(tokenizer.encode("Hello"), Number);
  try {
    await forward();
    const max = maxOf(lastLogits);
    if (!Number.isFinite(max)) throw new Error("logits are not finite");
    let min = Infinity;
    for (let i = 0; i < lastLogits.length; i++) if (lastLogits[i] < min) min = lastLogits[i];
    if (max - min < 1e-3) throw new Error("logits are flat (broken GPU compute?)");
  } finally {
    ids = [];
    lastLogits = null;
  }
}

async function loadModel(key, notify) {
  const spec = MODELS[key];
  if (!spec) throw new Error(`Unknown model ${key}`);

  tokenizer = null;
  model = null;
  currentKey = null;
  currentDtype = null;
  ids = [];
  lastLogits = null;

  tokenizer = await AutoTokenizer.from_pretrained(spec.repo, { progress_callback: notify });

  const gpuAttempts = [];
  if ((await pickDevice()) === "webgpu") {
    for (const dt of spec.webgpu) gpuAttempts.push(["webgpu", dt]);
  }
  const cpuAttempts = spec.wasm.map((dt) => ["wasm", dt]);
  // Android Chrome's WebGPU is unreliable at inference time (dropped GPU
  // instance mid-session), so phones get the dependable CPU backend first.
  const attempts = isAndroid ? [...cpuAttempts, ...gpuAttempts] : [...gpuAttempts, ...cpuAttempts];

  const failures = [];
  for (const [dev, dtype] of attempts) {
    notify({ file: "", status: "retry", message: `Trying ${dev}/${dtype} …` });
    try {
      model = await AutoModelForCausalLM.from_pretrained(spec.repo, {
        device: dev,
        dtype,
        progress_callback: notify,
      });
      await sanityCheckLogits();
      device = dev;
      currentDtype = dtype;
      currentKey = key;
      collectEosIds();
      notify({ file: "", status: "retry", message: `✓ ${dev}/${dtype} works` });
      return modelInfo();
    } catch (err) {
      const msg = String(err?.message ?? err);
      failures.push(`${dev}/${dtype}: ${msg}`);
      notify({ file: "", status: "retry", message: `✗ ${dev}/${dtype} failed: ${msg}` });
      try { model?.dispose?.(); } catch {}
      model = null;
    }
  }
  throw new Error("all backends failed — " + (failures.join(" | ") || "none attempted"));
}

function modelInfo() {
  return {
    model: currentKey,
    device: currentDtype ? `${device} ${currentDtype}` : device,
    has_chat_template: !!(tokenizer && tokenizer.chat_template),
  };
}

function encodePrompt(prompt, raw) {
  if (!raw && tokenizer.chat_template) {
    const text = tokenizer.apply_chat_template([{ role: "user", content: prompt }], {
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: false,
    });
    return tokenizer.encode(text, { add_special_tokens: false });
  }
  return tokenizer.encode(prompt);
}

async function forwardOnce() {
  const n = ids.length;
  const input_ids = new Tensor("int64", BigInt64Array.from(ids.map((i) => BigInt(i))), [1, n]);
  const attention_mask = new Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]);
  const out = await model({ input_ids, attention_mask });
  const logits = out.logits; // [1, seq, vocab]
  const [, seq, vocab] = logits.dims;
  const data = logits.data;
  lastLogits = Float32Array.from(data.slice((seq - 1) * vocab, seq * vocab));
}

async function forward() {
  try {
    await forwardOnce();
  } catch (err) {
    // WebGPU can die mid-session (e.g. Android Chrome dropping the GPU
    // instance). Once a model is fully loaded, transparently swap the same
    // model onto WASM — generation state is preserved — and retry once.
    if (device !== "webgpu" || !currentKey) throw err;
    const spec = MODELS[currentKey];
    self.postMessage({
      type: "progress",
      info: { file: "", status: "retry", message: `webgpu died mid-run (${err.message}) — switching to wasm…` },
    });
    try { model?.dispose?.(); } catch {}
    model = null;
    model = await AutoModelForCausalLM.from_pretrained(spec.repo, {
      device: "wasm",
      dtype: spec.wasm[0],
      progress_callback: (info) => self.postMessage({ type: "progress", info }),
    });
    device = "wasm";
    currentDtype = spec.wasm[0];
    await forwardOnce();
  }
}

function snapshot(temperature, chosen = null) {
  const logits = lastLogits;
  const max = maxOf(logits);
  const sumRaw = sumExp(logits, max, 1.0);
  const greedy = temperature < GREEDY_THRESHOLD;
  const sumT = greedy ? 1 : sumExp(logits, max, temperature);
  const argmax = greedy ? argmaxOf(logits) : -1;

  const candidates = topKIndices(logits, TOP_K).map((i) => ({
    id: i,
    text: tokenizer.decode([i]),
    raw: Math.exp(logits[i] - max) / sumRaw,
    temp: greedy ? (i === argmax ? 1 : 0) : Math.exp((logits[i] - max) / temperature) / sumT,
  }));

  const decodeSafe = (arr) => (arr.length ? tokenizer.decode(arr) : "");
  return {
    prompt: decodeSafe(ids.slice(0, promptLen)),
    generated: decodeSafe(ids.slice(promptLen)),
    n_generated: ids.length - promptLen,
    candidates,
    chosen,
    ...modelInfo(),
  };
}

function requireSession() {
  if (!lastLogits) throw new Error("No active session. Press Start first.");
}

function makeChosen(tokenId, text = null) {
  return {
    id: tokenId,
    text: text ?? tokenizer.decode([tokenId]),
    is_eos: eosIds.has(tokenId),
  };
}

// ---------- request handlers (mirror the Stage-1 endpoints) ----------

const handlers = {
  models: async () => ({
    available: Object.keys(MODELS),
    current: currentKey,
    default: DEFAULT_MODEL,
    device: device ?? (self.navigator?.gpu ? "webgpu" : "wasm"),
    has_chat_template: !!(tokenizer && tokenizer.chat_template),
  }),

  load: async (p, notify) => loadModel(p.model, notify),

  start: async (p) => {
    if (!model) throw new Error("Model still loading.");
    const newIds = encodePrompt(p.prompt, p.raw);
    if (!newIds.length) throw new Error("Prompt is empty.");
    ids = Array.from(newIds, Number);
    promptLen = ids.length;
    await forward();
    return snapshot(p.temperature);
  },

  choose: async (p) => {
    requireSession();
    const chosen = makeChosen(p.token_id);
    ids.push(p.token_id);
    await forward();
    return snapshot(p.temperature, chosen);
  },

  auto: async (p) => {
    requireSession();
    const tokenId = sampleTempered(lastLogits, p.temperature);
    const chosen = makeChosen(tokenId);
    ids.push(tokenId);
    await forward();
    return snapshot(p.temperature, chosen);
  },

  custom: async (p) => {
    requireSession();
    const newIds = Array.from(tokenizer.encode(p.text, { add_special_tokens: false }), Number);
    if (!newIds.length) throw new Error("Custom text is empty.");
    const chosen = makeChosen(newIds[newIds.length - 1], p.text);
    ids.push(...newIds);
    await forward();
    return snapshot(p.temperature, chosen);
  },

  undo: async (p) => {
    requireSession();
    if (ids.length <= promptLen) throw new Error("Nothing to undo.");
    ids.pop();
    await forward();
    return snapshot(p.temperature);
  },

  retemp: async (p) => {
    requireSession();
    return snapshot(p.temperature);
  },
};

self.onmessage = async (e) => {
  const { id, type, payload } = e.data;
  const notify = (info) => self.postMessage({ type: "progress", info });
  try {
    const handler = handlers[type];
    if (!handler) throw new Error(`Unknown request ${type}`);
    const data = await handler(payload ?? {}, notify);
    self.postMessage({ id, ok: true, data });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message ?? err) });
  }
};

self.postMessage({ type: "ready" });
