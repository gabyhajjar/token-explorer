// Web Worker that runs the language model in-browser via transformers.js.
// Mirrors the Stage-1 server API: each request returns a "snapshot" with the
// text so far and the top-k candidates (raw + temperature-adjusted probs).

import { AutoTokenizer, AutoModelForCausalLM, Tensor, env } from "./vendor/transformers.min.js";

env.allowLocalModels = false;
// Multithreaded WASM needs SharedArrayBuffer, which needs COOP/COEP headers.
// Without them, force single-thread instead of letting the runtime abort.
if (!self.crossOriginIsolated) {
  env.backends.onnx.wasm.numThreads = 1;
}

const TOP_K = 20;
const GREEDY_THRESHOLD = 0.05;

// dtype per backend: WebGPU gets 4-bit+fp16 for the Qwen models; the WASM
// fallback uses plain q4/q8 (fp16 is not supported there).
const MODELS = {
  "Qwen3-0.6B": { repo: "onnx-community/Qwen3-0.6B-ONNX", webgpu: "q4f16", wasm: "q8" },
  "Qwen3-1.7B": { repo: "onnx-community/Qwen3-1.7B-ONNX", webgpu: "q4f16", wasm: "q8" },
  "GPT-2": { repo: "onnx-community/gpt2-ONNX", webgpu: "fp16", wasm: "q8" },
};
const DEFAULT_MODEL = "Qwen3-0.6B";

let tokenizer = null;
let model = null;
let currentKey = null;
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

async function loadModel(key, notify) {
  const spec = MODELS[key];
  if (!spec) throw new Error(`Unknown model ${key}`);

  tokenizer = null;
  model = null;
  currentKey = null;
  ids = [];
  lastLogits = null;

  tokenizer = await AutoTokenizer.from_pretrained(spec.repo, { progress_callback: notify });

  let dev = await pickDevice();
  for (;;) {
    try {
      model = await AutoModelForCausalLM.from_pretrained(spec.repo, {
        device: dev,
        dtype: spec[dev],
        progress_callback: notify,
      });
      device = dev;
      break;
    } catch (err) {
      if (dev === "webgpu") {
        // GPU backend refused at session creation — retry on CPU.
        dev = "wasm";
        continue;
      }
      throw err;
    }
  }
  currentKey = key;
  collectEosIds();
  return modelInfo();
}

function modelInfo() {
  return {
    model: currentKey,
    device,
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

async function forward() {
  const n = ids.length;
  const input_ids = new Tensor("int64", BigInt64Array.from(ids.map((i) => BigInt(i))), [1, n]);
  const attention_mask = new Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]);
  const out = await model({ input_ids, attention_mask });
  const logits = out.logits; // [1, seq, vocab]
  const [, seq, vocab] = logits.dims;
  const data = logits.data;
  lastLogits = Float32Array.from(data.slice((seq - 1) * vocab, seq * vocab));
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
