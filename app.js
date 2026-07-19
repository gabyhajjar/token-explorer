"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  candidates: [],
  selected: 0,
  temperature: 1.0,
  autoRunning: false,
  busy: false,
};

// ---------- worker RPC ----------

const worker = new Worker("engine-worker.js", { type: "module" });
const pending = new Map();
let nextId = 1;

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "progress") return onProgress(msg.info);
  if (msg.type === "ready") return init();
  const p = pending.get(msg.id);
  if (!p) return;
  pending.delete(msg.id);
  if (msg.ok) p.resolve(msg.data);
  else p.reject(new Error(msg.error));
};
worker.onerror = (e) => showStatus("Engine error: " + e.message);

function rpc(type, payload) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, payload });
  });
}

async function call(type, body) {
  if (state.busy) return null;
  state.busy = true;
  try {
    const data = await rpc(type, { temperature: state.temperature, ...body });
    render(data);
    return data;
  } catch (e) {
    showStatus(e.message);
    return null;
  } finally {
    state.busy = false;
  }
}

// ---------- model loading ----------

const fileProgress = new Map();

function onProgress(info) {
  if (!info) return;
  if (info.status === "retry") {
    const line = document.createElement("div");
    line.textContent = info.message;
    $("load-log").appendChild(line);
    $("load-bar").style.width = "0%";
    fileProgress.clear();
    return;
  }
  if (!info.file) return;
  if (info.status === "progress") fileProgress.set(info.file, info);
  const items = [...fileProgress.values()];
  const loaded = items.reduce((s, i) => s + (i.loaded || 0), 0);
  const total = items.reduce((s, i) => s + (i.total || 0), 0);
  if (total > 0) {
    const pct = Math.min(100, (100 * loaded) / total);
    $("load-bar").style.width = pct.toFixed(1) + "%";
    $("load-text").textContent =
      `Downloading model… ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB ` +
      "(cached after first load)";
  }
}

async function loadModel(name) {
  state.autoRunning = false;
  state.candidates = [];
  fileProgress.clear();
  $("model-select").disabled = true;
  $("start-btn").disabled = true;
  $("controls").hidden = true;
  $("text-section").hidden = true;
  $("candidates-section").hidden = true;
  $("load-panel").hidden = false;
  $("load-text").textContent = "Loading " + name + " …";
  $("load-log").textContent = "";
  $("load-bar").style.width = "0%";
  try {
    const d = await rpc("load", { model: name });
    $("device-badge").textContent = d.device;
    $("chat-toggle").checked = d.has_chat_template;
    applyTemplateInfo(d.has_chat_template);
    $("load-panel").hidden = true;
    showStatus("Loaded " + d.model + " (" + d.device + ") — press Start.");
  } catch (e) {
    $("load-text").textContent = "Failed to load " + name + ": " + e.message;
  } finally {
    $("model-select").disabled = false;
    $("start-btn").disabled = false;
  }
}

async function init() {
  const d = await rpc("models", {});
  const sel = $("model-select");
  sel.textContent = "";
  for (const m of d.available) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  }
  const want = new URLSearchParams(location.search).get("model");
  sel.value = want && d.available.includes(want) ? want : d.current || d.default;
  $("device-badge").textContent = d.device;
  loadModel(sel.value);
}

// ---------- rendering (same as Stage 1) ----------

function visualizeToken(text) {
  const span = document.createElement("span");
  span.className = "tok";
  for (const ch of text) {
    if (ch === " " || ch === "\n" || ch === "\t") {
      const ws = document.createElement("span");
      ws.className = "ws";
      ws.textContent = ch === " " ? "␣" : ch === "\n" ? "⏎" : "⇥";
      span.appendChild(ws);
    } else {
      span.appendChild(document.createTextNode(ch));
    }
  }
  return span;
}

function render(data) {
  state.candidates = data.candidates;
  state.selected = Math.min(state.selected, data.candidates.length - 1);
  if (data.chosen === null || data.chosen === undefined) state.selected = 0;

  $("device-badge").textContent = data.device;
  applyTemplateInfo(data.has_chat_template);
  $("prompt-text").textContent = data.prompt;
  $("generated-text").textContent = data.generated;
  $("controls").hidden = false;
  $("text-section").hidden = false;
  $("candidates-section").hidden = false;

  const showRaw = Math.abs(state.temperature - 1.0) > 1e-9;
  $("col-labels").textContent = showRaw ? "temp %  (raw %)" : "raw %";

  const list = $("candidate-list");
  list.textContent = "";
  const maxTemp = Math.max(...data.candidates.map((c) => c.temp), 1e-12);

  data.candidates.forEach((c, i) => {
    const li = document.createElement("li");
    if (i === state.selected) li.classList.add("selected");

    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.width = (100 * c.temp / maxTemp).toFixed(1) + "%";
    li.appendChild(bar);

    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = i + 1;
    li.appendChild(rank);

    li.appendChild(visualizeToken(c.text));

    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = (100 * c.temp).toFixed(2) + "%";
    li.appendChild(pct);

    if (showRaw) {
      const raw = document.createElement("span");
      raw.className = "pct-raw";
      raw.textContent = "(" + (100 * c.raw).toFixed(2) + "%)";
      li.appendChild(raw);
    }

    li.addEventListener("click", () => {
      state.selected = i;
      chooseSelected();
    });
    list.appendChild(li);
  });

  $("text-display").scrollTop = $("text-display").scrollHeight;
}

function updateSelection() {
  const items = $("candidate-list").children;
  for (let i = 0; i < items.length; i++) {
    items[i].classList.toggle("selected", i === state.selected);
  }
  items[state.selected]?.scrollIntoView({ block: "nearest" });
}

function applyTemplateInfo(has) {
  if (has === undefined) return;
  const cb = $("chat-toggle");
  const label = $("chat-toggle-label");
  cb.disabled = !has;
  label.classList.toggle("disabled", !has);
  if (!has) {
    cb.checked = false;
    label.title = "This is a base model (no chat template) — input is always raw text.";
  } else {
    label.title = "On: wrap the prompt in the model's chat template (model replies to it). Off: feed the text raw (model continues it). Applies on Start.";
  }
}

let statusTimer = null;
function showStatus(msg) {
  const el = $("status");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => (el.hidden = true), 3000);
}

// ---------- actions ----------

function chooseSelected() {
  const c = state.candidates[state.selected];
  if (!c) return;
  call("choose", { token_id: c.id });
}

async function toggleAuto() {
  if (state.autoRunning) {
    state.autoRunning = false;
    return;
  }
  state.autoRunning = true;
  $("auto-btn").classList.add("active");
  $("auto-btn").textContent = "⏸ Pause";
  while (state.autoRunning) {
    const data = await call("auto", {});
    if (!data) break;
    if (data.chosen && data.chosen.is_eos) {
      showStatus("Model produced end-of-text — auto stopped.");
      break;
    }
    await new Promise((r) => setTimeout(r, 60));
  }
  state.autoRunning = false;
  $("auto-btn").classList.remove("active");
  $("auto-btn").textContent = "▷▷ Auto";
}

const debounce = (fn, ms) => {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};
const retemp = debounce(() => call("retemp", {}), 120);

// ---------- wiring ----------

$("model-select").addEventListener("change", (e) => loadModel(e.target.value));

$("start-btn").addEventListener("click", () => {
  state.autoRunning = false;
  const prompt = $("prompt").value;
  if (!prompt.trim()) return showStatus("Enter a prompt first.");
  call("start", { prompt, raw: !$("chat-toggle").checked });
});

$("temp-slider").addEventListener("input", (e) => {
  state.temperature = parseFloat(e.target.value);
  $("temp-value").textContent =
    state.temperature < 0.05 ? "0 (greedy)" : state.temperature.toFixed(2);
  if (state.candidates.length) retemp();
});

$("model-btn").addEventListener("click", () => {
  if (!state.autoRunning) call("auto", {});
});
$("auto-btn").addEventListener("click", toggleAuto);
$("undo-btn").addEventListener("click", () => call("undo", {}));
$("custom-btn").addEventListener("click", () => {
  const text = $("custom-input").value;
  if (!text) return;
  $("custom-input").value = "";
  call("custom", { text });
});
$("custom-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("custom-btn").click();
});

document.addEventListener("keydown", (e) => {
  const inInput = ["TEXTAREA", "INPUT"].includes(document.activeElement.tagName);
  if (inInput || !state.candidates.length) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    state.selected = Math.min(state.selected + 1, state.candidates.length - 1);
    updateSelection();
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    state.selected = Math.max(state.selected - 1, 0);
    updateSelection();
  } else if (e.key === "Enter") {
    e.preventDefault();
    chooseSelected();
  } else if (e.key === "Backspace") {
    e.preventDefault();
    call("undo", {});
  } else if (e.key === "Escape") {
    state.autoRunning = false;
  }
});
