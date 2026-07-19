// Service worker: precache the app shell so the PWA works offline.
// Model weights are NOT handled here — transformers.js caches them itself
// in the browser's Cache Storage after the first download.

const CACHE = "token-explorer-v10";
const SHELL = [
  ".",
  "index.html",
  "style.css",
  "app.js",
  "engine-worker.js",
  "manifest.json",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "vendor/transformers.min.js",
  "vendor/ort-wasm-simd-threaded.jsep.mjs",
  "vendor/ort-wasm-simd-threaded.jsep.wasm",
  "vendor/ort-wasm-simd-threaded.asyncify.mjs",
  "vendor/ort-wasm-simd-threaded.asyncify.wasm",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return; // HF downloads: network + tjs cache
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request))
  );
});
