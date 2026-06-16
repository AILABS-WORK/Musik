/*
 * Musik — minimal hand-rolled service worker (no build step, no deps).
 *
 * Goal: let the installed PWA launch offline by serving a cached app shell.
 * It does NOT cache API traffic — identify/library calls always hit the
 * network so results stay fresh (and so a stale clip is never replayed).
 *
 * Strategy:
 *   - navigations (HTML)  -> network-first, fall back to cached shell
 *   - same-origin assets  -> stale-while-revalidate
 *   - everything else     -> straight to network (incl. the engine API)
 */
const CACHE = "musik-shell-v1";

// Precache the bits needed to paint the first frame offline. Hashed JS/CSS
// bundles are picked up lazily by the asset handler below, so we keep this
// list tiny and resilient (a missing entry must not fail the whole install).
const SHELL = ["/", "/index.html", "/manifest.webmanifest", "/icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      await Promise.allSettled(SHELL.map((url) => cache.add(url)));
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;

  // Only ever touch same-origin GETs. The engine API (a different host:port,
  // and even same-origin /api/* paths) must always go to the network.
  if (!sameOrigin || url.pathname.startsWith("/api/")) return;

  // App navigations: network-first so deploys land immediately; fall back to
  // the cached shell when offline.
  if (req.mode === "navigate") {
    event.respondWith(
      (async () => {
        try {
          const fresh = await fetch(req);
          const cache = await caches.open(CACHE);
          cache.put("/index.html", fresh.clone());
          return fresh;
        } catch {
          const cache = await caches.open(CACHE);
          return (
            (await cache.match(req)) ??
            (await cache.match("/index.html")) ??
            (await cache.match("/")) ??
            Response.error()
          );
        }
      })(),
    );
    return;
  }

  // Static assets (hashed bundles, icons, fonts): serve cached copy fast,
  // refresh in the background.
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);
      const cached = await cache.match(req);
      const network = fetch(req)
        .then((res) => {
          if (res && res.ok && res.type === "basic") cache.put(req, res.clone());
          return res;
        })
        .catch(() => undefined);
      return cached ?? (await network) ?? Response.error();
    })(),
  );
});
