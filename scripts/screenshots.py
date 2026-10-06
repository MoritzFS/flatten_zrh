"""Screenshots of the built pages, and the site's social preview image.

Serves ``site/`` on a local port, opens the route finder and the explorer in
headless Chromium, waits for them to route, and writes:

* ``outputs/screenshot_route_finder.png`` -- the route finder on its default
  trip;
* ``site/preview.jpg`` -- the 1200x630 social card the page's Open Graph
  tags point at;
* ``outputs/screenshot_interactive.png`` and ``outputs/screenshot_warped.png``
  -- the explorer, plain and with the city warped by climbing cost.

Usage: python scripts/screenshots.py   (after ``python -m zrh_flat_routes map``)
"""
from __future__ import annotations

import functools
import glob
import http.server
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = ROOT / "outputs"


def chromium() -> str | None:
    for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",
                    os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main() -> int:
    from playwright.sync_api import sync_playwright

    httpd, port = serve(SITE)
    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium(),
                                     args=["--no-sandbox", "--disable-gpu"])

        def page(w, h, scale=1):
            p = browser.new_page(viewport={"width": w, "height": h},
                                 device_scale_factor=scale)
            p.on("pageerror", lambda e: errors.append(str(e)))
            p.on("console", lambda m: errors.append(m.text)
                 if m.type == "error" and "fonts.g" not in m.text
                 and "ERR_" not in m.text else None)
            return p

        ready = ("window.App && App.family && !App.family.partial"
                 " && !document.getElementById('result').hidden")

        # the route finder, and its social card
        p = page(1440, 900)
        p.goto(f"http://127.0.0.1:{port}/", wait_until="load", timeout=240_000)
        p.wait_for_function(ready, timeout=240_000)
        p.wait_for_timeout(1200)
        p.screenshot(path=str(OUT / "screenshot_route_finder.png"))
        print(p.evaluate("document.getElementById('status').textContent"))
        p.close()

        p = page(1200, 630)
        p.goto(f"http://127.0.0.1:{port}/", wait_until="load", timeout=240_000)
        p.wait_for_function(ready, timeout=240_000)
        p.evaluate("App.fit()")
        p.wait_for_timeout(1500)
        p.screenshot(path=str(SITE / "preview.jpg"), type="jpeg", quality=86)
        p.close()

        # the explorer: plain, then warped
        explorer = OUT / "zrh_flat_routes_map.html"
        if explorer.exists():
            p = page(1440, 900)
            p.goto(explorer.resolve().as_uri(), wait_until="load", timeout=240_000)
            p.wait_for_function("window.App && App.lastRoute", timeout=240_000)
            p.wait_for_timeout(1500)
            p.screenshot(path=str(OUT / "screenshot_interactive.png"))
            p.click("#warptoggle")
            p.wait_for_function("App.warpOn && App.warp && document.getElementById('busy')"
                                ".style.display !== 'flex'", timeout=240_000)
            p.wait_for_timeout(2500)
            p.screenshot(path=str(OUT / "screenshot_warped.png"))
            p.close()
        browser.close()
    httpd.shutdown()
    if errors:
        print("page errors:", *errors[:10], sep="\n  ")
        return 1
    print("wrote", OUT / "screenshot_route_finder.png", SITE / "preview.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
