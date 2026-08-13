# NERDS RUN OS — Chunked Build Prompts

Sequential prompts for building the single-file desktop OS HTML, broken into 8 manageable chunks for an autonomous coding agent.

---

## Persistent Context

Include this at the top of every chunk so the agent doesn't drift between turns:

```
You are building a single self-contained HTML file for "Nerds Run LLC" — a CRT/terminal-themed desktop OS in the browser. Constraints: vanilla HTML/CSS/JS only, no frameworks, no imports, no build step. Only allowed external resource is the JetBrains Mono Google Font. Use these CSS variables:

--bg-desktop: #0a0e0a; --bg-window: #0d0f0d; --bg-panel: #080b08;
--green: #22c55e; --green-dim: #22c55e80; --green-faint: #22c55e20;
--blue: #3b82f6; --purple: #a855f7; --red: #ef4444; --yellow: #f59e0b;
--cyan: #06b6d4; --pink: #ec4899;
--text-bright: #22c55eDD; --text-mid: #22c55e90; --text-dim: #22c55e60;
--font: 'JetBrains Mono', monospace;

Output ONLY raw code for the requested chunk. No markdown fences, no preamble.
```

---

## Chunk 1 — Foundation Skeleton

**Goal:** HTML skeleton, CSS reset, font import, CSS variables, visual effects (scanlines + dot grid), body/desktop layout.

```
Build the HTML skeleton and base CSS for the NERDS RUN OS.

Output a complete HTML file with:
1. <!DOCTYPE html>, <html>, <head> with JetBrains Mono link
2. <style> block containing:
   - CSS reset (* { box-sizing, margin: 0, padding: 0 })
   - All :root CSS variables (see persistent context)
   - body styles: bg-desktop, font, color: var(--text-bright), overflow: hidden
   - #scanlines: position fixed, full viewport, repeating-linear-gradient overlay, pointer-events none
   - #dotgrid: position fixed, radial-gradient dots at 24px, opacity 0.03
   - #desktop: full viewport flex container
   - #windows-container: relative positioned area for windows
3. <body> with skeleton: <div id="taskbar"></div><div id="desktop"><div id="scanlines"></div><div id="dotgrid"></div><div id="desktop-icons"></div><div id="windows-container"></div><div id="bottom-tagline"></div></div>
4. Empty <script> block at end of body

This is the foundation. Subsequent chunks will fill in taskbar, icons, windows, and JS.
```

---

## Chunk 2 — Taskbar + Clock

**Goal:** Taskbar markup, styles, and clock JavaScript.

```
Add the top taskbar to the NERDS RUN OS HTML file.

Add to the existing <style> block:
- #taskbar: fixed top, 36px tall, 100% wide, dark semi-transparent bg, green border-bottom, flex layout, z-index 1000
- .taskbar-brand: bright green NR:// with text-shadow glow, clickable
- .taskbar-nav button: transparent bg, green text, hover state
- .taskbar-pills: middle area for window indicators (empty for now)
- .taskbar-clock: right side, monospace
- .wallpaper-toggle: small button with ⚙ symbol

Add to <body> #taskbar:
- Left: <button class="taskbar-brand">NR://</button>
- Nav buttons: README.md, harness-go, services/, GAME.exe (each calls openWindow('id') — function will exist later)
- Middle: <div class="taskbar-pills"></div>
- Right: <button class="wallpaper-toggle">⚙</button> + <span class="taskbar-clock">--:--</span>

Add to <script>:
- updateClock() function that formats current time as "HH:MM AM/PM" and writes to .taskbar-clock
- Call updateClock() and setInterval(updateClock, 30000) on DOMContentLoaded

Output the full updated HTML file.
```

---

## Chunk 3 — Window System Core

**Goal:** The reusable window manager. State, create, close, focus, z-index, drag.

```
Add the vanilla JS window system to the NERDS RUN OS HTML file.

In the existing <script> block, add:

State:
- let windowState = []; // { id, appId, title, x, y, w, h, z, minimized, maximized }
- let zCounter = 10;

Functions:
- openWindow(appId, title, w, h, contentBuilder): If a window with this appId already exists and is visible, focus it. If minimized, restore and focus. Otherwise, create a new window: a div containing a 40px title bar (three colored dots: red close, yellow minimize, green maximize, then centered title) and a content area populated by contentBuilder(). Center on screen with random ±40px jitter. Set z-index from zCounter++. Append to #windows-container. Push to windowState.
- closeWindow(id): Remove the DOM element and the windowState entry.
- minimizeWindow(id): Hide the DOM element with display:none, mark minimized in state, update taskbar pills.
- maximizeWindow(id): Toggle between original and full-desktop size. Store original bounds before maximizing.
- focusWindow(id): Bring to front by setting z-index to ++zCounter.
- makeDraggable(titleBarEl, windowEl, id): mousedown on title bar records offset, mousemove updates left/top, mouseup releases. Use document.addEventListener for move/up so dragging works outside the window.

Add CSS for:
- .window: absolute position, var(--bg-window), green border, box-shadow glow
- .window-titlebar: 40px, dark gradient bg, flex, drag cursor, user-select none
- .window-dot: 12px circle, margin between dots
- .window-dot.close: red, .minimize: yellow, .maximize: green
- .window-dot:hover::before: show ✕, −, □ symbols
- .window-title: centered, var(--text-bright), font-weight 500
- .window-content: flex 1, overflow auto, padding 16px

Output the full updated HTML file.
```

---

## Chunk 4 — Desktop Icons + Wallpapers + Pills

**Goal:** Left-column icons, wallpaper cycling, taskbar running-window pills, bottom tagline.

```
Add the desktop icons, wallpaper system, and taskbar window indicators.

In <script>:
- const wallpapers = [array of 5 CSS gradient strings — varied dark green/black/blue gradients]
- let currentWallpaper = 0
- cycleWallpaper(): increment index modulo 5, apply to #desktop background
- Wire .wallpaper-toggle onclick to cycleWallpaper
- updateTaskbarPills(): rebuild .taskbar-pills with one small button per window in windowState. Dimmed if minimized. onclick: if minimized, restore + focus; else focus.
- Call updateTaskbarPills() at the end of openWindow, closeWindow, minimizeWindow, maximizeWindow.

In <body> #desktop-icons add 6 icon buttons in a left column:
- 📄 README.md → openWindow('readme', 'README.md', 620, 530, buildReadmeContent)
- ⑂ harness-go → openWindow('harness', 'harness-go', 560, 500, buildHarnessContent)
- 📁 services/ → openWindow('services', 'services/', 540, 480, buildServicesContent)
- 🎮 GAME.exe → openWindow('game', 'THE LAST DEPLOYMENT.exe', 480, 520, buildGameContent)
(buildXxxContent stubs return empty string for now — they'll be implemented in later chunks)

Each icon: button with 48px rounded colored div containing the symbol, label below in green monospace.

Add #bottom-tagline: centered at bottom, var(--text-ghost), small text: "NERDS RUN LLC · the last deployment is never the last · pid #" + a random 5-digit number.

Output the full updated HTML file.
```

---

## Chunk 5 — README Window + Morty SVG

**Goal:** The README content + the inline dog SVG. Auto-open on load.

```
Implement buildReadmeContent() for the NERDS RUN OS.

Replace the buildReadmeContent() stub with a function that returns an HTML string containing:

1. Header row: inline SVG of Morty the golden retriever (~60px) next to "NERDS RUN LLC" in large bright green text with text-shadow glow, and subtitle "S-Corp · Lake Geneva, WI · Est. 2024"

Morty SVG must include:
- Floppy golden-brown ears (ellipses, slightly rotated outward)
- Round golden head (#D4A853)
- Lighter face patch (#F5E6C8)
- Two black eyes with white highlight dots
- Black nose with small shine
- Open mouth with pink tongue
- Green collar bar with gold circle tag at the bottom
Must be recognizable as a dog.

2. ASCII art block in <pre>:
  ____  ____  ____  ____  ____
 ||N ||||E ||||R ||||D ||||S ||
 ||__||||__||||__||||__||||__||
 |/__\||/__\||/__\||/__\||/__\|

3. Tagline: "Platform Engineering. Infrastructure Automation. AI/LLM Tooling." and "The last deployment is never the last."

4. Install snippet box with green ❯ prompt and "go install github.com/nerdsrun/harness-go@latest" plus a copy button using navigator.clipboard.writeText(). Show ✓ for 2 seconds after copy.

5. Career list (▶ for current, · for past):
   ▶ GE HealthCare — Staff Software Architect
   · Alaska Airlines — Principal Platform Engineer
   · Expedia — Principal Platform Engineer
   · Northwestern Mutual — Principal Platform Engineer
   · Unity Technologies — Principal Platform Engineer
   · U.S. Military — Service-Connected Veteran

6. Three stat boxes in a row: "42U Home Lab Rack", "128GB Unified Memory", "6+ Rental Properties"

7. Personal section styled as code comments:
   // life outside the terminal
   engaged_to: Stephanie
   dog: Morty 🐕
   hobbies: Korean cooking, house remodeling, real estate
   property: The Silo House (HGTV featured)
   caregiver_for: Dad (Terry)

Add CSS for the README window's elements (.readme-header, .readme-ascii pre, .install-box, .career-list, .stat-box, .personal-block).

At the end of DOMContentLoaded, call openWindow('readme', ...) so README opens on page load.

Output the full updated HTML file.
```

---

## Chunk 6 — harness-go Window

```
Implement buildHarnessContent() for the NERDS RUN OS.

Replace the buildHarnessContent() stub with a function that returns:

1. Toolbar: breadcrumb "⑂ harness-go > pipeline > gen-3", green pulsing dot + "RUNNING" status. Pulse via @keyframes.

2. Generation selector: 5 buttons "gen-1" through "gen-5". The active one (gen-3 by default) has a green border glow. Clicking sets the active class on that button only (no content change needed). Use vanilla JS click handler.

3. Agent pipeline cards (column of 3):
   - PROCTOR — green left border, status: ORCHESTRATING, model: claude-sonnet-4
   - CODER — blue left border, status: GENERATING, model: qwen3-30b-a3b
   - REVIEWER — purple left border, status: EVALUATING, model: kimi-k2.5
   Each card: role name, status badge (colored), model name.

4. Cloud workers table with columns MODEL, REGION, LATENCY, STATUS:
   - MiniMax M2.7 | CN-East | 142ms | READY
   - Kimi K2.5 | CN-North | 189ms | ACTIVE  (latency >150 in yellow)
   - Z.ai GLM-5 | CN-South | 156ms | READY  (latency >150 in yellow)
   - Qwen3-30B | LOCAL | 23ms | ACTIVE  (LOCAL row in bright green)

5. NATS panel (tree style):
   nats:// JetStream consumer
   ├─ pipeline.tasks → 847 delivered
   ├─ pipeline.reviews → 412 delivered
   ├─ pipeline.results → 389 acked
   └─ pipeline.errors → 23 pending

Add CSS for: .pipeline-toolbar, .gen-selector, .agent-card with colored borders, .workers-table, .nats-tree, .pulse-dot @keyframes pulse.

Output the full updated HTML file.
```

---

## Chunk 7 — services/ Window

```
Implement buildServicesContent() for the NERDS RUN OS.

Replace the buildServicesContent() stub with a function that returns:

1. Toolbar: "📁 services > consulting" breadcrumb + grid/list view toggle buttons (▦ for grid, ☰ for list). Default to grid.

2. Grid view: 3-column CSS grid, 6 service cards. Each card: colored Unicode icon or small CSS circle, service name. Click toggles a green outline (selected state). Store selection in a JS variable.
   - Platform Engineering (green)
   - Infrastructure Automation (blue)
   - AI/LLM Tooling (purple)
   - Cloud Architecture (cyan)
   - Observability (yellow)
   - Security & Compliance (red)

3. List view: same 6 items as rows with icon, name, short description text. Hidden by default.

4. View toggle buttons swap which view is visible.

5. Tech stack badges: flex-wrap row of small pill-shaped badges, each with brand-color tinted background:
   Go, Kubernetes, ArgoCD, Cilium, NATS JetStream, PostgreSQL, BadgerDB, Bleve, Talos Linux, Pulumi, Proxmox, llama.cpp

6. Status bar at bottom: shows current selection name (or "no selection") on left, "nerdsrun.llc" on right.

Add CSS for: .services-toolbar, .services-grid, .services-list, .service-card, .service-card.selected, .tech-badges, .status-bar.

Output the full updated HTML file.
```

---

## Chunk 8 — GAME.exe Window + Final Polish

```
Implement buildGameContent() for the NERDS RUN OS and apply final polish.

Replace the buildGameContent() stub with a function that returns:

1. Red "COMING SOON" badge with ⚔️ or 💀 emoji, pulsing border.

2. Title: "NERDS RUN: THE LAST DEPLOYMENT" in large glowing green text.

3. Subtitle: "Isometric ARPG · Go + Ebitengine · UO-Style World Systems"

4. Lore block (in <pre>):
   ⚔️  The servers are dying.
   🏰  The last datacenter stands.
   💀  Deploy... or be deployed.

   Phase 1: Single-player campaign
   Phase 2: Small multiplayer (4-8)

5. Class selector: 2-column grid, 6 buttons. Click highlights with a colored border. Only one selected at a time (single-select).
   - SRE Paladin — Tank/Healer · Runbook of Restoration (green)
   - DevOps Necromancer — Summoner · Staff of Dead Containers (purple)
   - Platform Monk — Support · Fists of kubectl (blue)
   - Security Ranger — DPS · Bow of Zero Trust (red)
   - Data Sorcerer — Mage · Tome of SQL Incantations (yellow)
   - Chaos Engineer — Berserker · Hammer of Fault Injection (pink)

6. Footer: "architecture: v0.2 · PRD: v4 · engine: ebitengine · phase: 1"

Add CSS for: .game-badge, .game-title, .game-lore, .class-grid, .class-button, .class-button.selected.

FINAL POLISH PASS:
- Verify all 4 windows can be dragged by their title bars
- Verify close/minimize/maximize work on all windows
- Verify clicking a window brings it to front (z-index)
- Verify opening an already-open window focuses it instead of duplicating
- Verify clicking a minimized taskbar pill restores the window
- Verify the wallpaper cycle button changes the desktop background
- Verify the clock updates
- No console.log spam, no alerts
- File should be < 800 lines
- Works in Chrome/Firefox/Safari by double-clicking the file

Output the FINAL complete HTML file ready to save and open.
```

---

## Why this chunking works

1. **Each chunk has a single deliverable** — no chunk tries to build two things
2. **Dependencies are linear** — chunk N only references things built in chunks 1..N-1
3. **The persistent context** keeps the design system stable across calls
4. **Final chunk is QA + polish** — gives the model an explicit pass to verify everything works together
5. **Chunks 1–4 build the OS chassis** (~half the work), chunks 5–8 build the apps (the other half)
