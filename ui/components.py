"""Interactive Streamlit components for FaceGuard (Streamlit components v2).

- `face_canvas`: the frame with numbered face boxes. Click a box to hide or
  show that person; drag on empty space to draw a box over a missed face.
- `timeline`: one lane per track across the whole video, with flagged
  frames, scene cuts and the playhead. Click anywhere to jump to that frame.

Both are self-contained HTML/JS (no network, no build step). Everything
passed to them is generated here from numbers and our own JPEG encoding;
no user text is ever injected as HTML.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import streamlit as st

from core.review import flag_runs, frame_runs
from core.types import Box, Track

CANVAS_MAX_PX = 1280  # longer side of the image sent to the browser
CANVAS_JPEG_QUALITY = 85

_CANVAS_CSS = """
.fg-wrap { position: relative; width: 100%; user-select: none; }
.fg-wrap canvas { display: block; max-width: 100%; border-radius: 8px; touch-action: none; }
.fg-hint { font-size: 0.8rem; opacity: 0.7; margin-top: 6px; font-family: var(--st-font, sans-serif); }
.fg-tip { position: absolute; pointer-events: none; background: rgba(15, 23, 42, 0.92); color: #fff;
          font: 12px/1.3 var(--st-font, sans-serif); padding: 4px 8px; border-radius: 6px;
          white-space: nowrap; transform: translate(-50%, -130%); display: none; }
"""

_CANVAS_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let root = parentElement.querySelector('.fg-wrap');
  if (!root) {
    root = document.createElement('div');
    root.className = 'fg-wrap';
    root.innerHTML = '<canvas></canvas><div class="fg-tip"></div><div class="fg-hint"></div>';
    parentElement.appendChild(root);
  }
  const canvas = root.querySelector('canvas');
  const tip = root.querySelector('.fg-tip');
  root.querySelector('.fg-hint').textContent = data.hint || '';
  const ctx = canvas.getContext('2d');
  const W = data.w, H = data.h;
  const COLORS = { on: '#0d9488', off: '#d97706', manual: '#4f46e5', draft: '#db2777' };
  let scale = 1, drag = null, hoverId = null;

  const img = new Image();
  img.onload = () => { layout(); };
  img.src = data.img;

  function layout() {
    const dpr = window.devicePixelRatio || 1;
    const maxW = Math.max(root.clientWidth, 200);
    const maxH = data.max_height || 620;
    scale = Math.min(maxW / W, maxH / H);
    canvas.style.width = Math.round(W * scale) + 'px';
    canvas.style.height = Math.round(H * scale) + 'px';
    canvas.width = Math.round(W * scale * dpr);
    canvas.height = Math.round(H * scale * dpr);
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
    draw();
  }

  function colorOf(b) { return b.manual ? COLORS.manual : (b.redact ? COLORS.on : COLORS.off); }

  function rect(b, color, width, dashed) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width / scale;
    if (dashed) ctx.setLineDash([6 / scale, 4 / scale]);
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.restore();
  }

  function label(b, text, color) {
    const fs = Math.max(11, Math.min(16, 13)) / scale;
    ctx.save();
    ctx.font = `600 ${fs}px sans-serif`;
    const tw = ctx.measureText(text).width + 8 / scale;
    const th = fs + 6 / scale;
    let y = b.y - th - 2 / scale;
    if (y < 0) y = b.y + b.h + 2 / scale;
    ctx.fillStyle = color;
    ctx.fillRect(b.x, y, tw, th);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, b.x + 4 / scale, y + fs);
    ctx.restore();
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    if (img.complete) ctx.drawImage(img, 0, 0, W, H);
    if (data.show_boxes) {
      for (const b of data.boxes) {
        const c = colorOf(b);
        const hot = b.id === hoverId;
        if (!b.redact) {  // visible faces get a light veil so the state reads at a glance
          ctx.save(); ctx.fillStyle = 'rgba(217, 119, 6, 0.14)'; ctx.fillRect(b.x, b.y, b.w, b.h); ctx.restore();
        }
        rect(b, c, hot ? 3.5 : 2, b.flag);
        label(b, b.label, c);
      }
    }
    if (data.draft) { rect(data.draft, COLORS.draft, 2.5, true); label(data.draft, 'new', COLORS.draft); }
    if (drag && drag.moved) {
      const r = norm(drag);
      rect(r, COLORS.draft, 2, true);
    }
  }

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    return { x: Math.max(0, Math.min(W, (e.clientX - r.left) / scale)),
             y: Math.max(0, Math.min(H, (e.clientY - r.top) / scale)),
             px: e.clientX - r.left, py: e.clientY - r.top };
  }
  function norm(d) {
    return { x: Math.min(d.x0, d.x1), y: Math.min(d.y0, d.y1),
             w: Math.abs(d.x1 - d.x0), h: Math.abs(d.y1 - d.y0) };
  }
  function hit(p) {
    if (!data.show_boxes) return null;
    let best = null;
    for (const b of data.boxes) {
      if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) {
        if (!best || b.w * b.h < best.w * best.h) best = b;
      }
    }
    return best;
  }

  canvas.onpointerdown = (e) => {
    const p = pos(e);
    drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, moved: false };
    canvas.setPointerCapture(e.pointerId);
  };
  canvas.onpointermove = (e) => {
    const p = pos(e);
    if (drag) {
      drag.x1 = p.x; drag.y1 = p.y;
      if (Math.abs(drag.x1 - drag.x0) * scale > 6 || Math.abs(drag.y1 - drag.y0) * scale > 6) drag.moved = true;
      tip.style.display = 'none';
      draw();
      return;
    }
    const b = hit(p);
    const id = b ? b.id : null;
    canvas.style.cursor = b ? 'pointer' : (data.can_draw ? 'crosshair' : 'default');
    if (b) {
      tip.textContent = `Person ${b.label}: ${b.redact ? 'hidden' : 'visible'} · click to ${b.redact ? 'show' : 'hide'}`
        + (b.flag ? ' · tracker estimate, check' : '');
      tip.style.left = p.px + 'px'; tip.style.top = p.py + 'px'; tip.style.display = 'block';
    } else {
      tip.style.display = 'none';
    }
    if (id !== hoverId) { hoverId = id; draw(); }
  };
  canvas.onpointerleave = () => { tip.style.display = 'none'; if (hoverId !== null) { hoverId = null; draw(); } };
  canvas.onpointerup = (e) => {
    if (!drag) return;
    const d = drag; drag = null;
    if (d.moved && data.can_draw) {
      const r = norm(d);
      if (r.w >= 4 && r.h >= 4) {
        setTriggerValue('drawn', { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.w), h: Math.round(r.h) });
      }
      draw();
      return;
    }
    const b = hit(pos(e));
    if (b) setTriggerValue('clicked', b.id);
    draw();
  };

  const ro = new ResizeObserver(() => { if (img.complete) layout(); });
  ro.observe(root);
  return () => ro.disconnect();
}
"""

_TIMELINE_CSS = """
.fg-tl { width: 100%; max-height: 260px; overflow-y: auto; font-family: var(--st-font, sans-serif); }
.fg-tl svg { display: block; cursor: pointer; }
.fg-tl text { font-size: 10px; fill: var(--st-text-color, #333); }
.fg-legend { font-size: 0.75rem; opacity: 0.75; margin: 4px 0 0 0; font-family: var(--st-font, sans-serif); }
.fg-legend span { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px;
                  vertical-align: -1px; }
"""

_TIMELINE_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let root = parentElement.querySelector('.fg-tl');
  if (!root) {
    root = document.createElement('div');
    root.className = 'fg-tl';
    parentElement.appendChild(root);
    const legend = document.createElement('div');
    legend.className = 'fg-legend';
    legend.innerHTML = '<span style="background:#0d9488"></span>hidden'
      + '<span style="background:#d97706"></span>left visible'
      + '<span style="background:#4f46e5"></span>manual'
      + '<span style="background:#dc2626"></span>tracker estimate (check)'
      + '<span style="background:#7c3aed"></span>scene cut';
    parentElement.appendChild(legend);
  }
  const NS = 'http://www.w3.org/2000/svg';
  const LANE = 11, GAP = 3, LEFT = 34, TOP = 16, RIGHT = 8;

  function render() {
    const width = Math.max(root.clientWidth, 300);
    const n = Math.max(data.n, 1);
    const plotW = width - LEFT - RIGHT;
    const x = (f) => LEFT + (f / Math.max(n - 1, 1)) * plotW;
    const height = TOP + data.rows.length * (LANE + GAP) + 4;
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('width', width);
    svg.setAttribute('height', Math.max(height, 40));
    const el = (tag, attrs) => { const e = document.createElementNS(NS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };

    // ticks every ~1 s, labelled sparsely
    const fps = data.fps || 30;
    const secs = n / fps;
    const step = secs > 60 ? 10 : secs > 20 ? 5 : secs > 8 ? 2 : 1;
    for (let s = 0; s * fps < n; s += step) {
      el('line', { x1: x(s * fps), x2: x(s * fps), y1: TOP - 4, y2: height, stroke: 'rgba(128,128,128,0.18)' });
      const t = el('text', { x: x(s * fps) + 2, y: 10 }); t.textContent = s + 's';
    }
    data.rows.forEach((r, k) => {
      const y = TOP + k * (LANE + GAP);
      const lab = el('text', { x: 2, y: y + LANE - 2 }); lab.textContent = r.label;
      const color = r.manual ? '#4f46e5' : (r.redact ? '#0d9488' : '#d97706');
      for (const [a, b] of r.spans) {
        el('rect', { x: x(a), y: y, width: Math.max(x(b + 1) - x(a), 1.5), height: LANE, rx: 2,
                     fill: color, opacity: r.redact ? 0.85 : 0.55 });
      }
      for (const [a, b, kind] of r.flags) {
        el('rect', { x: x(a), y: y + LANE - 4, width: Math.max(x(b + 1) - x(a), 1.5), height: 4,
                     fill: kind === 'lost' ? '#dc2626' : '#ef4444', opacity: 0.95 });
      }
    });
    for (const c of data.cuts) {
      el('line', { x1: x(c), x2: x(c), y1: 0, y2: height, stroke: '#7c3aed', 'stroke-width': 2,
                   'stroke-dasharray': '4 3' });
    }
    el('line', { x1: x(data.cur), x2: x(data.cur), y1: 0, y2: height, stroke: '#dc2626', 'stroke-width': 2 });
    el('polygon', { points: `${x(data.cur) - 5},0 ${x(data.cur) + 5},0 ${x(data.cur)},6`, fill: '#dc2626' });

    svg.onclick = (e) => {
      const rr = svg.getBoundingClientRect();
      const f = Math.round(((e.clientX - rr.left - LEFT) / plotW) * (n - 1));
      setTriggerValue('seek', Math.max(0, Math.min(n - 1, f)));
    };
    root.replaceChildren(svg);
  }
  render();
  const ro = new ResizeObserver(render);
  ro.observe(root);
  return () => ro.disconnect();
}
"""

_canvas = st.components.v2.component("faceguard_canvas", css=_CANVAS_CSS, js=_CANVAS_JS)
_timeline = st.components.v2.component("faceguard_timeline", css=_TIMELINE_CSS, js=_TIMELINE_JS)


def jpeg_data_url(img: np.ndarray, max_px: int = CANVAS_MAX_PX, quality: int = CANVAS_JPEG_QUALITY) -> str:
    """Encode a BGR image as a JPEG data URL (OpenCV's encoder takes BGR directly)."""
    h, w = img.shape[:2]
    s = min(1.0, max_px / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def face_canvas(
    img: np.ndarray,
    tracks: list[Track],
    frame: int,
    *,
    key: str,
    show_boxes: bool = True,
    draft: Box | None = None,
    can_draw: bool = True,
    hint: str = "",
    max_height: int = 620,
):
    """Render the editing canvas. Returns the component result (`.clicked`, `.drawn`)."""
    h, w = img.shape[:2]
    boxes = []
    for t in tracks:
        b = t.history.get(frame)
        if b is None:
            continue
        boxes.append({"id": t.id, "x": b.x, "y": b.y, "w": b.w, "h": b.h, "redact": t.redact,
                      "manual": t.source == "manual", "flag": frame in t.flags, "label": str(t.id)})
    data = {
        "img": jpeg_data_url(img), "w": w, "h": h, "boxes": boxes, "show_boxes": show_boxes,
        "draft": draft.__dict__ if draft is not None and draft.area > 0 else None,
        "can_draw": can_draw, "hint": hint, "max_height": max_height,
    }
    return _canvas(key=key, data=data, on_clicked_change=lambda: None, on_drawn_change=lambda: None)


def timeline(tracks: list[Track], n_frames: int, fps: float, current: int, cuts: list[int], *, key: str):
    """Render the track timeline. Returns the component result (`.seek`)."""
    rows = sorted(tracks, key=lambda t: (min(t.history) if t.history else 0, t.id))
    data = {
        "n": n_frames, "fps": fps, "cur": current, "cuts": cuts,
        "rows": [{"label": str(t.id), "redact": t.redact, "manual": t.source == "manual",
                  "spans": [list(r) for r in frame_runs(t.history)],
                  "flags": [list(r) for r in flag_runs(t)]} for t in rows],
    }
    return _timeline(key=key, data=data, on_seek_change=lambda: None)
