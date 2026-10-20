/* Minimal SVG line-chart renderer.
 *
 * Deliberately dependency-free: no CDN, no build step, nothing to pin. It draws
 * 2px lines on a recessive grid, keeps an 8px hover target, and ships a shared
 * crosshair + tooltip, which is the default interaction layer for a line chart.
 */

const SVG = "http://www.w3.org/2000/svg";

function el(name, attrs = {}) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  return node;
}

const rupees = (value) =>
  "₹" + Math.round(value).toLocaleString("en-IN");

function niceTicks(min, max, count) {
  const span = max - min || 1;
  const rawStep = span / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude)
    .find((candidate) => candidate >= rawStep) || magnitude * 10;
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let value = start; value <= max + step * 0.5; value += step) ticks.push(value);
  return ticks;
}

/**
 * Draw a multi-series line chart into a container.
 *
 * @param {HTMLElement} host container to render into (cleared first)
 * @param {Array} series  [{name, color, points: [{week, value}]}]
 */
export function lineChart(host, series) {
  host.textContent = "";
  const live = series.filter((s) => s.points.some((p) => p.value !== null));
  if (!live.length) {
    host.textContent = "No data for this selection.";
    return;
  }

  const W = 900, H = 320;
  const pad = { top: 14, right: 18, bottom: 32, left: 62 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const weeks = live[0].points.map((p) => p.week);
  const values = live.flatMap((s) => s.points.map((p) => p.value)).filter((v) => v !== null);
  const vMin = Math.min(...values), vMax = Math.max(...values);
  const headroom = (vMax - vMin) * 0.08 || 1;
  const yMin = Math.max(0, vMin - headroom), yMax = vMax + headroom;

  const x = (i) => pad.left + (weeks.length === 1 ? plotW / 2 : (i / (weeks.length - 1)) * plotW);
  const y = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  // recessive grid + y axis labels
  for (const tick of niceTicks(yMin, yMax, 5)) {
    if (tick < yMin || tick > yMax) continue;
    svg.appendChild(el("line", {
      x1: pad.left, x2: W - pad.right, y1: y(tick), y2: y(tick),
      stroke: "var(--gridline)", "stroke-width": 1,
    }));
    const label = el("text", {
      x: pad.left - 9, y: y(tick) + 4, "text-anchor": "end",
      fill: "var(--text-muted)", "font-size": 11,
    });
    label.textContent = rupees(tick);
    svg.appendChild(label);
  }

  // baseline
  svg.appendChild(el("line", {
    x1: pad.left, x2: W - pad.right, y1: pad.top + plotH, y2: pad.top + plotH,
    stroke: "var(--axis)", "stroke-width": 1,
  }));

  // x labels: first, middle, last only - never one per point
  const xTickIdx = weeks.length > 2 ? [0, Math.floor((weeks.length - 1) / 2), weeks.length - 1] : [0, weeks.length - 1];
  for (const i of new Set(xTickIdx)) {
    const label = el("text", {
      x: x(i), y: H - 10, "text-anchor": i === 0 ? "start" : i === weeks.length - 1 ? "end" : "middle",
      fill: "var(--text-muted)", "font-size": 11,
    });
    label.textContent = weeks[i];
    svg.appendChild(label);
  }

  // series paths
  for (const s of live) {
    const d = s.points.map((p, i) => (p.value === null ? null : `${i === 0 ? "M" : "L"}${x(i)},${y(p.value)}`))
      .filter(Boolean).join(" ");
    svg.appendChild(el("path", {
      d, fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      "stroke-dasharray": s.dashed ? "5 4" : null,
    }));
  }

  // hover layer: crosshair + per-point markers
  const crosshair = el("line", {
    y1: pad.top, y2: pad.top + plotH, stroke: "var(--axis)", "stroke-width": 1, opacity: 0,
  });
  svg.appendChild(crosshair);
  const markers = live.map((s) => {
    const dot = el("circle", {
      r: 4.5, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2, opacity: 0,
    });
    svg.appendChild(dot);
    return dot;
  });

  const tooltip = document.querySelector("#tooltip");
  const hit = el("rect", {
    x: pad.left, y: pad.top, width: plotW, height: plotH, fill: "transparent",
  });
  svg.appendChild(hit);

  const show = (event) => {
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const ratio = (px - pad.left) / plotW;
    const i = Math.max(0, Math.min(weeks.length - 1, Math.round(ratio * (weeks.length - 1))));
    crosshair.setAttribute("x1", x(i));
    crosshair.setAttribute("x2", x(i));
    crosshair.setAttribute("opacity", 1);

    let rows = `<div class="tt-week">Week of ${weeks[i]}</div>`;
    live.forEach((s, k) => {
      const value = s.points[i]?.value;
      if (value === null || value === undefined) {
        markers[k].setAttribute("opacity", 0);
        return;
      }
      markers[k].setAttribute("cx", x(i));
      markers[k].setAttribute("cy", y(value));
      markers[k].setAttribute("opacity", 1);
      rows += `<div class="tt-row"><span><span class="swatch" style="background:${s.color}"></span> ${s.name}</span><b>${rupees(value)}</b></div>`;
    });
    tooltip.innerHTML = rows;
    tooltip.style.opacity = 1;
    const tipW = tooltip.offsetWidth;
    tooltip.style.left = Math.min(window.innerWidth - tipW - 12, event.clientX + 14) + "px";
    tooltip.style.top = Math.max(8, event.clientY - 16) + "px";
  };
  const hide = () => {
    tooltip.style.opacity = 0;
    crosshair.setAttribute("opacity", 0);
    markers.forEach((m) => m.setAttribute("opacity", 0));
  };
  hit.addEventListener("mousemove", show);
  hit.addEventListener("mouseleave", hide);
  svg.addEventListener("touchmove", (e) => { if (e.touches[0]) show(e.touches[0]); }, { passive: true });

  host.appendChild(svg);
}

export { rupees };
