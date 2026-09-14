(() => {
  const ARCH = window.ARCH;
  const svg = document.getElementById("flow-svg");
  const edgesG = document.getElementById("edges");
  const nodesG = document.getElementById("nodes");
  const particlesG = document.getElementById("particles");
  const breadcrumb = document.getElementById("breadcrumb");
  const detailEmpty = document.getElementById("detail-empty");
  const detailContent = document.getElementById("detail-content");
  const canvasWrap = document.getElementById("canvas-wrap");
  const canvasHint = document.getElementById("canvas-hint");

  let focusId = null;
  let trail = ["root"];
  let currentView = "flow";
  let animRunning = false;
  let particles = [];
  let transform = { x: 0, y: 0, k: 1 };
  let drag = null;
  let rafId = null;

  const kindColor = {
    source: "var(--source)",
    transform: "var(--transform)",
    estimate: "var(--estimate)",
    output: "var(--output)",
    support: "var(--support)",
  };

  function node(id) {
    return ARCH.nodes[id];
  }

  function layoutFor(scopeId) {
    if (scopeId === "root") return ARCH.layouts.root;
    const n = node(scopeId);
    if (!n || !n.children.length) return null;
    const kids = n.children;
    const layout = {};
    const cols = Math.min(3, kids.length);
    const rows = Math.ceil(kids.length / cols);
    const startX = 180;
    const startY = 160;
    const gapX = 280;
    const gapY = 160;
    kids.forEach((id, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      layout[id] = {
        x: startX + col * gapX,
        y: startY + row * gapY + (rows === 1 ? 80 : 0),
        w: 200,
        h: 88,
      };
    });
    // parent hub
    layout[scopeId] = { x: 500, y: 40, w: 220, h: 72, hub: true };
    return layout;
  }

  function edgesFor(scopeId, layout) {
    if (scopeId === "root") {
      return ARCH.edges.filter((e) => layout[e.from] && layout[e.to]);
    }
    const n = node(scopeId);
    if (!n) return [];
    return n.children.map((c) => ({ from: scopeId, to: c }));
  }

  function clearAnim() {
    animRunning = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    particles = [];
    particlesG.innerHTML = "";
  }

  function centerPoint(box) {
    return { x: box.x + box.w / 2, y: box.y + box.h / 2 };
  }

  function route(a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    if (Math.abs(dx) < 8) {
      return `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
    }
    const mx = a.x + dx * 0.5;
    return `M ${a.x} ${a.y} C ${mx} ${a.y}, ${mx} ${b.y}, ${b.x} ${b.y}`;
  }

  function applyTransform() {
    const g = svg.querySelector("g.viewport");
    if (g) {
      g.setAttribute(
        "transform",
        `translate(${transform.x}, ${transform.y}) scale(${transform.k})`
      );
    }
  }

  function renderGraph(scopeId) {
    clearAnim();
    const layout = layoutFor(scopeId);
    if (!layout) return;

    edgesG.innerHTML = "";
    nodesG.innerHTML = "";

    applyTransform();

    const edgeList = edgesFor(scopeId, layout);
    edgeList.forEach((e) => {
      const A = layout[e.from];
      const B = layout[e.to];
      if (!A || !B) return;
      const a = centerPoint(A);
      const b = centerPoint(B);
      // attach to box edges roughly
      const start = {
        x: a.x + (b.x > a.x ? A.w / 2 - 4 : b.x < a.x ? -A.w / 2 + 4 : 0),
        y: a.y + (Math.abs(b.x - a.x) < A.w ? (b.y > a.y ? A.h / 2 - 4 : -A.h / 2 + 4) : 0),
      };
      const end = {
        x: b.x + (a.x > b.x ? B.w / 2 - 4 : a.x < b.x ? -B.w / 2 + 4 : 0),
        y: b.y + (Math.abs(a.x - b.x) < B.w ? (a.y > b.y ? B.h / 2 - 4 : -B.h / 2 + 4) : 0),
      };
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", route(start, end));
      path.setAttribute("class", "edge-path" + (e.dashed ? " dashed" : ""));
      if (e.dashed) path.setAttribute("stroke-dasharray", "6 5");
      path.dataset.from = e.from;
      path.dataset.to = e.to;
      edgesG.appendChild(path);

      spawnParticle(path, e.from === focusId || e.to === focusId);
    });

    startParticleLoop();

    const ids = Object.keys(layout);
    ids.forEach((id) => {
      const n = node(id);
      const box = layout[id];
      if (!n || !box) return;
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.classList.add("node-group", n.kind);
      if (id === focusId) g.classList.add("active");
      if (box.hub) g.classList.add("hub");
      g.dataset.id = id;
      g.setAttribute("tabindex", "0");
      g.setAttribute("role", "button");
      g.setAttribute("aria-label", n.title);

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.classList.add("node-body");
      rect.setAttribute("x", box.x);
      rect.setAttribute("y", box.y);
      rect.setAttribute("width", box.w);
      rect.setAttribute("height", box.h);
      rect.setAttribute("rx", 14);
      rect.setAttribute("ry", 14);

      const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
      title.classList.add("node-title");
      title.setAttribute("x", box.x + 16);
      title.setAttribute("y", box.y + 32);
      title.textContent = n.title;

      const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
      sub.classList.add("node-sub");
      sub.setAttribute("x", box.x + 16);
      sub.setAttribute("y", box.y + 52);
      sub.textContent = n.subtitle;

      if (n.children && n.children.length && !box.hub) {
        const badge = document.createElementNS("http://www.w3.org/2000/svg", "text");
        badge.classList.add("node-sub");
        badge.setAttribute("x", box.x + box.w - 14);
        badge.setAttribute("y", box.y + box.h - 14);
        badge.setAttribute("text-anchor", "end");
        badge.setAttribute("fill", kindColor[n.kind] || "#8a9e97");
        badge.textContent = "↳ drill";
        g.append(rect, title, sub, badge);
      } else {
        g.append(rect, title, sub);
      }

      g.addEventListener("click", (ev) => {
        ev.stopPropagation();
        selectNode(id, { drill: n.children && n.children.length > 0 && id !== scopeId });
      });
      g.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          selectNode(id, { drill: n.children && n.children.length > 0 && id !== scopeId });
        }
      });

      nodesG.appendChild(g);
    });

    // Highlight connected edges
    highlightEdges();
    canvasHint.textContent =
      scopeId === "root"
        ? "Click any stage to drill in · Esc to go up · drag to pan · scroll to zoom"
        : `Inside ${node(scopeId).title} · click a child · Esc / breadcrumb to go up`;
  }

  function spawnParticle(pathEl, hot) {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("r", hot ? 3.2 : 2.2);
    dot.setAttribute("class", "flow-dot");
    if (hot) dot.style.fill = "#d4a574";
    particlesG.appendChild(dot);
    particles.push({
      el: dot,
      path: pathEl,
      t: Math.random(),
      speed: 0.12 + Math.random() * 0.1,
    });
  }

  function startParticleLoop() {
    if (animRunning) return;
    animRunning = true;
    const tick = () => {
      if (!animRunning) return;
      particles.forEach((p) => {
        const len = p.path.getTotalLength();
        if (!len) return;
        p.t += p.speed * 0.016;
        if (p.t > 1) p.t = 0;
        const pt = p.path.getPointAtLength(p.t * len);
        p.el.setAttribute("cx", pt.x);
        p.el.setAttribute("cy", pt.y);
      });
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }

  function highlightEdges() {
    edgesG.querySelectorAll(".edge-path").forEach((p) => {
      const hot =
        focusId && (p.dataset.from === focusId || p.dataset.to === focusId);
      p.classList.toggle("hot", Boolean(hot));
    });
  }

  function renderMath(el, latex) {
    if (window.katex) {
      katex.render(latex, el, {
        throwOnError: false,
        displayMode: true,
        trust: false,
      });
    } else {
      el.textContent = latex;
    }
  }

  function showDetail(id) {
    const n = node(id);
    if (!n) return;
    detailEmpty.classList.add("hidden");
    detailContent.classList.remove("hidden");

    document.getElementById("detail-layer").textContent = n.layer;
    document.getElementById("detail-title").textContent = n.title;
    document.getElementById("detail-file").textContent = n.file;
    document.getElementById("detail-desc").textContent = n.desc;

    const inUl = document.getElementById("detail-inputs");
    const outUl = document.getElementById("detail-outputs");
    inUl.innerHTML = n.inputs.map((x) => `<li>${escapeHtml(x)}</li>`).join("");
    outUl.innerHTML = n.outputs.map((x) => `<li>${escapeHtml(x)}</li>`).join("");

    const fSec = document.getElementById("formula-section");
    const fBox = document.getElementById("detail-formulas");
    if (n.formulas && n.formulas.length) {
      fSec.classList.remove("hidden");
      fBox.innerHTML = "";
      n.formulas.forEach((f) => {
        const card = document.createElement("div");
        card.className = "formula-card";
        const h = document.createElement("h4");
        h.textContent = f.title;
        const math = document.createElement("div");
        renderMath(math, f.latex);
        card.append(h, math);
        if (f.note) {
          const note = document.createElement("p");
          note.className = "note";
          note.textContent = f.note;
          card.append(note);
        }
        fBox.appendChild(card);
      });
    } else {
      fSec.classList.add("hidden");
      fBox.innerHTML = "";
    }

    const cSec = document.getElementById("children-section");
    const cBox = document.getElementById("detail-children");
    if (n.children && n.children.length) {
      cSec.classList.remove("hidden");
      cBox.innerHTML = "";
      n.children.forEach((cid) => {
        const child = node(cid);
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "child-btn";
        btn.innerHTML = `<strong>${escapeHtml(child.title)}</strong><span>${escapeHtml(
          child.subtitle
        )}</span>`;
        btn.addEventListener("click", () => selectNode(cid, { drill: true }));
        cBox.appendChild(btn);
      });
    } else {
      cSec.classList.add("hidden");
      cBox.innerHTML = "";
    }

    const fnSec = document.getElementById("funcs-section");
    const fnUl = document.getElementById("detail-funcs");
    if (n.functions && n.functions.length) {
      fnSec.classList.remove("hidden");
      fnUl.innerHTML = n.functions
        .map((fn) => `<li><code>${escapeHtml(fn)}</code></li>`)
        .join("");
    } else {
      fnSec.classList.add("hidden");
      fnUl.innerHTML = "";
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function updateBreadcrumb() {
    breadcrumb.innerHTML = "";
    const labels = trail.map((id) =>
      id === "root" ? "System" : node(id)?.title || id
    );
    trail.forEach((id, i) => {
      if (i > 0) {
        const sep = document.createElement("span");
        sep.className = "sep";
        sep.textContent = "/";
        breadcrumb.appendChild(sep);
      }
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = labels[i];
      btn.dataset.crumb = id;
      btn.addEventListener("click", () => {
        trail = trail.slice(0, i + 1);
        focusId = id === "root" ? null : id;
        transform = { x: 0, y: 0, k: 1 };
        renderGraph(trail[trail.length - 1]);
        if (focusId) showDetail(focusId);
        else {
          detailContent.classList.add("hidden");
          detailEmpty.classList.remove("hidden");
        }
        updateBreadcrumb();
      });
      breadcrumb.appendChild(btn);
    });
  }

  function selectNode(id, { drill = false } = {}) {
    const n = node(id);
    if (!n) return;
    focusId = id;
    showDetail(id);

    const scope = trail[trail.length - 1];

    // Enter a node's own child canvas
    if (drill && n.children && n.children.length) {
      if (trail[trail.length - 1] !== id) trail.push(id);
      transform = { x: 0, y: 0, k: 1 };
      renderGraph(id);
      updateBreadcrumb();
      return;
    }

    // Leaf selected from detail "Drill into": open the parent's child canvas
    const parent = Object.values(ARCH.nodes).find(
      (p) => p.children && p.children.includes(id)
    );
    if (drill && parent && (scope === "root" || scope !== parent.id)) {
      trail = ["root", parent.id];
      transform = { x: 0, y: 0, k: 1 };
      renderGraph(parent.id);
      updateBreadcrumb();
      highlightEdges();
      nodesG.querySelectorAll(".node-group").forEach((g) => {
        g.classList.toggle("active", g.dataset.id === id);
      });
      return;
    }

    renderGraph(scope);
    highlightEdges();
    nodesG.querySelectorAll(".node-group").forEach((g) => {
      g.classList.toggle("active", g.dataset.id === id);
    });
  }

  function goUp() {
    if (trail.length <= 1) {
      focusId = null;
      detailContent.classList.add("hidden");
      detailEmpty.classList.remove("hidden");
      renderGraph("root");
      updateBreadcrumb();
      return;
    }
    trail.pop();
    const scope = trail[trail.length - 1];
    focusId = scope === "root" ? null : scope;
    transform = { x: 0, y: 0, k: 1 };
    renderGraph(scope);
    if (focusId) showDetail(focusId);
    else {
      detailContent.classList.add("hidden");
      detailEmpty.classList.remove("hidden");
    }
    updateBreadcrumb();
  }

  /* Pan / zoom */
  canvasWrap.addEventListener("wheel", (e) => {
    if (currentView !== "flow") return;
    e.preventDefault();
    const rect = canvasWrap.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const prev = transform.k;
    const next = Math.min(2.2, Math.max(0.55, prev * (e.deltaY < 0 ? 1.08 : 0.92)));
    const sx = (mx - transform.x) / prev;
    const sy = (my - transform.y) / prev;
    transform.k = next;
    transform.x = mx - sx * next;
    transform.y = my - sy * next;
    applyTransform();
  }, { passive: false });

  canvasWrap.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".node-group")) return;
    drag = { x: e.clientX, y: e.clientY, ox: transform.x, oy: transform.y };
    canvasWrap.classList.add("dragging");
    canvasWrap.setPointerCapture(e.pointerId);
  });
  canvasWrap.addEventListener("pointermove", (e) => {
    if (!drag) return;
    transform.x = drag.ox + (e.clientX - drag.x);
    transform.y = drag.oy + (e.clientY - drag.y);
    applyTransform();
  });
  canvasWrap.addEventListener("pointerup", () => {
    drag = null;
    canvasWrap.classList.remove("dragging");
  });

  document.getElementById("reset-view").addEventListener("click", () => {
    transform = { x: 0, y: 0, k: 1 };
    applyTransform();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") goUp();
  });

  /* Views */
  document.querySelectorAll(".chip[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentView = btn.dataset.view;
      document.querySelectorAll(".chip[data-view]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn))
      );
      const flow = document.querySelector(".layout");
      const gal = document.getElementById("formula-gallery");
      const mods = document.getElementById("module-grid");
      flow.classList.toggle("hidden", currentView !== "flow");
      gal.classList.toggle("hidden", currentView !== "formulas");
      mods.classList.toggle("hidden", currentView !== "modules");
      if (currentView === "formulas") renderGallery();
      if (currentView === "modules") renderModules();
    });
  });

  function renderGallery() {
    const grid = document.getElementById("gallery-grid");
    grid.innerHTML = "";
    Object.values(ARCH.nodes)
      .filter((n) => n.formulas && n.formulas.length)
      .forEach((n) => {
        n.formulas.forEach((f) => {
          const item = document.createElement("article");
          item.className = "gallery-item";
          item.innerHTML = `<span class="stage-tag">${escapeHtml(
            n.title
          )}</span><h3>${escapeHtml(f.title)}</h3>`;
          const math = document.createElement("div");
          renderMath(math, f.latex);
          item.appendChild(math);
          if (f.note) {
            const note = document.createElement("p");
            note.className = "note";
            note.style.color = "var(--muted)";
            note.style.fontSize = "0.82rem";
            note.textContent = f.note;
            item.appendChild(note);
          }
          const jump = document.createElement("button");
          jump.type = "button";
          jump.className = "chip";
          jump.style.marginTop = "0.75rem";
          jump.textContent = "Show in flow";
          jump.addEventListener("click", () => {
            document.querySelector('.chip[data-view="flow"]').click();
            // navigate to parent scope if needed
            openNodeInFlow(n.id);
          });
          item.appendChild(jump);
          grid.appendChild(item);
        });
      });
  }

  function renderModules() {
    const box = document.getElementById("modules");
    box.innerHTML = "";
    ARCH.modules.forEach((m) => {
      const n = node(m.id);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "module-card";
      btn.innerHTML = `<p class="path">${escapeHtml(m.path)}</p><strong>${escapeHtml(
        n.title
      )}</strong><p>${escapeHtml(m.blurb)}</p>`;
      btn.addEventListener("click", () => {
        document.querySelector('.chip[data-view="flow"]').click();
        openNodeInFlow(m.id);
      });
      box.appendChild(btn);
    });
  }

  function openNodeInFlow(id) {
    // Find a sensible trail: if node is a root node, stay at root; else parent that lists it
    const parent = Object.values(ARCH.nodes).find(
      (n) => n.children && n.children.includes(id)
    );
    if (parent && !ARCH.rootNodes.includes(id)) {
      trail = ["root", parent.id];
      transform = { x: 0, y: 0, k: 1 };
      renderGraph(parent.id);
      focusId = id;
      showDetail(id);
      updateBreadcrumb();
      highlightEdges();
      return;
    }
    trail = ["root"];
    transform = { x: 0, y: 0, k: 1 };
    renderGraph("root");
    selectNode(id, { drill: false });
    updateBreadcrumb();
  }

  // Double-click empty canvas to reset selection
  canvasWrap.addEventListener("dblclick", (e) => {
    if (e.target.closest(".node-group")) return;
    if (trail.length > 1) goUp();
  });

  function boot() {
    updateBreadcrumb();
    renderGraph("root");
    // Preselect center engine for first impression
    setTimeout(() => selectNode("repeat_sales", { drill: false }), 400);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
