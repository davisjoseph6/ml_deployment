/* Minimal JS for analyze + annotate pages */

function b64ToImage(b64) {
  return "data:image/png;base64," + b64;
}

function renderMetricsTable(metrics) {
  if (!metrics || metrics.length === 0) return "<p>No components detected.</p>";
  let html = `<table><tr>
    <th>Component</th><th>Component area</th><th>Total void area</th><th>Void %</th><th>Max void %</th>
  </tr>`;
  for (const m of metrics) {
    html += `<tr>
      <td>${m.component_id}</td>
      <td>${m.component_area_px}</td>
      <td>${m.total_void_area_px}</td>
      <td>${(m.void_pct * 100).toFixed(4)}%</td>
      <td>${(m.max_void_pct * 100).toFixed(4)}%</td>
    </tr>`;
  }
  html += "</table>";
  return html;
}

/* ---------------- Analyze ---------------- */
const analyzeForm = document.getElementById("analyze-form");
if (analyzeForm) {
  analyzeForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.getElementById("files");
    if (!input.files.length) return;

    const fd = new FormData();
    for (const f of input.files) fd.append("files", f);

    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();

    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    for (const r of data.results) {
      const imgHtml = `<img class="overlay" src="${b64ToImage(r.overlay_png_base64)}">`;
      const metricsHtml = renderMetricsTable(r.metrics);
      const csvLink = `<p><a href="${r.csv_report}">Download CSV</a></p>`;
      resultsDiv.innerHTML += `<div class="card">
        <h3>${r.filename}</h3>
        ${imgHtml}
        ${csvLink}
        ${metricsHtml}
        <p>Unassigned voids: ${r.unassigned_voids_count}</p>
      </div>`;
    }
  });
}

/* ---------------- Annotate ---------------- */
let currentImageId = null;
let canvas = document.getElementById("canvas");
let ctx = canvas ? canvas.getContext("2d") : null;
let overlayImg = null;

let drag = false;
let startX = 0, startY = 0, endX = 0, endY = 0;

function getCanvasXY(e) {
  const rect = canvas.getBoundingClientRect();
  // Map displayed coords -> intrinsic canvas pixels
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function draw() {
  if (!ctx || !overlayImg) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(overlayImg, 0, 0);

  if (drag) {
    const x = Math.min(startX, endX);
    const y = Math.min(startY, endY);
    const w = Math.abs(startX - endX);
    const h = Math.abs(startY - endY);
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
  }
}

async function refineBBox(x1, y1, x2, y2) {
  const target = document.getElementById("target-class").value;
  const fd = new FormData();
  fd.append("image_id", currentImageId);
  fd.append("target_class", target);
  fd.append("x1", Math.round(x1));
  fd.append("y1", Math.round(y1));
  fd.append("x2", Math.round(x2));
  fd.append("y2", Math.round(y2));

  const res = await fetch("/api/refine", { method: "POST", body: fd });
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }

  // update overlay
  overlayImg = new Image();
  overlayImg.onload = () => draw();
  overlayImg.src = b64ToImage(data.overlay_png_base64);

  // update metrics
  const metricsEl = document.getElementById("metrics");
  if (metricsEl) {
    metricsEl.innerHTML = renderMetricsTable(data.metrics) +
      `<p>Unassigned voids: ${data.unassigned_voids_count}</p>`;
  }
}

if (canvas) {
  canvas.addEventListener("mousedown", (e) => {
    drag = true;
    const p = getCanvasXY(e);
    startX = p.x;
    startY = p.y;
    endX = startX;
    endY = startY;
    draw();
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const p = getCanvasXY(e);
    endX = p.x;
    endY = p.y;
    draw();
  });

  canvas.addEventListener("mouseup", async () => {
    drag = false;
    draw();
    const x1 = Math.min(startX, endX);
    const y1 = Math.min(startY, endY);
    const x2 = Math.max(startX, endX);
    const y2 = Math.max(startY, endY);
    if (Math.abs(x2 - x1) < 5 || Math.abs(y2 - y1) < 5) return;
    await refineBBox(x1, y1, x2, y2);
  });
}

const btnPrelabel = document.getElementById("btn-prelabel");
if (btnPrelabel) {
  btnPrelabel.addEventListener("click", async () => {
    const fileInput = document.getElementById("anno-file");
    if (!fileInput.files.length) return;

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);

    const res = await fetch("/api/prelabel", { method: "POST", body: fd });
    const data = await res.json();

    currentImageId = data.image_id;
    document.getElementById("anno-meta").innerText =
      `image_id: ${currentImageId} (${data.filename})`;

    overlayImg = new Image();
    overlayImg.onload = () => {
      canvas.width = overlayImg.width;
      canvas.height = overlayImg.height;
      draw();
    };
    overlayImg.src = b64ToImage(data.overlay_png_base64);

    const metricsEl = document.getElementById("metrics");
    if (metricsEl) metricsEl.innerHTML = "<p>Draw a bbox to refine, then validate.</p>";
  });
}

const btnValidate = document.getElementById("btn-validate");
if (btnValidate) {
  btnValidate.addEventListener("click", async () => {
    if (!currentImageId) return;
    const fd = new FormData();
    fd.append("image_id", currentImageId);
    const res = await fetch("/api/validate", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) alert(data.error);
    else {
      alert("Saved to pending retrain.");
      currentImageId = null;
    }
  });
}

const btnRetrain = document.getElementById("btn-retrain");
if (btnRetrain) {
  btnRetrain.addEventListener("click", async () => {
    const res = await fetch("/api/retrain", { method: "POST" });
    const data = await res.json();
    const el = document.getElementById("train-status");
    if (el) el.innerText = JSON.stringify(data, null, 2);
  });

  // poll status
  setInterval(async () => {
    const res = await fetch("/api/retrain/status");
    const data = await res.json();
    const el = document.getElementById("train-status");
    if (el) el.innerText = JSON.stringify(data, null, 2);
  }, 1500);
}

