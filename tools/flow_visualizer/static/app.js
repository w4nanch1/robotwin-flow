const state = {
  file: "",
  camera: "",
  frame: 0,
  pairCount: 0,
  width: 0,
  height: 0,
  playing: false,
  timer: null,
  renderGeneration: 0,
  renderDebounce: null,
  renderController: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const controls = {
  file: $("#fileSelect"),
  camera: $("#cameraSelect"),
  frame: $("#frameSlider"),
  fps: $("#fpsInput"),
  maximum: $("#maximumInput"),
  opacity: $("#opacityInput"),
  arrow: $("#arrowInput"),
  threshold: $("#thresholdInput"),
  hideOccluded: $("#occlusionInput"),
};

function setStatus(text, kind = "ready") {
  const pill = $("#statusPill");
  pill.className = `status-pill ${kind}`;
  pill.querySelector("span").textContent = text;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.remove("hidden");
  clearTimeout(node.timer);
  node.timer = setTimeout(() => node.classList.add("hidden"), 5000);
}

async function getJson(url, signal = undefined) {
  const response = await fetch(url, { cache: "no-store", signal });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function queryBase() {
  const params = new URLSearchParams({
    file: state.file,
    camera: state.camera,
    frame: String(state.frame),
    maximum: controls.maximum.value || "0",
    opacity: controls.opacity.value || "0.7",
    arrow_step: controls.arrow.value || "0",
    occlusion_threshold: String((Number(controls.threshold.value) || 0) / 100),
    hide_occluded: controls.hideOccluded.checked ? "1" : "0",
  });
  return params;
}

async function loadFiles(preserveSelection = true) {
  setStatus("扫描数据", "loading");
  const previous = preserveSelection ? state.file : "";
  try {
    const { files } = await getJson("/api/files");
    controls.file.innerHTML = "";
    if (!files.length) {
      state.file = "";
      $("#workspace").classList.add("hidden");
      $("#emptyState").classList.remove("hidden");
      setStatus("无数据", "error");
      return;
    }
    files.forEach((file) => {
      const option = new Option(`${file.label} · ${formatBytes(file.size)}`, file.id);
      option.title = file.path;
      controls.file.add(option);
    });
    state.file = files.some((file) => file.id === previous) ? previous : files[0].id;
    controls.file.value = state.file;
    await loadMetadata();
  } catch (error) {
    fail(error);
  }
}

async function loadMetadata() {
  stopPlayback();
  setStatus("读取结构", "loading");
  try {
    const metadata = await getJson(`/api/meta?file=${encodeURIComponent(state.file)}`);
    const ready = metadata.cameras.filter((camera) => camera.ready);
    if (!ready.length) {
      const details = metadata.cameras.map((camera) => `${camera.name}: ${(camera.missing || []).join(", ")}`).join("; ");
      throw new Error(`没有可计算的相机流。${details}`);
    }
    controls.camera.innerHTML = "";
    ready.forEach((camera) => controls.camera.add(new Option(camera.name, camera.name)));
    controls.camera.disabled = false;
    state.camera = ready.some((camera) => camera.name === state.camera) ? state.camera : ready[0].name;
    controls.camera.value = state.camera;
    applyCameraMetadata(ready.find((camera) => camera.name === state.camera));
    $("#emptyState").classList.add("hidden");
    $("#workspace").classList.remove("hidden");
    await renderFrame();
  } catch (error) {
    fail(error);
  }
}

function applyCameraMetadata(camera) {
  state.pairCount = camera.pair_count;
  state.width = camera.width;
  state.height = camera.height;
  state.frame = Math.min(state.frame, Math.max(0, state.pairCount - 1));
  controls.frame.max = Math.max(0, state.pairCount - 1);
  controls.frame.value = state.frame;
  $("#lastFrameLabel").textContent = `FRAME ${Math.max(0, state.pairCount - 1)}`;
  updateFrameLabels();
}

async function changeCamera() {
  state.camera = controls.camera.value;
  const metadata = await getJson(`/api/meta?file=${encodeURIComponent(state.file)}`);
  applyCameraMetadata(metadata.cameras.find((camera) => camera.name === state.camera));
  renderFrame();
}

function updateFrameLabels() {
  controls.frame.value = state.frame;
  $("#framePair").textContent = `${state.frame} → ${state.frame + 1}`;
}

async function renderFrame() {
  if (!state.file || !state.camera) return;
  state.renderController?.abort();
  const controller = new AbortController();
  state.renderController = controller;
  const generation = ++state.renderGeneration;
  setStatus("计算 FLOW", "loading");
  updateFrameLabels();
  $$(".viewer").forEach((viewer) => {
    // Keep the previous decoded frame visible while the next one is loading.
    // Hiding it here caused a checkerboard flash even on cache hits.
    if (!viewer.querySelector("img").classList.contains("ready")) {
      viewer.querySelector(".image-loading").classList.remove("hidden");
    }
  });
  const base = queryBase();
  try {
    const results = await Promise.all([
      getJson(`/api/frame-info?${base}`, controller.signal),
      ...$$(".viewer").map((viewer) => loadViewerImage(viewer, base, controller.signal)),
    ]);
    const [summary, ...preparedImages] = results;
    if (generation !== state.renderGeneration) {
      preparedImages.forEach(({ objectUrl }) => URL.revokeObjectURL(objectUrl));
      return;
    }
    // Commit all four decoded views together, leaving the previous frame visible
    // until every panel for the next frame is ready.
    preparedImages.forEach(commitViewerImage);
    updateSummary(summary);
    setStatus(`${state.width} × ${state.height}`);
  } catch (error) {
    if (error.name !== "AbortError" && generation === state.renderGeneration) fail(error);
  }
}

async function loadViewerImage(viewer, base, signal) {
  const image = viewer.querySelector("img");
  const params = new URLSearchParams(base);
  params.set("view", viewer.dataset.view);
  const response = await fetch(`/api/render.png?${params}`, { cache: "no-store", signal });
  if (!response.ok) {
    let message = `无法渲染 ${viewer.dataset.view} 视图`;
    try { message = (await response.json()).error || message; } catch (_) { /* PNG error fallback */ }
    throw new Error(message);
  }
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const loader = new Image();
  loader.src = objectUrl;
  try {
    await loader.decode();
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    throw new Error(`无法解码 ${viewer.dataset.view} 视图`);
  }
  return { viewer, image, objectUrl };
}

function commitViewerImage({ viewer, image, objectUrl }) {
  const previousUrl = image.dataset.objectUrl;
  image.src = objectUrl;
  image.dataset.objectUrl = objectUrl;
  image.classList.add("ready");
  viewer.querySelector(".image-loading").classList.add("hidden");
  if (previousUrl) URL.revokeObjectURL(previousUrl);
}

function updateSummary(summary) {
  $("#meanMetric").textContent = summary.mean_magnitude.toFixed(2);
  $("#p95Metric").textContent = summary.p95_magnitude.toFixed(2);
  $("#validMetric").textContent = `${(summary.valid_ratio * 100).toFixed(1)}%`;
  $("#validDetail").textContent = `${summary.valid_pixels.toLocaleString()} px`;
  $("#occludedMetric").textContent = summary.occluded_pixels.toLocaleString();
  const entities = summary.visible_entities || [];
  $("#entityCount").textContent = `${entities.length} entities`;
  $("#entityList").innerHTML = entities.length ? entities.map((entity) => `
    <div class="entity-row" style="--entity-color:${entityColor(entity.id)}">
      <i></i><code>ID ${entity.id}</code><b title="${escapeHtml(entity.name)}">${escapeHtml(entity.name)}</b><span>${entity.pixels.toLocaleString()} px</span>
    </div>`).join("") : '<div class="empty-list-hint">当前帧只有背景</div>';
}

function entityColor(id) {
  const red = (id * 113 + 7) % 211 + 35;
  const green = (id * 71 + 19) % 211 + 35;
  const blue = (id * 29 + 47) % 211 + 35;
  return `rgb(${red},${green},${blue})`;
}

async function step(delta) {
  state.frame = (state.frame + delta + state.pairCount) % state.pairCount;
  return renderFrame();
}

function togglePlayback() {
  state.playing ? stopPlayback() : startPlayback();
}

function startPlayback() {
  if (!state.pairCount) return;
  state.playing = true;
  $("#playButton").textContent = "❚❚";
  const tick = async () => {
    if (!state.playing) return;
    const started = performance.now();
    await step(1);
    if (!state.playing) return;
    const interval = 1000 / Math.max(1, Number(controls.fps.value) || 8);
    state.timer = setTimeout(tick, Math.max(0, interval - (performance.now() - started)));
  };
  tick();
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.timer);
  $("#playButton").textContent = "▶";
}

function fail(error) {
  console.error(error);
  stopPlayback();
  setStatus("错误", "error");
  toast(error.message || String(error));
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function escapeHtml(text) { return String(text).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]); }

controls.file.addEventListener("change", () => { state.file = controls.file.value; state.frame = 0; loadMetadata(); });
controls.camera.addEventListener("change", changeCamera);
controls.frame.addEventListener("input", () => {
  state.frame = Number(controls.frame.value);
  stopPlayback();
  updateFrameLabels();
  clearTimeout(state.renderDebounce);
  state.renderDebounce = setTimeout(renderFrame, 70);
});
$("#reloadButton").addEventListener("click", () => loadFiles(true));
$("#prevButton").addEventListener("click", () => { stopPlayback(); step(-1); });
$("#nextButton").addEventListener("click", () => { stopPlayback(); step(1); });
$("#playButton").addEventListener("click", togglePlayback);
[controls.maximum, controls.opacity, controls.arrow, controls.threshold, controls.hideOccluded].forEach((control) => control.addEventListener("change", renderFrame));
document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
  if (event.key === " ") { event.preventDefault(); togglePlayback(); }
  if (event.key === "ArrowLeft") { stopPlayback(); step(-1); }
  if (event.key === "ArrowRight") { stopPlayback(); step(1); }
});

loadFiles(false);
