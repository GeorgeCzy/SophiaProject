const DEFAULT_EXPERIMENT_ID = "complete";
const STORAGE_KEY_PREFIX = "sophia-public-rating-session-v4";
const ratingValues = [1, 2, 3, 4, 5];
const config = {
  submitEndpoint: "",
  submitMode: "no-cors",
  ...(window.SOPHIA_RATING_CONFIG || {}),
};
const fallbackExperiment = {
  id: DEFAULT_EXPERIMENT_ID,
  publicLabel: "Sophia Study A",
  condition: "complete",
  stimulusSet: "daily-dialogue-40",
  manifest: "/manifests/complete.json",
  enabled: true,
};

const criteria = [
  {
    key: "verbalAppropriateness",
    label: "Verbal appropriateness",
    statement: "The robot's verbal response was appropriate for the user's situation.",
  },
  {
    key: "verbalHelpfulness",
    label: "Verbal helpfulness",
    statement: "The robot's verbal response was clear and helpful.",
  },
  {
    key: "motionNaturalness",
    label: "Motion naturalness",
    statement: "The robot's body/hand motion looked natural and smooth.",
  },
  {
    key: "motionExpressiveness",
    label: "Motion expressiveness",
    statement: "The robot's motion made the response more expressive.",
  },
  {
    key: "speechMotionCoordination",
    label: "Speech-motion coordination",
    statement: "The robot's speech and motion matched each other well.",
  },
  {
    key: "nonExcessiveness",
    label: "Non-excessiveness",
    statement: "The robot's motion was not excessive or distracting.",
  },
  {
    key: "overallQuality",
    label: "Overall quality",
    statement: "Overall, the robot behaved well in this interaction.",
  },
];

const state = {
  experiments: [],
  experiment: fallbackExperiment,
  experimentId: DEFAULT_EXPERIMENT_ID,
  experimentLabel: fallbackExperiment.publicLabel,
  condition: fallbackExperiment.condition,
  stimulusSet: fallbackExperiment.stimulusSet,
  clips: [],
  participantId: "",
  sessionLabel: "",
  randomizeOrder: true,
  started: false,
  currentIndex: 0,
  order: [],
  ratings: {},
  videoErrors: {},
  startedAt: "",
  orderSeed: "",
  resumeAvailable: false,
  submitting: false,
  submitStatus: "",
  loadError: "",
};

const elements = {
  setupScreen: document.querySelector("#setupScreen"),
  ratingApp: document.querySelector("#ratingApp"),
  studyLabels: document.querySelectorAll("[data-study-label]"),
  studySwitcher: document.querySelector("#studySwitcher"),
  participantInput: document.querySelector("#participantInput"),
  sessionInput: document.querySelector("#sessionInput"),
  randomizeInput: document.querySelector("#randomizeInput"),
  loadError: document.querySelector("#loadError"),
  resumePanel: document.querySelector("#resumePanel"),
  resumeSummary: document.querySelector("#resumeSummary"),
  resumeButton: document.querySelector("#resumeButton"),
  startButton: document.querySelector("#startButton"),
  videoCountLabel: document.querySelector("#videoCountLabel"),
  criterionPreview: document.querySelector("#criterionPreview"),
  participantLabel: document.querySelector("#participantLabel"),
  completeLabel: document.querySelector("#completeLabel"),
  progressFill: document.querySelector("#progressFill"),
  clipDots: document.querySelector("#clipDots"),
  clipPosition: document.querySelector("#clipPosition"),
  clipTitle: document.querySelector("#clipTitle"),
  clipVideo: document.querySelector("#clipVideo"),
  videoWarning: document.querySelector("#videoWarning"),
  clipSituation: document.querySelector("#clipSituation"),
  clipUser: document.querySelector("#clipUser"),
  ratingList: document.querySelector("#ratingList"),
  commentInput: document.querySelector("#commentInput"),
  playbackIssueInput: document.querySelector("#playbackIssueInput"),
  previousButton: document.querySelector("#previousButton"),
  resetButton: document.querySelector("#resetButton"),
  exportButton: document.querySelector("#exportButton"),
  submitButton: document.querySelector("#submitButton"),
  submitStatus: document.querySelector("#submitStatus"),
  saveButton: document.querySelector("#saveButton"),
};

function getBasePath() {
  const path = window.location.pathname;
  if (path.includes("/SophiaProject/")) return "/SophiaProject/";
  return path.endsWith("/") ? path : path.replace(/[^/]*$/, "");
}

function resolveAsset(src) {
  if (!src) return "";
  if (/^https?:\/\//i.test(src)) return src;
  if (src.startsWith("/")) return `${getBasePath()}${src.slice(1)}`;
  return src;
}

function getRequestedExperimentId() {
  const params = new URLSearchParams(window.location.search);
  return (params.get("experiment") || DEFAULT_EXPERIMENT_ID).trim() || DEFAULT_EXPERIMENT_ID;
}

function switchExperiment(experimentId) {
  if (!experimentId || experimentId === state.experimentId) return;
  const url = new URL(window.location.href);
  url.searchParams.set("experiment", experimentId);
  window.location.assign(url.toString());
}

function getStorageKey() {
  return `${STORAGE_KEY_PREFIX}:${state.experimentId}`;
}

function normalizeExperiment(experiment) {
  return {
    ...fallbackExperiment,
    ...experiment,
    publicLabel: experiment.publicLabel || experiment.label || fallbackExperiment.publicLabel,
    condition: experiment.condition || experiment.id || fallbackExperiment.condition,
    stimulusSet: experiment.stimulusSet || fallbackExperiment.stimulusSet,
    enabled: experiment.enabled !== false,
  };
}

function fallbackClips() {
  return Array.from({ length: 40 }, (_, index) => {
    const id = index + 1;
    return {
      id,
      title: `Video ${String(id).padStart(2, "0")}`,
      situation: "Daily social robot conversation",
      user: "User utterance for this clip",
      src: `videos/video-${String(id).padStart(2, "0")}.mp4`,
    };
  });
}

function seededShuffle(ids, seedText) {
  let seed = 2166136261;
  for (const char of seedText) {
    seed ^= char.charCodeAt(0);
    seed = Math.imul(seed, 16777619);
  }

  const shuffled = [...ids];
  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    seed += 0x6d2b79f5;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const random = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    const j = Math.floor(random * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function makePresentationOrder(ids, seedText) {
  const shuffled = seededShuffle(ids, seedText || String(Date.now()));
  const unchanged = shuffled.every((id, index) => id === ids[index]);
  if (unchanged && shuffled.length > 1) {
    return [...shuffled.slice(1), shuffled[0]];
  }
  return shuffled;
}

function getOrderedClips() {
  const byId = new Map(state.clips.map((clip) => [clip.id, clip]));
  return state.order.map((id) => byId.get(id)).filter(Boolean);
}

function getCurrentClip() {
  return getOrderedClips()[state.currentIndex] || state.clips[0];
}

function isRatingComplete(rating) {
  return Boolean(rating && criteria.every((criterion) => ratingValues.includes(rating[criterion.key])));
}

function hasAnyRating(rating) {
  return Boolean(
    rating &&
      (criteria.some((criterion) => ratingValues.includes(rating[criterion.key])) ||
        rating.playbackIssue ||
        rating.comment?.trim()),
  );
}

function getRatingValue(rating, key) {
  const value = rating?.[key];
  return ratingValues.includes(value) ? value : "";
}

function completedCount() {
  return state.clips.filter((clip) => isRatingComplete(state.ratings[clip.id])).length;
}

function attemptedCount() {
  return state.clips.filter((clip) => hasAnyRating(state.ratings[clip.id])).length;
}

function csvEscape(value) {
  const text = value === undefined || value === null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function saveSession() {
  window.localStorage.setItem(
    getStorageKey(),
    JSON.stringify({
      experimentId: state.experimentId,
      experimentLabel: state.experimentLabel,
      condition: state.condition,
      stimulusSet: state.stimulusSet,
      participantId: state.participantId,
      sessionLabel: state.sessionLabel,
      randomizeOrder: state.randomizeOrder,
      started: state.started,
      currentIndex: state.currentIndex,
      order: state.order,
      orderSeed: state.orderSeed,
      ratings: state.ratings,
      startedAt: state.startedAt,
    }),
  );
}

function restoreSession() {
  const saved = window.localStorage.getItem(getStorageKey());
  if (!saved) return;
  try {
    const parsed = JSON.parse(saved);
    if (parsed.experimentId && parsed.experimentId !== state.experimentId) return;
    state.participantId = parsed.participantId || "";
    state.sessionLabel = parsed.sessionLabel || "";
    state.randomizeOrder = parsed.randomizeOrder ?? true;
    state.started = false;
    state.currentIndex = parsed.currentIndex || 0;
    state.order = Array.isArray(parsed.order) ? parsed.order : [];
    state.orderSeed = parsed.orderSeed || "";
    state.ratings = parsed.ratings || {};
    state.startedAt = parsed.startedAt || "";
    state.resumeAvailable = Boolean(
      parsed.started || parsed.startedAt || Object.keys(state.ratings).length,
    );
  } catch {
    window.localStorage.removeItem(getStorageKey());
  }
}

function renderCriterionPreview() {
  elements.criterionPreview.innerHTML = "";
  criteria.forEach((criterion) => {
    const item = document.createElement("span");
    item.textContent = criterion.label;
    elements.criterionPreview.appendChild(item);
  });
}

function renderExperimentSwitcher() {
  if (!elements.studySwitcher) return;
  const experiments = state.experiments.length ? state.experiments : [state.experiment];
  elements.studySwitcher.innerHTML = "";
  experiments.forEach((experiment) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "study-switch-option";
    button.textContent = experiment.publicLabel || experiment.id;
    button.classList.toggle("active", experiment.id === state.experimentId);
    button.disabled = experiment.enabled === false;
    button.setAttribute("aria-pressed", experiment.id === state.experimentId ? "true" : "false");
    button.addEventListener("click", () => switchExperiment(experiment.id));
    elements.studySwitcher.appendChild(button);
  });
}

function renderRatingControls() {
  elements.ratingList.innerHTML = "";
  criteria.forEach((criterion) => {
    const field = document.createElement("fieldset");
    field.className = "rating-card";
    field.innerHTML = `
      <legend>
        <span>${criterion.label}</span>
        <small>${criterion.statement}</small>
      </legend>
      <div class="rating-scale" role="radiogroup" aria-label="${criterion.label}"></div>
    `;

    const scale = field.querySelector(".rating-scale");
    ratingValues.forEach((value) => {
      const label = document.createElement("label");
      label.innerHTML = `
        <input type="radio" name="${criterion.key}" value="${value}" />
        ${value}
      `;
      label.addEventListener("click", () => {
        const clip = getCurrentClip();
        state.ratings[clip.id] = {
          ...(state.ratings[clip.id] || {}),
          [criterion.key]: value,
        };
        saveSession();
        render();
      });
      scale.appendChild(label);
    });
    elements.ratingList.appendChild(field);
  });
}

function renderDots() {
  elements.clipDots.innerHTML = "";
  getOrderedClips().forEach((clip, index) => {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "clip-dot";
    if (index === state.currentIndex) dot.classList.add("current");
    if (isRatingComplete(state.ratings[clip.id])) dot.classList.add("complete");
    dot.textContent = String(index + 1);
    dot.setAttribute("aria-label", `Open video ${index + 1}`);
    dot.addEventListener("click", () => {
      state.currentIndex = index;
      saveSession();
      render();
    });
    elements.clipDots.appendChild(dot);
  });
}

function render() {
  const complete = completedCount();
  const attempted = attemptedCount();
  const resumeVisible = state.resumeAvailable && (Boolean(state.startedAt) || attempted > 0);

  elements.studyLabels.forEach((label) => {
    label.textContent = state.experimentLabel;
  });
  elements.videoCountLabel.textContent = state.loadError
    ? "Study version unavailable"
    : `${state.clips.length} videos`;
  elements.startButton.disabled = Boolean(state.loadError) || state.clips.length === 0;
  elements.participantInput.value = state.participantId;
  elements.sessionInput.value = state.sessionLabel;
  elements.sessionInput.placeholder = state.experimentId;
  renderExperimentSwitcher();
  if (elements.randomizeInput) {
    elements.randomizeInput.checked = true;
  }
  if (elements.loadError) {
    elements.loadError.hidden = !state.loadError;
    elements.loadError.textContent = state.loadError;
  }
  if (elements.resumePanel) {
    elements.resumePanel.hidden = !resumeVisible;
    elements.resumeSummary.textContent = `${state.participantId || "Saved participant"} · ${complete}/${state.clips.length} complete · ${attempted} started`;
    elements.resumeButton.disabled = !resumeVisible;
  }

  elements.setupScreen.hidden = state.started;
  elements.ratingApp.hidden = !state.started;
  if (!state.started) return;

  const orderedClips = getOrderedClips();
  const clip = getCurrentClip();
  const rating = state.ratings[clip.id] || {};
  const progress = state.clips.length ? Math.round((complete / state.clips.length) * 100) : 0;

  elements.participantLabel.textContent = state.participantId || "Participant";
  elements.completeLabel.textContent = `${complete}/${state.clips.length} complete`;
  elements.progressFill.style.width = `${progress}%`;
  elements.clipPosition.textContent = `Video ${state.currentIndex + 1} of ${orderedClips.length}`;
  elements.clipTitle.textContent = clip.title;
  elements.clipSituation.textContent = clip.situation;
  elements.clipUser.textContent = clip.user;

  const videoSrc = resolveAsset(clip.src);
  if (elements.clipVideo.getAttribute("src") !== videoSrc) {
    elements.clipVideo.setAttribute("src", videoSrc);
    elements.clipVideo.load();
  }

  elements.videoWarning.hidden = !state.videoErrors[clip.id];
  elements.videoWarning.textContent = `Video file not found: ${clip.src}`;
  elements.commentInput.value = rating.comment || "";
  elements.playbackIssueInput.checked = Boolean(rating.playbackIssue);

  document.querySelectorAll(".rating-card").forEach((card) => {
    const key = card.querySelector("input")?.name;
    card.querySelectorAll("label").forEach((label) => {
      const value = Number(label.querySelector("input").value);
      label.classList.toggle("selected", getRatingValue(rating, key) === value);
      label.querySelector("input").checked = getRatingValue(rating, key) === value;
    });
  });

  elements.previousButton.disabled = state.currentIndex === 0;
  elements.exportButton.disabled = attempted === 0;
  elements.exportButton.textContent =
    config.submitEndpoint
      ? "Download Backup CSV"
      : complete === state.clips.length
        ? "Export CSV"
        : "End & Export CSV";
  if (elements.submitButton) {
    elements.submitButton.hidden = !config.submitEndpoint;
    elements.submitButton.disabled = !config.submitEndpoint || attempted === 0 || state.submitting;
    elements.submitButton.textContent = state.submitting
      ? "Submitting..."
      : complete === state.clips.length
        ? "Submit Online"
        : "End & Submit Online";
  }
  if (elements.submitStatus) {
    elements.submitStatus.textContent = config.submitEndpoint ? state.submitStatus : "";
  }
  elements.saveButton.disabled = !isRatingComplete(rating);
  elements.saveButton.textContent =
    state.currentIndex === orderedClips.length - 1
      ? complete === state.clips.length
        ? "Finished"
        : "Save"
      : "Save & Next";
  renderDots();
}

function startSession() {
  if (state.loadError || !state.clips.length) return;
  const id = elements.participantInput.value.trim() || `P-${String(Date.now()).slice(-6)}`;
  state.participantId = id;
  state.sessionLabel = elements.sessionInput.value.trim();
  state.randomizeOrder = true;
  state.ratings = {};
  state.videoErrors = {};
  state.resumeAvailable = false;
  state.submitStatus = "";
  state.orderSeed = `${state.experimentId}:${id}`;
  state.order = makePresentationOrder(
    state.clips.map((clip) => clip.id),
    state.orderSeed,
  );
  state.currentIndex = 0;
  state.started = true;
  state.startedAt = new Date().toISOString();
  saveSession();
  render();
}

function resumeSession() {
  if (!state.resumeAvailable) return;
  if (!state.order.length) {
    state.order = state.clips.map((clip) => clip.id);
  }
  if (!state.startedAt) {
    state.startedAt = new Date().toISOString();
  }
  state.started = true;
  state.submitStatus = "";
  saveSession();
  render();
}

function completeCurrent() {
  const orderedClips = getOrderedClips();
  const clip = getCurrentClip();
  const rating = state.ratings[clip.id] || {};
  if (!isRatingComplete(rating)) return;
  state.ratings[clip.id] = {
    ...rating,
    completedAt: new Date().toISOString(),
  };
  state.currentIndex = Math.min(state.currentIndex + 1, orderedClips.length - 1);
  saveSession();
  render();
}

function getCsvHeader() {
  return [
    "participant_id",
    "session_label",
    "started_at",
    "exported_at",
    "presentation_order",
    "clip_id",
    "clip_title",
    "condition",
    "video_src",
    "is_complete",
    ...criteria.map((criterion) => criterion.key),
    "playback_issue",
    "comment",
    "completed_at",
    "experiment_id",
    "experiment_label",
    "stimulus_set",
    "order_seed",
    "dialogue_id",
  ];
}

function buildExportRows(exportedAt = new Date().toISOString()) {
  const orderedClips = getOrderedClips();
  const exportableClips = orderedClips.filter((clip) => hasAnyRating(state.ratings[clip.id]));
  return exportableClips.map((clip) => {
      const rating = state.ratings[clip.id] || {};
      const orderedIndex = orderedClips.findIndex((orderedClip) => orderedClip.id === clip.id);
      return [
        state.participantId,
        state.sessionLabel,
        state.startedAt,
        exportedAt,
        orderedIndex + 1,
        clip.id,
        clip.title,
        clip.condition,
        clip.src,
        isRatingComplete(rating),
        ...criteria.map((criterion) => getRatingValue(rating, criterion.key)),
        rating.playbackIssue || false,
        rating.comment,
        rating.completedAt,
        state.experimentId,
        state.experimentLabel,
        state.stimulusSet,
        state.orderSeed,
        clip.id,
      ];
    });
}

function rowsToObjects(rows) {
  const header = getCsvHeader();
  return rows.map((row) =>
    Object.fromEntries(header.map((key, index) => [key, row[index] ?? ""])),
  );
}

function exportCsv() {
  const rows = buildExportRows();
  if (!rows.length) return;

  const lines = [
    getCsvHeader().map(csvEscape).join(","),
    ...rows.map((row) => row.map(csvEscape).join(",")),
  ];

  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.participantId || "participant"}_sophia_ratings.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function submitResponses() {
  const submittedAt = new Date().toISOString();
  const rows = buildExportRows(submittedAt);
  if (!rows.length || !config.submitEndpoint) return;

  state.submitting = true;
  state.submitStatus = "Submitting...";
  render();

  try {
    const noCors = config.submitMode !== "cors";
    const response = await fetch(config.submitEndpoint, {
      method: "POST",
      mode: noCors ? "no-cors" : "cors",
      headers: {
        "Content-Type": noCors ? "text/plain;charset=utf-8" : "application/json",
      },
      body: JSON.stringify({
        participantId: state.participantId,
        sessionLabel: state.sessionLabel,
        startedAt: state.startedAt,
        submittedAt,
        completedCount: completedCount(),
        attemptedCount: attemptedCount(),
        totalClips: state.clips.length,
        rows: rowsToObjects(rows),
      }),
    });

    if (!noCors && !response.ok) {
      throw new Error(`Submit failed with ${response.status}`);
    }
    state.submitStatus = `Submitted ${rows.length} row${rows.length === 1 ? "" : "s"}.`;
  } catch {
    state.submitStatus = "Submit failed. Export CSV as backup.";
  } finally {
    state.submitting = false;
    saveSession();
    render();
  }
}

function resetSession() {
  const confirmed = window.confirm("Clear this browser's current rating session?");
  if (!confirmed) return;
  window.localStorage.removeItem(getStorageKey());
  state.participantId = "";
  state.sessionLabel = "";
  state.randomizeOrder = true;
  state.started = false;
  state.currentIndex = 0;
  state.order = state.clips.map((clip) => clip.id);
  state.ratings = {};
  state.videoErrors = {};
  state.startedAt = "";
  state.orderSeed = "";
  state.resumeAvailable = false;
  state.submitStatus = "";
  render();
}

async function loadExperiment() {
  const requestedId = getRequestedExperimentId();
  state.experimentId = requestedId;
  state.loadError = "";

  try {
    const response = await fetch(resolveAsset("/experiments.json"), { cache: "no-store" });
    if (!response.ok) throw new Error(`Experiment request failed: ${response.status}`);
    const experiments = await response.json();
    state.experiments = experiments.map(normalizeExperiment);
  } catch {
    state.experiments = [fallbackExperiment];
  }

  const selected = state.experiments.find((experiment) => experiment.id === requestedId);
  if (!selected) {
    state.experiment = normalizeExperiment({ id: requestedId, enabled: false });
    state.experimentLabel = "Sophia Study";
    state.condition = requestedId;
    state.stimulusSet = fallbackExperiment.stimulusSet;
    state.loadError = "This study version is not available. Please check the study link.";
    return;
  }

  state.experiment = normalizeExperiment(selected);
  state.experimentLabel = state.experiment.publicLabel;
  state.condition = state.experiment.condition;
  state.stimulusSet = state.experiment.stimulusSet;
  if (!state.experiment.enabled) {
    state.loadError =
      state.experiment.message || "This study version is not ready yet. Please check the study link.";
  }
}

async function loadClips() {
  if (state.loadError) return;

  try {
    const response = await fetch(resolveAsset(state.experiment.manifest), { cache: "no-store" });
    if (!response.ok) throw new Error(`Manifest request failed: ${response.status}`);
    const clips = await response.json();
    state.clips = clips.map((clip) => ({
      ...clip,
      condition: state.condition,
      stimulusSet: state.stimulusSet,
    }));
  } catch {
    state.clips = [];
    state.loadError = "This study version could not load. Please check the study link.";
  }
  const clipIds = state.clips.map((clip) => clip.id);
  const clipIdSet = new Set(clipIds);
  const orderStillValid =
    state.order.length === clipIds.length && state.order.every((id) => clipIdSet.has(id));
  if (!orderStillValid) {
    state.order = clipIds;
  }
}

function bindEvents() {
  elements.participantInput.addEventListener("input", (event) => {
    state.participantId = event.target.value;
    saveSession();
  });
  elements.sessionInput.addEventListener("input", (event) => {
    state.sessionLabel = event.target.value;
    saveSession();
  });
  elements.randomizeInput?.addEventListener("change", () => {
    state.randomizeOrder = true;
    saveSession();
  });
  elements.startButton.addEventListener("click", startSession);
  elements.resumeButton?.addEventListener("click", resumeSession);
  elements.previousButton.addEventListener("click", () => {
    state.currentIndex = Math.max(state.currentIndex - 1, 0);
    saveSession();
    render();
  });
  elements.resetButton.addEventListener("click", resetSession);
  elements.exportButton.addEventListener("click", exportCsv);
  elements.submitButton?.addEventListener("click", submitResponses);
  elements.saveButton.addEventListener("click", completeCurrent);
  elements.commentInput.addEventListener("input", (event) => {
    const clip = getCurrentClip();
    state.ratings[clip.id] = {
      ...(state.ratings[clip.id] || {}),
      comment: event.target.value,
    };
    saveSession();
  });
  elements.playbackIssueInput.addEventListener("change", (event) => {
    const clip = getCurrentClip();
    state.ratings[clip.id] = {
      ...(state.ratings[clip.id] || {}),
      playbackIssue: event.target.checked,
    };
    saveSession();
  });
  elements.clipVideo.addEventListener("error", () => {
    const clip = getCurrentClip();
    state.videoErrors[clip.id] = true;
    render();
  });
}

async function init() {
  renderCriterionPreview();
  renderRatingControls();
  await loadExperiment();
  restoreSession();
  await loadClips();
  bindEvents();
  render();
}

init();
