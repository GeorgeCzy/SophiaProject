const STORAGE_KEY = "sophia-public-rating-session";

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
};

const elements = {
  setupScreen: document.querySelector("#setupScreen"),
  ratingApp: document.querySelector("#ratingApp"),
  participantInput: document.querySelector("#participantInput"),
  sessionInput: document.querySelector("#sessionInput"),
  randomizeInput: document.querySelector("#randomizeInput"),
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

function getOrderedClips() {
  const byId = new Map(state.clips.map((clip) => [clip.id, clip]));
  return state.order.map((id) => byId.get(id)).filter(Boolean);
}

function getCurrentClip() {
  return getOrderedClips()[state.currentIndex] || state.clips[0];
}

function isRatingComplete(rating) {
  return Boolean(rating && criteria.every((criterion) => rating[criterion.key]));
}

function completedCount() {
  return state.clips.filter((clip) => isRatingComplete(state.ratings[clip.id])).length;
}

function csvEscape(value) {
  const text = value === undefined || value === null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function saveSession() {
  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      participantId: state.participantId,
      sessionLabel: state.sessionLabel,
      randomizeOrder: state.randomizeOrder,
      started: state.started,
      currentIndex: state.currentIndex,
      order: state.order,
      ratings: state.ratings,
      startedAt: state.startedAt,
    }),
  );
}

function restoreSession() {
  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (!saved) return;
  try {
    const parsed = JSON.parse(saved);
    state.participantId = parsed.participantId || "";
    state.sessionLabel = parsed.sessionLabel || "";
    state.randomizeOrder = parsed.randomizeOrder ?? true;
    state.started = parsed.started || false;
    state.currentIndex = parsed.currentIndex || 0;
    state.order = Array.isArray(parsed.order) ? parsed.order : [];
    state.ratings = parsed.ratings || {};
    state.startedAt = parsed.startedAt || "";
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
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
    for (let value = 1; value <= 7; value += 1) {
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
    }
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
    dot.textContent = String(clip.id);
    dot.setAttribute("aria-label", `Open video ${clip.id}`);
    dot.addEventListener("click", () => {
      state.currentIndex = index;
      saveSession();
      render();
    });
    elements.clipDots.appendChild(dot);
  });
}

function render() {
  elements.videoCountLabel.textContent = `${state.clips.length} videos`;
  elements.participantInput.value = state.participantId;
  elements.sessionInput.value = state.sessionLabel;
  elements.randomizeInput.checked = state.randomizeOrder;

  elements.setupScreen.hidden = state.started;
  elements.ratingApp.hidden = !state.started;
  if (!state.started) return;

  const orderedClips = getOrderedClips();
  const clip = getCurrentClip();
  const rating = state.ratings[clip.id] || {};
  const complete = completedCount();
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
      label.classList.toggle("selected", rating[key] === value);
      label.querySelector("input").checked = rating[key] === value;
    });
  });

  elements.previousButton.disabled = state.currentIndex === 0;
  elements.exportButton.disabled = complete === 0;
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
  const id = elements.participantInput.value.trim() || `P-${String(Date.now()).slice(-6)}`;
  state.participantId = id;
  state.sessionLabel = elements.sessionInput.value.trim();
  state.randomizeOrder = elements.randomizeInput.checked;
  state.order = state.randomizeOrder
    ? seededShuffle(
        state.clips.map((clip) => clip.id),
        id,
      )
    : state.clips.map((clip) => clip.id);
  state.currentIndex = 0;
  state.started = true;
  state.startedAt = new Date().toISOString();
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

function exportCsv() {
  const header = [
    "participant_id",
    "session_label",
    "started_at",
    "exported_at",
    "presentation_order",
    "clip_id",
    "clip_title",
    "condition",
    "video_src",
    ...criteria.map((criterion) => criterion.key),
    "playback_issue",
    "comment",
    "completed_at",
  ];
  const exportedAt = new Date().toISOString();
  const lines = [
    header.map(csvEscape).join(","),
    ...getOrderedClips().map((clip, index) => {
      const rating = state.ratings[clip.id] || {};
      return [
        state.participantId,
        state.sessionLabel,
        state.startedAt,
        exportedAt,
        index + 1,
        clip.id,
        clip.title,
        clip.condition,
        clip.src,
        ...criteria.map((criterion) => rating[criterion.key]),
        rating.playbackIssue || false,
        rating.comment,
        rating.completedAt,
      ]
        .map(csvEscape)
        .join(",");
    }),
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

function resetSession() {
  const confirmed = window.confirm("Clear this browser's current rating session?");
  if (!confirmed) return;
  window.localStorage.removeItem(STORAGE_KEY);
  state.participantId = "";
  state.sessionLabel = "";
  state.randomizeOrder = true;
  state.started = false;
  state.currentIndex = 0;
  state.order = state.clips.map((clip) => clip.id);
  state.ratings = {};
  state.videoErrors = {};
  state.startedAt = "";
  render();
}

async function loadClips() {
  try {
    const response = await fetch(resolveAsset("/video-manifest.json"), { cache: "no-store" });
    if (!response.ok) throw new Error(`Manifest request failed: ${response.status}`);
    state.clips = await response.json();
  } catch {
    state.clips = fallbackClips();
  }
  if (!state.order.length) {
    state.order = state.clips.map((clip) => clip.id);
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
  elements.randomizeInput.addEventListener("change", (event) => {
    state.randomizeOrder = event.target.checked;
    saveSession();
  });
  elements.startButton.addEventListener("click", startSession);
  elements.previousButton.addEventListener("click", () => {
    state.currentIndex = Math.max(state.currentIndex - 1, 0);
    saveSession();
    render();
  });
  elements.resetButton.addEventListener("click", resetSession);
  elements.exportButton.addEventListener("click", exportCsv);
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
  restoreSession();
  await loadClips();
  bindEvents();
  render();
}

init();
