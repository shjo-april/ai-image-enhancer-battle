// AI Image Quality Boost Battle
// Copyright 2026 Sanghyun Jo

const API = window.location.origin;
let sessionId = null;
let originalTags = [];
let originalRating = {};
let originalTop5 = new Set();

const $landing     = document.getElementById("landing-section");
const $workspace   = document.getElementById("workspace");
const $dropZone    = document.getElementById("drop-zone");
const $fileInput   = document.getElementById("file-input");
const $creator     = document.getElementById("creator-name");
const $startBanner = document.getElementById("start-banner");
const $imgOrig     = document.getElementById("img-original");
const $imgCur      = document.getElementById("img-current");
const $turnLabel   = document.getElementById("current-turn-label");
const $scoreOrig   = document.getElementById("score-original");
const $scoreCur    = document.getElementById("score-current");
const $scoreDelta  = document.getElementById("score-delta");
const $bestScore   = document.getElementById("best-score");
const $bestGap     = document.getElementById("best-gap");
const $infoPanel   = document.getElementById("info-panel");
const $prompt      = document.getElementById("edit-prompt");
const $editBtn     = document.getElementById("edit-btn");
const $resetBtn    = document.getElementById("reset-btn");
const $loading     = document.getElementById("loading");
const $loadTitle   = document.getElementById("loading-title");
const $loadPrompt  = document.getElementById("loading-prompt");
const $gpuStatus   = document.getElementById("gpu-status");
const $lbRow       = document.getElementById("lb-row");
const $lbRowWs     = document.getElementById("lb-row-ws");

function showLoading(title, detail) {
    $loadTitle.textContent = title || "Processing...";
    $loadPrompt.textContent = detail || "";
    $loading.classList.add("active");
}
function hideLoading() { $loading.classList.remove("active"); }

// Name gate
$creator.addEventListener("input", () => {
    const hasName = $creator.value.trim().length > 0;
    $dropZone.classList.toggle("disabled", !hasName);
    $dropZone.querySelector("p").textContent = hasName
        ? "Drag & drop your image here, or click to browse"
        : "Enter your name above to unlock image upload";
    $fileInput.disabled = !hasName;
});

// Upload
$dropZone.addEventListener("click", () => { if (!$fileInput.disabled) $fileInput.click(); });
$dropZone.addEventListener("dragover", e => { e.preventDefault(); if (!$fileInput.disabled) $dropZone.classList.add("dragover"); });
$dropZone.addEventListener("dragleave", () => $dropZone.classList.remove("dragover"));
$dropZone.addEventListener("drop", e => {
    e.preventDefault(); $dropZone.classList.remove("dragover");
    if (!$fileInput.disabled && e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
});
$fileInput.addEventListener("change", () => { if ($fileInput.files.length) upload($fileInput.files[0]); });

async function upload(file) {
    const name = $creator.value.trim();
    if (!name) { alert("Please enter your name first."); return; }
    showLoading("Analyzing your image...", "Scoring visual quality & detecting tags");
    const form = new FormData();
    form.append("image", file);
    form.append("creator_name", name);
    try {
        const res = await fetch(`${API}/api/session/create`, { method: "POST", body: form });
        if (!res.ok) { const err = await res.json(); alert(err.detail || "Upload failed."); return; }
        const data = await res.json();
        sessionId = data.session_id;
        $landing.style.display = "none";
        $workspace.classList.add("active");
        $startBanner.innerHTML = `<span>@${data.creator}</span>, start editing to boost your score!`;
        $startBanner.style.display = "block";
        $imgOrig.src = `data:image/png;base64,${data.original_image}`;
        $imgCur.src = $imgOrig.src;
        $scoreOrig.textContent = data.score;
        $scoreCur.textContent = data.score;
        $scoreDelta.textContent = "";
        $turnLabel.textContent = "-";
        $bestScore.textContent = data.score;
        $bestGap.textContent = "";
        originalTags = data.tags || [];
        originalRating = data.rating || {};
        originalTop5 = new Set((data.tags || []).slice(0, 5).map(t => t.name || t));
        renderInfoPanel(originalTags, originalRating, null, null, null, originalTop5);
        $editBtn.disabled = false;
    } catch (e) {
        alert("Upload error: " + e.message);
    } finally {
        hideLoading();
    }
}

// Edit
$editBtn.addEventListener("click", () => doEdit());
$prompt.addEventListener("keydown", e => { if (e.key === "Enter" && !$editBtn.disabled) doEdit(); });

async function doEdit() {
    const prompt = $prompt.value.trim();
    if (!prompt || !sessionId) return;
    $startBanner.style.display = "none";
    $editBtn.disabled = true;
    showLoading("Editing in progress...", `"${prompt}"`);
    try {
        const form = new FormData();
        form.append("editing_prompt", prompt);
        const res = await fetch(`${API}/api/session/${sessionId}/edit`, { method: "POST", body: form });
        const data = await res.json();
        if (data.error) { alert(data.message || data.error); $editBtn.disabled = false; return; }

        $imgCur.src = `data:image/png;base64,${data.image}`;
        $scoreCur.textContent = data.score;
        $turnLabel.textContent = `Round ${data.turn_id}`;
        const delta = data.delta;
        $scoreDelta.textContent = delta >= 0 ? `(+${delta.toFixed(2)})` : `(${delta.toFixed(2)})`;
        $scoreDelta.className = `delta ${delta >= 0 ? "up" : "down"}`;
        $bestScore.textContent = data.best_score;
        if (data.best_gap > 0) {
            $bestGap.textContent = `+${data.best_gap.toFixed(2)}`;
            $bestGap.className = "best-gap up";
        }

        // Update original_top5 from server if provided
        if (data.original_top5) originalTop5 = new Set(data.original_top5);

        renderInfoPanel(
            originalTags, originalRating,
            data.edited_tags || [], data.edited_rating || {},
            data.tags_valid, originalTop5,
        );
        $prompt.value = "";
        $editBtn.disabled = false;
        refreshLeaderboard();
    } catch (e) {
        alert("Edit error: " + e.message);
        $editBtn.disabled = false;
    } finally {
        hideLoading();
    }
}

// Reset
$resetBtn.addEventListener("click", async () => {
    if (sessionId) { try { await fetch(`${API}/api/session/${sessionId}/reset`, { method: "POST" }); } catch {} }
    sessionId = null;
    originalTags = []; originalRating = {}; originalTop5 = new Set();
    $workspace.classList.remove("active");
    $landing.style.display = "";
    $fileInput.value = "";
    $editBtn.disabled = true;
    refreshLeaderboard();
});

// ─── Info Panel ───

function renderTagList(tags, highlightSet, mode) {
    // mode: "original" highlights top-5, "edited" marks match/miss against highlightSet
    if (!Array.isArray(tags) || !tags.length) return '<span class="tag muted">none</span>';
    return tags.map((t, i) => {
        const name = (t && typeof t === "object") ? t.name : String(t);
        const score = (t && typeof t === "object") ? t.score : null;
        const scoreHtml = score !== null ? ` <span class="tag-score">${score}</span>` : "";
        let cls = "tag";
        let icon = "";
        if (mode === "original" && i < 5) {
            cls += " tag-top5";
        } else if (mode === "edited" && highlightSet && highlightSet.size > 0) {
            if (highlightSet.has(name)) {
                cls += " tag-match";
                icon = '<span class="tag-icon match">&#10003;</span>';
            } else if (i < 5) {
                cls += " tag-miss";
            }
        }
        return `<span class="${cls}">${icon}${name}${scoreHtml}</span>`;
    }).join("");
}

function renderRatingBar(rating) {
    if (!rating || typeof rating !== "object" || !Object.keys(rating).length) return "";
    return Object.entries(rating).map(([k, v]) => {
        const pct = (v * 100).toFixed(1);
        const isHigh = v > 0.3;
        return `<div class="rating-item${isHigh ? " high" : ""}">
            <div class="rating-bar-bg"><div class="rating-bar-fill" style="width:${pct}%"></div></div>
            <span class="rating-name">${k}</span>
            <span class="rating-val">${pct}%</span>
        </div>`;
    }).join("");
}

function renderInfoPanel(origTags, origRating, editedTags, editedRating, tagsValid, top5Set) {
    let html = "";

    // Rating
    const origR = renderRatingBar(origRating);
    const editedR = editedRating ? renderRatingBar(editedRating) : null;
    if (origR) {
        html += `<div class="rating-section"><div class="info-title">Content Rating</div><div class="rating-compare">
            <div class="rating-col"><div class="col-label">Original</div><div class="rating-grid">${origR}</div></div>`;
        if (editedR) html += `<div class="rating-col"><div class="col-label">Edited</div><div class="rating-grid">${editedR}</div></div>`;
        html += `</div></div>`;
    }

    // Tags
    html += `<div class="tags-section"><div class="info-title">Detected Tags <span class="tag-hint">(top-5 highlighted)</span></div>`;
    if (tagsValid === false) {
        html += `<div class="tag-warning">None of the original top-5 tags found in the edited result. This edit will not count toward the leaderboard.</div>`;
    }
    html += `<div class="tags-compare">
        <div class="tags-col"><div class="col-label">Original</div>
            <div class="tags-wrap">${renderTagList(origTags, top5Set, "original")}</div></div>`;
    if (editedTags) {
        html += `<div class="tags-col"><div class="col-label">Edited</div>
            <div class="tags-wrap">${renderTagList(editedTags, top5Set, "edited")}</div></div>`;
    }
    html += `</div></div>`;

    $infoPanel.innerHTML = html;
}

// GPU polling
async function pollGpu() {
    try {
        const res = await fetch(`${API}/api/gpu/status`);
        const s = await res.json();
        $gpuStatus.textContent = s.available_count > 0 ? `GPU: ${s.available_count}/${s.total} available` : `GPU: busy`;
        if (sessionId) $editBtn.disabled = s.available_count === 0;
    } catch {}
}
setInterval(pollGpu, 3000);
pollGpu();

// Leaderboard
function renderLeaderboardInto($el, board) {
    if (!Array.isArray(board) || !board.length) {
        $el.innerHTML = '<div class="lb-empty">No entries yet. Be the first!</div>';
        return;
    }
    const medals = ["&#129351;", "&#129352;", "&#129353;"];
    $el.innerHTML = board.map((e, i) => {
        let timeStr = "";
        if (e.updated_at) {
            const d = new Date(e.updated_at);
            timeStr = `${d.getFullYear()}.${String(d.getMonth()+1).padStart(2,"0")}.${String(d.getDate()).padStart(2,"0")} ${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}`;
        }
        return `<div class="lb-card">
            <div class="lb-header">
                <span class="lb-rank">${medals[i] || ""}</span>
                <span class="lb-creator">@${e.creator}</span>
                <span class="lb-gap">+${e.gap.toFixed(2)}</span>
            </div>
            <div class="lb-images">
                <div class="lb-img-wrap"><img src="data:image/png;base64,${e.original_image}" alt="before"><div class="lb-img-label">Before ${e.initial_score.toFixed(2)}</div></div>
                <div class="lb-arrow">&rarr;</div>
                <div class="lb-img-wrap"><img src="data:image/png;base64,${e.final_image}" alt="after"><div class="lb-img-label">After ${e.final_score.toFixed(2)}</div></div>
            </div>
            ${timeStr ? `<div class="lb-time">${timeStr}</div>` : ""}
        </div>`;
    }).join("");
}

async function refreshLeaderboard() {
    try {
        const res = await fetch(`${API}/api/leaderboard`);
        const board = await res.json();
        if ($lbRow) renderLeaderboardInto($lbRow, board);
        if ($lbRowWs) renderLeaderboardInto($lbRowWs, board);
    } catch {}
}
setInterval(refreshLeaderboard, 10000);
refreshLeaderboard();
