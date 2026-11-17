const API_BASE = (window.API_BASE || "").replace(/\/$/, "");
const SHARED_TEXT_KEY = "medtermExtractorText";

const state = {
  mode: "original", // "original" | "enhanced"
  selectedFile: null,
  uploadPromise: null, // resolves to the (possibly downsized) file actually sent to the API
  pendingCacheText: null,
};

// ---------------------------- element refs --------------------------------
const modeButtons = document.querySelectorAll(".mode-item");
const modeCaption = document.getElementById("modeCaption");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzoneTitle = document.getElementById("dropzoneTitle");
const previewImg = document.getElementById("previewImg");
const scanBtn = document.getElementById("scanBtn");
const scanBtnLabel = document.getElementById("scanBtnLabel");
const scanSpinner = document.getElementById("scanSpinner");
const saveCacheBtn = document.getElementById("saveCacheBtn");
const cacheSaveBtnLabel = document.getElementById("cacheSaveBtnLabel");
const cacheSaveSpinner = document.getElementById("cacheSaveSpinner");

const textInput = document.getElementById("textInput");
const evaluateBtn = document.getElementById("evaluateBtn");
const evalBtnLabel = document.getElementById("evalBtnLabel");
const evalSpinner = document.getElementById("evalSpinner");

const resultsContent = document.getElementById("resultsContent");
const resultModeBadge = document.getElementById("resultModeBadge");
const highlightedText = document.getElementById("highlightedText");
const termsTableBody = document.getElementById("termsTableBody");
const confidenceHeader = document.getElementById("confidenceHeader");
const scoreHeader = document.getElementById("scoreHeader");
const indicationHeader = document.getElementById("indicationHeader");
const noTermsMsg = document.getElementById("noTermsMsg");
const abbrevCol = document.getElementById("abbrevCol");
const abbrevList = document.getElementById("abbrevList");
const droppedPanel = document.getElementById("droppedPanel");
const droppedList = document.getElementById("droppedList");
const errorBanner = document.getElementById("errorBanner");

const saveResultBtn = document.getElementById("saveResultBtn");
const saveBtnLabel = document.getElementById("saveBtnLabel");
const saveSpinner = document.getElementById("saveSpinner");
let lastResult = null; // { text, data } from the most recent successful analysis

const savedText = sessionStorage.getItem(SHARED_TEXT_KEY);
if (savedText && !textInput.value.trim()) textInput.value = savedText;

// ------------------------------ mode toggle --------------------------------
const CAPTIONS = {
  original:
    "Running the <strong>baseline Aho-Corasick</strong> automaton — exact matching only, no context awareness, no fuzzy matching, no abbreviation lookup.",
  enhanced:
    "Running the <strong>enhanced Aho-Corasick</strong> pipeline — skip-table search, context-aware validation, priority-weighted scoring, abbreviation meanings, and fuzzy matching.",
};

function setMode(mode) {
  state.mode = mode;
  document.body.dataset.mode = mode;
  modeButtons.forEach((btn) => {
    const isActive = btn.dataset.mode === mode;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-selected", String(isActive));
  });
  modeCaption.innerHTML = CAPTIONS[mode];

  // If there's already text (typed or scanned), automatically re-run the
  // analysis under the newly selected mode instead of wiping the results.
  if (textInput.value.trim()) {
    runAnalysis();
  } else {
    hideResults();
  }
}

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

// ------------------------------ file upload --------------------------------
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  }),
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  }),
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  // Check MIME type OR file extension (some browsers don't recognize HEIC MIME type)
  const validMimeTypes = /^image\/(jpeg|png|webp|heic|heif)$/;
  const validExtensions = /\.(jpg|jpeg|png|webp|heic|heif)$/i;

  if (!validMimeTypes.test(file.type) && !validExtensions.test(file.name)) {
    showError(
      "Unsupported file type. Please upload a JPG, PNG, WEBP, or HEIC image.",
    );
    return;
  }
  state.selectedFile = file;
  state.uploadPromise = prepareUpload(file);
  state.pendingCacheText = null;
  saveCacheBtn.hidden = true;
  scanBtn.disabled = false;

  // Browsers can't render HEIC/HEIF natively, so skip preview for those formats
  const isHeic =
    /\.(heic|heif)$/i.test(file.name) ||
    file.type.match(/^image\/(heic|heif)$/);
  if (isHeic) {
    previewImg.hidden = true;
    dropzoneTitle.textContent = file.name;
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropzoneTitle.textContent = file.name;
  };
  reader.readAsDataURL(file);
}

// ------------------------- phone-photo helpers ------------------------------
const MAX_CLIENT_SIDE = 2400;          // px, long edge
const RESIZE_THRESHOLD = 2.5 * 1024 * 1024; // only touch files bigger than 2.5 MB

// Large phone photos (often 4-12 MB) are downscaled in the browser before upload
// so they send reliably over mobile data. Small files and HEIC are sent as-is
// (HEIC is decoded server-side). Any failure falls back to the original file.
async function prepareUpload(file) {
  try {
    const isHeic = /\.(heic|heif)$/i.test(file.name) || /^image\/(heic|heif)$/.test(file.type);
    if (isHeic || file.size <= RESIZE_THRESHOLD) return file;
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, MAX_CLIENT_SIDE / Math.max(bmp.width, bmp.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * scale);
    canvas.height = Math.round(bmp.height * scale);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.9));
    if (!blob || blob.size >= file.size) return file;
    return new File([blob], file.name.replace(/\.\w+$/, "") + ".jpg", { type: "image/jpeg" });
  } catch (_) {
    return file;
  }
}

// fetch wrapper with a long timeout (OCR on CPU can take a while) and readable
// errors for the failures that are common on phones / sleeping servers.
async function apiFetch(path, options = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 5 * 60 * 1000);
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...options, signal: ctrl.signal });
  } catch (err) {
    throw new Error(
      err.name === "AbortError"
        ? "The server took too long to respond. Please try again."
        : "Cannot reach the server. Check your connection, or wait a minute if the server is waking up, then try again.",
    );
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
  if (!res.ok) {
    const fallback = res.status === 413 ? "Image is too large to upload."
      : [502, 503, 504].includes(res.status) ? "The server is busy or still starting up. Please retry in a minute."
      : `Request failed (${res.status}).`;
    throw new Error((data && data.error) || fallback);
  }
  return data;
}

// ------------------------------ Scan & Extract ------------------------------
scanBtn.addEventListener("click", async () => {
  if (!state.selectedFile) return;
  setBusy(scanBtn, scanSpinner, scanBtnLabel, true, "Scanning…");
  clearError();

  const formData = new FormData();
  const uploadFile = await state.uploadPromise;
  formData.append("image", uploadFile);

  try {
    const data = await apiFetch("/api/ocr", { method: "POST", body: formData });
    state.pendingCacheText = data.cached ? null : (data.raw_text || "");
    saveCacheBtn.hidden = data.cached;
    textInput.value = data.text || "";
    if (textInput.value.trim())
      sessionStorage.setItem(SHARED_TEXT_KEY, textInput.value);
    if (textInput.value.trim()) {
      await runAnalysis();
    }
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(scanBtn, scanSpinner, scanBtnLabel, false, "Scan &amp; Extract");
  }
});

saveCacheBtn.addEventListener("click", async () => {
  if (!state.selectedFile || state.pendingCacheText === null) {
    showError("Scan a new prescription before saving the OCR result.");
    return;
  }
  setBusy(saveCacheBtn, cacheSaveSpinner, cacheSaveBtnLabel, true, "Saving…");
  clearError();

  const cacheFormData = new FormData();
  cacheFormData.append("image", await state.uploadPromise);
  cacheFormData.append("raw_text", state.pendingCacheText);

  try {
    await apiFetch("/api/ocr/cache", { method: "POST", body: cacheFormData });
    state.pendingCacheText = null;
    saveCacheBtn.hidden = true;
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(saveCacheBtn, cacheSaveSpinner, cacheSaveBtnLabel, false, "↓");
  }
});

// ------------------------------ Re-evaluate ---------------------------------
evaluateBtn.addEventListener("click", () => runAnalysis());

async function runAnalysis() {
  const text = textInput.value.trim();
  if (!text) {
    showError("Please upload a prescription image or type/paste text first.");
    return;
  }
  clearError();
  sessionStorage.setItem(SHARED_TEXT_KEY, text);
  setBusy(evaluateBtn, evalSpinner, evalBtnLabel, true, "Analyzing…");

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode: state.mode }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis request failed.");
    renderResults(text, data);
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(evaluateBtn, evalSpinner, evalBtnLabel, false, "↻ Re-evaluate");
  }
}

// ------------------------- input stats (chars + memory) ---------------------
// Memory is estimated with the same "cells" formula used on the SOP 3 page
// (Tiered Hot/Cold Transition Storage): a trie is built from the terms
// actually detected in this run, then storage is counted per SOP 3's
// |S| x |Sigma| (baseline) and hot/cold tiered (enhanced) formulas.
const STATS_THETA = 4;
const STATS_FULL_ALPHABET = Array.from(
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-/%()+:#•—",
);

const statCharCount = document.getElementById("statCharCount");
const statMemory = document.getElementById("statMemory");

function statsBuildTrie(patterns) {
  const nodes = [{ id: 0, children: new Map() }];
  for (const pattern of patterns) {
    let current = 0;
    for (const ch of pattern) {
      if (!nodes[current].children.has(ch)) {
        nodes.push({ id: nodes.length, children: new Map() });
        nodes[current].children.set(ch, nodes.length - 1);
      }
      current = nodes[current].children.get(ch);
    }
  }

  // Failure links, needed to count the transitions each state inherits.
  const failures = Array(nodes.length).fill(0);
  const queue = [...nodes[0].children.values()];
  while (queue.length) {
    const state = queue.shift();
    for (const [char, child] of nodes[state].children) {
      let fallback = failures[state];
      while (fallback && !nodes[fallback].children.has(char))
        fallback = failures[fallback];
      const transition = nodes[fallback].children.get(char);
      failures[child] =
        transition !== undefined && transition !== child ? transition : 0;
      queue.push(child);
    }
  }
  nodes.forEach((node, i) => {
    node.fail = failures[i];
  });
  return nodes;
}

// Cold-row size d(v): the number of trie edges a cold state actually owns.
// This is the SAME definition SOP 3 uses (coldRowSizes() in sop3/index.html),
// so the memory shown on the analysis page matches SOP 3 exactly. Transitions
// a state would inherit through the failure chain are resolved via the failure
// link and are not counted as this state's own row storage.
function statsColdRowSize(node) {
  return node.children.size;
}

function statsGetAlphabet(trieNodes) {
  return [...new Set(trieNodes.flatMap((node) => [...node.children.keys()]))];
}

function computeMemoryStats(matches) {
  const patterns = [
    ...new Set(
      (matches || [])
        .filter((m) => !m.dropped)
        .map((m) => (m.matched_dictionary_term || m.term || "").toUpperCase())
        .filter(Boolean),
    ),
  ];

  const nodes = statsBuildTrie(patterns);
  const alphabetSize = STATS_FULL_ALPHABET.length;
  const originalStorage = nodes.length * alphabetSize;

  const enhancedAlphabet = statsGetAlphabet(nodes);
  const enhancedAlphabetSize = enhancedAlphabet.length;
  const hotStates = nodes.filter((n) => n.children.size >= STATS_THETA).length;
  const hotDenseStorage = hotStates * enhancedAlphabetSize;
  const coldSparseStorage = nodes
    .filter((n) => n.children.size < STATS_THETA)
    .reduce((sum, n) => sum + statsColdRowSize(n), 0);
  const enhancedStorage = hotDenseStorage + coldSparseStorage;

  const saved = originalStorage - enhancedStorage;
  const savedPercent = originalStorage ? (saved / originalStorage) * 100 : 0;

  return { originalStorage, enhancedStorage, saved, savedPercent };
}

function renderInputStats(text, matches, isEnhanced) {
  statCharCount.textContent = text.length;

  const mem = computeMemoryStats(matches);
  const cells = isEnhanced ? mem.enhancedStorage : mem.originalStorage;
  statMemory.textContent = `${cells} cells`;
}

// ------------------------------ rendering -----------------------------------
function renderResults(originalText, data) {
  resultsContent.hidden = false;
  lastResult = { text: originalText, data };

  const isEnhanced = data.mode === "enhanced";
  resultModeBadge.textContent = isEnhanced ? "ENHANCED" : "ORIGINAL";
  categoryHeader.hidden = !isEnhanced;
  indicationHeader.hidden = !isEnhanced;
  confidenceHeader.hidden = !isEnhanced;
  scoreHeader.hidden = !isEnhanced;
  abbrevCol.hidden = !isEnhanced;
  saveResultBtn.hidden = !isEnhanced;

  renderInputStats(originalText, data.matches, isEnhanced);
  renderHighlightedText(originalText, data.matches);
  renderTermsTable(data.matches, isEnhanced, originalText);

  if (isEnhanced) {
    renderAbbreviations(data.abbreviations || []);
    renderDroppedMatches(data.matches || []);
  }

  recordHistoryEntry(originalText, data);
}

function renderHighlightedText(text, matches) {
  highlightedText.textContent = text;
}

function renderTermsTable(matches, isEnhanced, originalText) {
  termsTableBody.innerHTML = "";
  const keptMatches = matches.filter(
    (m) => !m.dropped && !isStruckThroughQuantity(m, originalText),
  );
  noTermsMsg.hidden = keptMatches.length > 0;

  let lastInstruction = null;
  for (const m of keptMatches) {
    if (isEnhanced && !m.dropped && m.instruction != null) {
      if (lastInstruction === null || m.instruction !== lastInstruction) {
        const instructionRow = document.createElement("tr");
        instructionRow.className = "instruction-separator";
        const instructionCell = document.createElement("td");
        instructionCell.colSpan = 5;
        instructionCell.textContent = instructionLabel(
          m.instruction,
          getInstructionType(m, originalText),
        );
        instructionRow.appendChild(instructionCell);
        termsTableBody.appendChild(instructionRow);
      }
      lastInstruction = m.instruction;
    }
    const tr = document.createElement("tr");

    const displayTerm =
      m.matched_dictionary_term || m.term || m.canonical_term || "";
    if (["S", "SIG"].includes(displayTerm.toUpperCase())) {
      tr.classList.add("sig-row");
    }
    const termTd = document.createElement("td");
    termTd.className = "mono";
    termTd.textContent = displayTerm;
    tr.appendChild(termTd);

    if (isEnhanced) {
      const catTd = document.createElement("td");
      catTd.textContent = m.category_label || m.category || "—";
      tr.appendChild(catTd);

      const indicationTd = document.createElement("td");
      indicationTd.textContent = displayIndication(m);
      tr.appendChild(indicationTd);
    }

    if (isEnhanced) {
      const confTd = document.createElement("td");
      const pill = document.createElement("span");
      pill.className = `pill ${m.confidence === "fuzzy" ? "pill-fuzzy" : "pill-exact"}`;
      pill.textContent = m.confidence === "fuzzy" ? "Fuzzy" : "Exact";
      confTd.appendChild(pill);
      tr.appendChild(confTd);

      const scoreTd = document.createElement("td");
      scoreTd.className = "mono";
      scoreTd.textContent = m.score;
      tr.appendChild(scoreTd);
    }

    termsTableBody.appendChild(tr);
  }
}

function isStruckThroughQuantity(match, text) {
  const term = match.matched_dictionary_term || match.term || "";
  if (match.category !== "Symbol" || !term.startsWith("#")) return false;

  const start = Math.max(0, (match.start || 0) - 1);
  const end = Math.min(text.length, (match.end || start) + 3);
  return text.slice(start, end).includes("\u0336");
}

function renderDroppedMatches(matches) {
  const droppedMatches = matches.filter((m) => m.dropped);
  droppedPanel.hidden = droppedMatches.length === 0;
  droppedList.innerHTML = "";
  for (const match of droppedMatches) {
    const li = document.createElement("li");
    const term = match.matched_dictionary_term || match.term || "";
    li.innerHTML = `<span class="abbrev-term">${escapeHtml(term)}</span><span class="abbrev-meaning">${escapeHtml(match.drop_reason || "Dropped")}</span>`;
    droppedList.appendChild(li);
  }
}

function renderAbbreviations(abbreviations) {
  abbrevList.innerHTML = "";
  if (!abbreviations.length) {
    const li = document.createElement("li");
    li.textContent = "No abbreviations detected.";
    abbrevList.appendChild(li);
    return;
  }
  for (const a of abbreviations) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="abbrev-term">${escapeHtml(a.term)}</span><span class="abbrev-meaning">${escapeHtml(a.meaning)}</span>`;
    abbrevList.appendChild(li);
  }
}

// ------------------------------ save result as PDF --------------------------
saveResultBtn.addEventListener("click", downloadResultPdf);

function downloadResultPdf() {
  if (!lastResult) return;
  if (!window.jspdf || !window.jspdf.jsPDF) {
    showError(
      "PDF library failed to load. Check your connection and try again.",
    );
    return;
  }

  setBusy(saveResultBtn, saveSpinner, saveBtnLabel, true, "Preparing PDF…");

  try {
    const { jsPDF } = window.jspdf;
    const { text, data } = lastResult;
    const isEnhanced = data.mode === "enhanced";
    const matches = data.matches || [];
    const abbreviations = data.abbreviations || [];

    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const margin = 40;
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const maxWidth = pageWidth - margin * 2;
    let y = margin;

    const ensureSpace = (needed) => {
      if (y + needed > pageHeight - margin) {
        doc.addPage();
        y = margin;
      }
    };

    // Title
    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.text("Medical Term Extractor — Analysis Result", margin, y);
    y += 18;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(110);
    doc.text(
      `Mode: ${isEnhanced ? "Enhanced Aho-Corasick" : "Original Aho-Corasick"}`,
      margin,
      y,
    );
    y += 12;
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, y);
    doc.setTextColor(20);
    y += 22;

    // Extracted text
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    ensureSpace(20);
    doc.text("Extracted Medical Text", margin, y);
    y += 16;

    doc.setFont("courier", "normal");
    doc.setFontSize(9.5);
    const textLines = doc.splitTextToSize(text || "(no text)", maxWidth);
    for (const line of textLines) {
      ensureSpace(12);
      doc.text(line, margin, y);
      y += 12;
    }
    y += 14;

    // Medical terms table
    ensureSpace(24);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text("Identified Medical Terms", margin, y);
    y += 6;

    const head = isEnhanced
      ? [["Term", "Category", "Indication", "Confidence", "Score"]]
      : [["Term"]];
    let lastInstruction = null;
    const body = matches.length
      ? matches.flatMap((m) => {
          const row = isEnhanced
            ? [
                m.term,
                m.category_label || m.category || "—",
                displayIndication(m),
                m.confidence === "fuzzy" ? "Fuzzy" : "Exact",
                String(m.score ?? "—"),
              ]
            : [m.term];
          const rows = [];
          if (isEnhanced && !m.dropped && m.instruction != null) {
            if (lastInstruction === null || m.instruction !== lastInstruction) {
              rows.push([
                {
                  content: instructionLabel(
                    m.instruction,
                    getInstructionType(m, originalText),
                  ),
                  colSpan: 5,
                  styles: { fontStyle: "bold", textColor: [255, 255, 255], fillColor: [53, 102, 197] },
                },
              ]);
            }
            lastInstruction = m.instruction;
          }
          rows.push(row);
          return rows;
        })
      : [isEnhanced ? ["—", "No medical terms matched.", "", "", ""] : ["—"]];

    doc.autoTable({
      startY: y + 6,
      margin: { left: margin, right: margin },
      head,
      body,
      styles: { font: "helvetica", fontSize: 9, cellPadding: 5 },
      headStyles: { fillColor: [15, 157, 120], textColor: 255 },
      alternateRowStyles: { fillColor: [245, 247, 250] },
    });
    y = doc.lastAutoTable.finalY + 24;

    // Abbreviations & symbols panel
    if (isEnhanced) {
      ensureSpace(24);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(12);
      doc.text("Abbreviations & Symbols Panel", margin, y);
      y += 6;

      const abbrevBody = abbreviations.length
        ? abbreviations.map((a) => [a.term, a.meaning])
        : [["—", "No abbreviations detected."]];

      doc.autoTable({
        startY: y + 6,
        margin: { left: margin, right: margin },
        head: [["Term", "Meaning"]],
        body: abbrevBody,
        styles: { font: "helvetica", fontSize: 9, cellPadding: 5 },
        headStyles: { fillColor: [15, 157, 120], textColor: 255 },
        alternateRowStyles: { fillColor: [245, 247, 250] },
      });
    }

    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    doc.save(
      `medterm-result-${isEnhanced ? "enhanced" : "original"}-${stamp}.pdf`,
    );
  } catch (err) {
    console.error(err);
    showError("Could not generate the PDF. Please try again.");
  } finally {
    setBusy(
      saveResultBtn,
      saveSpinner,
      saveBtnLabel,
      false,
      "⬇ Save Result (PDF)",
    );
  }
}

function getInstructionType(match, originalText) {
  if (match.instruction_type) return match.instruction_type;
  if (match.instruction === 2 && /\bor\b/i.test(originalText)) return "alternative";
  return "medication";
}

function instructionLabel(n, type) {
  if (type === "alternative") return "Alternative Medication";
  const ordinals = ["", "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"];
  return ordinals[n] ? `${ordinals[n]} Medication Entry` : `Medication Entry ${n}`;
}

function displayIndication(match) {
  const term = (match.matched_dictionary_term || match.term || "").toUpperCase();
  if ((term === "S" || term === "SIG") && (!match.indication || match.indication === "-")) {
    return "Directions for use";
  }
  return match.indication || "-";
}

function categoryClass(category) {
  const map = {
    Drug: "hit-drug",
    Dosage: "hit-dosage",
    "Dosage Form": "hit-dosage-form",
    Duration: "hit-duration",
    Instruction: "hit-instruction",
    Frequency: "hit-frequency",
    Abbreviation: "hit-abbreviation",
    Treatment: "hit-treatment",
  };
  return map[category] || "hit-other";
}

// ------------------------------ utilities -----------------------------------
function setBusy(btn, spinner, label, busy, busyText) {
  btn.disabled = busy || (btn === scanBtn && !state.selectedFile);
  spinner.hidden = !busy;
  label.innerHTML = busy ? busyText : label.dataset.default || label.innerHTML;
  if (!busy && label.dataset.default) label.innerHTML = label.dataset.default;
}

// store default labels once
scanBtnLabel.dataset.default = scanBtnLabel.innerHTML;
evalBtnLabel.dataset.default = evalBtnLabel.innerHTML;
cacheSaveBtnLabel.dataset.default = cacheSaveBtnLabel.innerHTML;

function hideResults() {
  resultsContent.hidden = true;
  saveResultBtn.hidden = true;
  lastResult = null;
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.hidden = false;
}
function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ------------------------------ scan history --------------------------------
const HISTORY_KEY = "medtermExtractorHistory";
const HISTORY_LIMIT = 50;

function recordHistoryEntry(text, data) {
  saveActivityHistory({
    source: "main",
    sourcePage: "../index.html",
    mode: data.mode || state.mode,
    text,
    matchCount: (data.matches || []).length,
    terms: (data.matches || []).map((m) => m.term),
  });
}

// init
setMode("original");
const restoredEntry = consumeActivityHistory("main");
if (restoredEntry) {
  textInput.value = restoredEntry.text || "";
  setMode(restoredEntry.mode || "original");
}