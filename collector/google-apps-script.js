const SPREADSHEET_ID = "1YolTxn3pPpkXveJAuRzRd-opewVickylJRmsbmTvhd4";
const SHEET_NAME = "responses";

const HEADERS = [
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
  "verbalAppropriateness",
  "verbalHelpfulness",
  "motionNaturalness",
  "motionExpressiveness",
  "speechMotionCoordination",
  "nonExcessiveness",
  "overallQuality",
  "playback_issue",
  "comment",
  "completed_at",
  "experiment_id",
  "experiment_label",
  "stimulus_set",
  "order_seed",
  "dialogue_id",
  "overall",
  "visualGrounding",
  "proactiveInteraction",
  "memory",
  "scenario_id",
  "variant",
  "source_folder",
];

function doPost(event) {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);

  try {
    const payload = parsePayload(event);
    const sheet = getResponseSheet();
    ensureHeaders(sheet);

    const rows = (payload.rows || []).map((row) =>
      HEADERS.map((header) => normalizeCellValue(row[header])),
    );

    if (rows.length) {
      sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, HEADERS.length).setValues(rows);
    }

    return jsonResponse({ ok: true, rows: rows.length });
  } catch (error) {
    return jsonResponse({ ok: false, error: String(error && error.message ? error.message : error) });
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  const sheet = getResponseSheet();
  ensureHeaders(sheet);
  return jsonResponse({ ok: true, sheet: SHEET_NAME, headers: HEADERS.length });
}

function parsePayload(event) {
  const contents = event && event.postData && event.postData.contents
    ? event.postData.contents
    : "{}";
  return JSON.parse(contents);
}

function normalizeCellValue(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
  return value;
}

function getResponseSheet() {
  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  return spreadsheet.getSheetByName(SHEET_NAME) || spreadsheet.insertSheet(SHEET_NAME);
}

function ensureHeaders(sheet) {
  const range = sheet.getRange(1, 1, 1, HEADERS.length);
  const current = range.getValues()[0];
  const hasHeaders = HEADERS.every((header, index) => current[index] === header);
  if (!hasHeaders) {
    range.setValues([HEADERS]);
  }
  if (sheet.getFrozenRows() < 1) {
    sheet.setFrozenRows(1);
  }
}

function jsonResponse(payload) {
  return ContentService
    .createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}
