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
];

function doPost(event) {
  const payload = JSON.parse(event.postData.contents || "{}");
  const sheet = getResponseSheet();
  ensureHeaders(sheet);

  const rows = (payload.rows || []).map((row) =>
    HEADERS.map((header) => row[header] ?? ""),
  );

  if (rows.length) {
    sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, HEADERS.length).setValues(rows);
  }

  return ContentService
    .createTextOutput(JSON.stringify({ ok: true, rows: rows.length }))
    .setMimeType(ContentService.MimeType.JSON);
}

function doGet() {
  return ContentService
    .createTextOutput(JSON.stringify({ ok: true }))
    .setMimeType(ContentService.MimeType.JSON);
}

function getResponseSheet() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  return spreadsheet.getSheetByName(SHEET_NAME) || spreadsheet.insertSheet(SHEET_NAME);
}

function ensureHeaders(sheet) {
  const range = sheet.getRange(1, 1, 1, HEADERS.length);
  const current = range.getValues()[0];
  const hasHeaders = HEADERS.every((header, index) => current[index] === header);
  if (!hasHeaders) {
    range.setValues([HEADERS]);
  }
}
