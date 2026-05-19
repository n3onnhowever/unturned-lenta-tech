import recognitionColumns from "constants/recognitionColumns";
import { parseCsv, validateRecognitionCsvHeaders } from "utils/csvParser";

const POLL_INTERVAL_MS = 4000;

const notEmpty = (value) => {
  const normalized = String(value ?? "").trim().toLowerCase();
  return normalized !== "" && normalized !== "нет" && normalized !== "рґрµс‚";
};

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function getBackendBaseUrl() {
  return (process.env.REACT_APP_API_URL || "").replace(/\/+$/, "");
}

export function isBackendEnabled() {
  const mode = (process.env.REACT_APP_RECOGNITION_MODE || "").toLowerCase();
  if (mode === "mock") return false;
  if (mode === "backend") return Boolean(getBackendBaseUrl());
  return Boolean(getBackendBaseUrl());
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail || JSON.stringify(body) : body;
    throw new Error(detail || `Backend request failed with status ${response.status}`);
  }

  return body;
}

async function requestText(url) {
  const response = await fetch(url);
  const body = await response.text();

  if (!response.ok) {
    throw new Error(body || `Backend request failed with status ${response.status}`);
  }

  return body;
}

export async function checkBackendHealth() {
  const baseUrl = getBackendBaseUrl();
  if (!baseUrl) {
    return { online: false, detail: "REACT_APP_API_URL is not set" };
  }

  try {
    const data = await requestJson(`${baseUrl}/health`);
    return { online: true, data };
  } catch (error) {
    return { online: false, detail: error.message };
  }
}

export async function checkBackendMlHealth() {
  const baseUrl = getBackendBaseUrl();
  if (!baseUrl) {
    return { ready: false, detail: "REACT_APP_API_URL is not set" };
  }

  try {
    const data = await requestJson(`${baseUrl}/health/ml`);
    return { ready: Boolean(data.ready), data, detail: data.detail || "" };
  } catch (error) {
    return { ready: false, detail: error.message };
  }
}

export async function uploadRecognitionVideo(file) {
  const baseUrl = getBackendBaseUrl();
  if (!baseUrl) {
    throw new Error("Backend mode requires REACT_APP_API_URL.");
  }

  const formData = new FormData();
  formData.append("file", file);

  return requestJson(`${baseUrl}/jobs/upload`, {
    method: "POST",
    body: formData,
  });
}

export async function getRecognitionJob(jobId) {
  return requestJson(`${getBackendBaseUrl()}/jobs/${jobId}`);
}

export async function getRecognitionResult(jobId) {
  return requestJson(`${getBackendBaseUrl()}/jobs/${jobId}/result`);
}

export async function getRecognitionCsvText(jobId) {
  return requestText(`${getBackendBaseUrl()}/jobs/${jobId}/result.csv`);
}

export async function downloadBackendCsv(jobId, filename = "recognition_result.csv") {
  const csvText = await getRecognitionCsvText(jobId);
  const blob = new Blob([`\uFEFF${csvText.replace(/^\uFEFF/, "")}`], {
    type: "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function stageToProgress(job) {
  const stage = job?.current_stage;
  const status = job?.status;

  if (status === "completed") {
    return { progress: 100, label: "Done" };
  }
  if (status === "failed") {
    return { progress: 0, label: "Failed" };
  }
  if (status === "queued" && stage === "detect") {
    return { progress: 15, label: "Video intake" };
  }
  if (stage === "detect") {
    return { progress: 30, label: "Frame sampling / price tag detection" };
  }
  if (stage === "classify") {
    return { progress: 50, label: "Quality filtering / crop processing" };
  }
  if (stage === "ocr") {
    return { progress: 70, label: "QR/OCR recognition" };
  }
  if (stage === "finalize") {
    return { progress: 90, label: "Deduplication / CSV export" };
  }

  return { progress: 5, label: "Uploading video" };
}

function deriveIssueTypes(row, confidence) {
  const issues = [];
  const hasAnyPrice = [row.price_default, row.price_card, row.price_discount].some(notEmpty);

  if (!notEmpty(row.barcode) && !notEmpty(row.qr_code_barcode)) {
    issues.push("QR_MISSING");
  }
  if (!hasAnyPrice) {
    issues.push("PRICE_NOT_FOUND");
  }
  if (confidence < 0.72) {
    issues.push("LOW_CONFIDENCE");
  }

  const filledCount = recognitionColumns.filter((column) => notEmpty(row[column])).length;
  if (filledCount < 7) {
    issues.push("OCR_INCOMPLETE");
  }

  return [...new Set(issues)];
}

function recommendationForIssues(issues) {
  if (issues.includes("PRICE_NOT_FOUND")) return "Проверить промо-цену";
  if (issues.includes("QR_MISSING")) return "Проверить QR/штрихкод";
  if (issues.includes("LOW_CONFIDENCE")) return "Повторить сканирование зоны";
  if (issues.includes("OCR_INCOMPLETE")) return "Использовать OCR fallback";
  return "Готово к проверке";
}

function addUiFields(row, index) {
  const filledCount = recognitionColumns.filter((column) => notEmpty(row[column])).length;
  const confidence = Math.max(0.55, Math.min(0.98, 0.58 + filledCount / recognitionColumns.length + index * 0.003));
  const issueTypes = deriveIssueTypes(row, confidence);

  return {
    ...row,
    confidence,
    status: issueTypes.length ? "review" : "ready",
    issueTypes,
    recommendation: recommendationForIssues(issueTypes),
    detectionSource: "backend",
    shelfZone: row.frame_timestamp ? `frame ${row.frame_timestamp}` : "unknown",
    category: row.color || "unknown",
    duplicateMerged: false,
  };
}

function countFilledFields(rows) {
  return rows.reduce(
    (total, row) =>
      total + recognitionColumns.filter((column) => String(row[column] ?? "").trim() !== "").length,
    0
  );
}

export function mapBackendToFrontendResult({ job, resultJson, csvRows, csvText, elapsedSec }) {
  const rows = csvRows.map(addUiFields);
  const framesProcessed =
    resultJson?.pipeline?.detection?.frames_extracted ??
    resultJson?.video_metadata?.frame_count ??
    0;
  const priceTagsFound =
    resultJson?.crop_stats?.after_final_dedupe ??
    resultJson?.bundle_meta?.rows_after_dedupe ??
    rows.length;
  const qrCodesDecoded = rows.filter((row) => notEmpty(row.qr_code_barcode)).length;

  return {
    jobId: job.job_id,
    filename: job.filename,
    status: "completed",
    summary: {
      framesProcessed,
      priceTagsFound,
      qrCodesDecoded,
      fieldsFilled: countFilledFields(rows),
      processingTimeSec: elapsedSec,
    },
    rows,
    backend: {
      jobId: job.job_id,
      resultJsonUrl: job.result_json_url,
      resultCsvUrl: job.result_csv_url,
      rawJob: job,
      rawResult: resultJson,
      csvText,
    },
  };
}

export async function processVideoWithBackend(file, onProgress) {
  const startedAt = Date.now();
  onProgress?.(5, { stageLabel: "Uploading video" });
  const upload = await uploadRecognitionVideo(file);
  const jobId = upload.job_id;

  onProgress?.(15, { jobId, stageLabel: "Video intake" });

  // Keep polling until the backend returns a terminal state. Some real videos
  // can take much longer than the original demo timeout on CPU-only machines.
  while (true) {
    const job = await getRecognitionJob(jobId);
    const { progress, label } = stageToProgress(job);
    onProgress?.(progress, { jobId, stageLabel: label, rawJob: job });

    if (job.status === "failed") {
      throw new Error(job.error_message || "Backend recognition job failed.");
    }

    if (job.status === "completed") {
      const resultJson = await getRecognitionResult(jobId);
      const csvText = await getRecognitionCsvText(jobId);
      const validation = validateRecognitionCsvHeaders(csvText);

      if (!validation.exactMatch) {
        console.warn("Backend CSV header does not exactly match the 29-column contract.", validation);
      }

      const csvRows = parseCsv(csvText);
      onProgress?.(100, { jobId, stageLabel: "Done", csvValidation: validation });

      return mapBackendToFrontendResult({
        job,
        resultJson,
        csvRows,
        csvText,
        elapsedSec: Number(((Date.now() - startedAt) / 1000).toFixed(1)),
      });
    }

    await wait(POLL_INTERVAL_MS);
  }

}
