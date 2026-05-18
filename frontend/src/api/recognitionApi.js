import recognitionColumns from "constants/recognitionColumns";
import {
  downloadBackendCsv,
  isBackendEnabled,
  processVideoWithBackend,
} from "api/recognitionBackendApi";
import { createMockRecognitionRows } from "mocks/recognitionMockData";
import { createSummary } from "utils/recognitionAnalytics";

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export async function processVideo(file, onProgress) {
  if (isBackendEnabled()) {
    return processVideoWithBackend(file, onProgress);
  }

  return mockProcessVideo(file, onProgress);
}

export async function mockProcessVideo(file, onProgress) {
  if (!file) {
    throw new Error("Выберите видеофайл перед запуском распознавания.");
  }

  const progressSteps = [0, 8, 18, 32, 45, 58, 72, 84, 93, 100];

  for (const progress of progressSteps) {
    onProgress?.(progress, { stageLabel: "Mock processing" });
    await wait(progress === 0 ? 150 : 280);
  }

  const rows = createMockRecognitionRows(file.name);

  return {
    jobId: `mock-${Date.now()}`,
    filename: file.name,
    status: "completed",
    summary: createSummary(rows),
    rows,
  };
}

const escapeCsvValue = (value) => {
  const stringValue = value === null || value === undefined ? "" : String(value);
  const escaped = stringValue.replace(/"/g, '""');

  return /[",\r\n]/.test(escaped) ? `"${escaped}"` : escaped;
};

export function selectCsvFields(rows) {
  return rows.map((row) =>
    recognitionColumns.reduce(
      (csvRow, column) => ({
        ...csvRow,
        [column]: row[column] ?? "",
      }),
      {}
    )
  );
}

export function buildCsv(rows) {
  const header = recognitionColumns.join(",");
  const body = selectCsvFields(rows).map((row) =>
    recognitionColumns.map((column) => escapeCsvValue(row[column])).join(",")
  );

  return [header, ...body].join("\r\n");
}

export async function downloadCsv(rows, filename = "recognition_result.csv", options = {}) {
  if (options.backend?.jobId) {
    try {
      await downloadBackendCsv(options.backend.jobId, filename);
      return;
    } catch (error) {
      console.warn("Backend CSV download failed, falling back to frontend CSV build.", error);
    }
  }

  const csv = buildCsv(rows);
  const blob = new Blob([`\uFEFF${csv}`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
