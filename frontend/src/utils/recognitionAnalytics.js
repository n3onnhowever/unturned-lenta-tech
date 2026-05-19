import recognitionColumns from "constants/recognitionColumns";

const isFilled = (value) => value !== "" && value !== "нет" && value !== null && value !== undefined;

const percent = (value) => `${Math.round(value)}%`;

export const getFindings = (rows) =>
  rows
    .filter((row) => row.issueTypes?.length)
    .map((row) => ({
      issueType: row.issueTypes[0],
      product_name: row.product_name || "Не распознано",
      confidence: row.confidence,
      timestamp: row.frame_timestamp,
      recommendation: row.recommendation || "Проверить ценник на полке",
    }));

export const buildRecognitionAnalytics = (rows) => {
  if (!rows.length) {
    return {
      framesProcessed: 0,
      priceTagsFound: 0,
      qrCodesDecoded: 0,
      fieldsFilled: 0,
      issuesFound: 0,
      averageConfidence: 0,
      processingTimeSec: 0,
      shelfHealthScore: 0,
      qrSuccessRate: "0%",
      fieldCompleteness: "0%",
      averageConfidenceLabel: "0%",
      scanQuality: "0%",
      issueBreakdown: {},
      statusBreakdown: {},
      priceTypeDistribution: {
        default: 0,
        card: 0,
        discount: 0,
      },
    };
  }

  const total = rows.length;
  const qrDecoded = rows.filter((row) => isFilled(row.qr_code_barcode)).length;
  const issueRows = rows.filter((row) => row.issueTypes?.length).length;
  const avgConfidence =
    rows.reduce((sum, row) => sum + Number(row.confidence ?? 0), 0) / total;
  const filledFields = rows.reduce(
    (sum, row) => sum + recognitionColumns.filter((column) => isFilled(row[column])).length,
    0
  );
  const completeness = (filledFields / (total * recognitionColumns.length)) * 100;

  const issueBreakdown = rows.reduce((acc, row) => {
    row.issueTypes?.forEach((issue) => {
      acc[issue] = (acc[issue] || 0) + 1;
    });
    return acc;
  }, {});

  const statusBreakdown = rows.reduce((acc, row) => {
    acc[row.status] = (acc[row.status] || 0) + 1;
    return acc;
  }, {});

  const priceTypeDistribution = {
    default: rows.filter((row) => isFilled(row.price_default)).length,
    card: rows.filter((row) => isFilled(row.price_card)).length,
    discount: rows.filter((row) => isFilled(row.price_discount)).length,
  };

  return {
    framesProcessed: 184,
    priceTagsFound: rows.length,
    qrCodesDecoded: qrDecoded,
    fieldsFilled: filledFields,
    issuesFound: issueRows,
    averageConfidence: avgConfidence,
    processingTimeSec: 7.4,
    shelfHealthScore: Math.max(0, Math.round(100 - issueRows * 8 - (1 - avgConfidence) * 20)),
    qrSuccessRate: percent((qrDecoded / total) * 100),
    fieldCompleteness: percent(completeness),
    averageConfidenceLabel: percent(avgConfidence * 100),
    scanQuality: percent(Math.min(98, completeness * 0.55 + avgConfidence * 45)),
    issueBreakdown,
    statusBreakdown,
    priceTypeDistribution,
  };
};

export const createSummary = (rows) => {
  const analytics = buildRecognitionAnalytics(rows);

  return {
    framesProcessed: analytics.framesProcessed,
    priceTagsFound: analytics.priceTagsFound,
    qrCodesDecoded: analytics.qrCodesDecoded,
    fieldsFilled: analytics.fieldsFilled,
    issuesFound: analytics.issuesFound,
    averageConfidence: analytics.averageConfidenceLabel,
    processingTimeSec: analytics.processingTimeSec,
  };
};

export const getPipelineStageState = (progress, threshold) => {
  if (progress >= threshold.complete) return "completed";
  if (progress >= threshold.active) return "active";
  return "pending";
};
