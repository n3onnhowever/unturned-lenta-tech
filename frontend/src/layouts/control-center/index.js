import { useEffect, useMemo, useState } from "react";

import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";
import Button from "@mui/material/Button";
import LinearProgress from "@mui/material/LinearProgress";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";

import VuiBox from "components/VuiBox";
import VuiTypography from "components/VuiTypography";
import VuiButton from "components/VuiButton";

import DashboardLayout from "examples/LayoutContainers/DashboardLayout";
import DashboardNavbar from "examples/Navbars/DashboardNavbar";

import uiText, {
  getBreakdownLabel,
  getIssueLabel,
  getProcessingStatusLabel,
  getStatusLabel,
} from "constants/uiText";
import { downloadCsv, processVideo } from "api/recognitionApi";
import { isBackendEnabled } from "api/recognitionBackendApi";
import { sampleRecognitionRows } from "mocks/recognitionMockData";
import {
  buildRecognitionAnalytics,
  getPipelineStageState,
} from "utils/recognitionAnalytics";

import {
  IoAnalytics,
  IoBarChart,
  IoCheckmark,
  IoCloudDownload,
  IoCloudUpload,
  IoDocumentText,
  IoGitNetwork,
  IoPlay,
  IoRefresh,
  IoWarning,
} from "react-icons/io5";

const pipelineStages = uiText.processing.stages;
const tableColumnWidths = {
  product_name: 220,
  price_default: 92,
  price_card: 100,
  price_discount: 80,
  barcode: 112,
  color: 70,
  frame_timestamp: 96,
  status: 98,
  confidence: 110,
  issues: 138,
};
const resultColumns = uiText.results.columns.map((column) => ({
  ...column,
  width: tableColumnWidths[column.key] ?? column.width,
}));

const allowedVideoExtensions = [".mp4", ".mov", ".avi", ".mkv"];

const formatFileSize = (size) => `${(size / 1024 / 1024).toFixed(2)} MB`;

const isSupportedVideoFile = (file) => {
  const lowerName = file.name.toLowerCase();
  const hasAllowedExtension = allowedVideoExtensions.some((extension) =>
    lowerName.endsWith(extension)
  );

  return file.type.startsWith("video/") || hasAllowedExtension;
};

const formatPercent = (value) => `${Math.round(Number(value ?? 0) * 100)}%`;

const getStatusColor = (status) => {
  const normalizedStatus = String(status || "").toLowerCase();

  if (["ready", "completed", "ok"].includes(normalizedStatus)) return "#01B574";
  if (["issue", "failed", "error"].includes(normalizedStatus)) return "#F53C2B";
  return "#FFB547";
};

const formatTableValue = (value) => {
  if (value === undefined || value === null || value === "") return "—";
  return value;
};

const renderResultCell = (row, column) => {
  if (column.key === "status") {
    const status = row.status || "review";
    const color = getStatusColor(status);
    const label = getStatusLabel(status);

    return (
      <VuiBox
        component="span"
        sx={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          minWidth: "86px",
          minHeight: "26px",
          px: 1.25,
          borderRadius: "999px",
          color,
          background: `${color}1F`,
          border: `1px solid ${color}55`,
          fontSize: "11px",
          fontWeight: 700,
          textTransform: "none",
          letterSpacing: "0.02em",
        }}
      >
        {label}
      </VuiBox>
    );
  }

  if (column.key === "confidence") {
    const value = Math.max(0, Math.min(1, Number(row.confidence ?? 0)));
    const color = value >= 0.82 ? "#01B574" : value >= 0.65 ? "#FFB547" : "#F53C2B";

    return (
      <VuiBox minWidth={96}>
        <VuiBox display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
          <VuiTypography color="white" variant="caption" fontWeight="bold">
            {formatPercent(value)}
          </VuiTypography>
        </VuiBox>
        <VuiBox
          sx={{
            width: "100%",
            height: "7px",
            borderRadius: "999px",
            background: "rgba(255, 255, 255, 0.08)",
            overflow: "hidden",
          }}
        >
          <VuiBox
            sx={{
              width: `${Math.round(value * 100)}%`,
              height: "100%",
              borderRadius: "999px",
              background: color,
            }}
          />
        </VuiBox>
      </VuiBox>
    );
  }

  if (column.key === "issues") {
    const issues = row.issueTypes?.length ? row.issueTypes : ["\u043D\u0435\u0442"];

    return (
      <VuiBox display="flex" flexWrap="wrap" gap={0.75} maxWidth={column.width} alignItems="center">
        {issues.slice(0, 2).map((issue) => {
          const isEmpty = issue === "\u043D\u0435\u0442";
          const label = isEmpty ? issue : getIssueLabel(issue);

          return (
            <VuiBox
              key={issue}
              component="span"
              sx={{
                px: 1,
                py: 0.35,
                borderRadius: "999px",
                color: isEmpty ? "#A0AEC0" : "#FFB547",
                background: isEmpty ? "rgba(255, 255, 255, 0.06)" : "rgba(255, 181, 71, 0.12)",
                border: isEmpty
                  ? "1px solid rgba(255, 255, 255, 0.08)"
                  : "1px solid rgba(255, 181, 71, 0.22)",
                fontSize: "11px",
                fontWeight: 700,
                lineHeight: 1.4,
                maxWidth: "128px",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
              title={label}
            >
              {label}
            </VuiBox>
          );
        })}
        {issues.length > 2 && (
          <VuiBox
            component="span"
            sx={{
              px: 1,
              py: 0.35,
              borderRadius: "999px",
              color: "#A0AEC0",
              background: "rgba(255, 255, 255, 0.06)",
              fontSize: "11px",
              fontWeight: 700,
            }}
          >
            +{issues.length - 2}
          </VuiBox>
        )}
      </VuiBox>
    );
  }

  const value = formatTableValue(row[column.key]);
  const isMuted = value === "—" || value === "\u043D\u0435\u0442";

  return (
    <VuiTypography
      color={isMuted ? "text" : "white"}
      variant="caption"
      fontWeight={column.key.startsWith("price_") ? "bold" : "regular"}
      sx={{
        display: "block",
        maxWidth: column.width,
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        textAlign: column.align || "left",
        marginLeft: column.key === "product_name" ? "18px" : 0,
      }}
      title={String(value)}
    >
      {value}
    </VuiTypography>
  );
};

function MetricCard({ label, value, icon, tone = "info" }) {
  return (
    <Card
      sx={{
        height: "100%",
        minHeight: "128px",
        display: "flex",
        justifyContent: "center",
      }}
    >
      <VuiBox display="flex" alignItems="center" justifyContent="space-between" gap={2} width="100%">
        <VuiBox display="flex" flexDirection="column" justifyContent="space-between" minHeight="76px" flex={1}>
          <VuiTypography
            color="text"
            variant="caption"
            textTransform="uppercase"
            sx={{
              display: "block",
              minHeight: "32px",
              maxWidth: "122px",
              lineHeight: 1.35,
              letterSpacing: "0.02em",
            }}
          >
            {label}
          </VuiTypography>
          <VuiTypography color="white" variant="h4" fontWeight="bold" lineHeight={1}>
            {value}
          </VuiTypography>
        </VuiBox>
        <VuiBox
          bgColor={tone}
          width="42px"
          height="42px"
          minWidth="42px"
          borderRadius="lg"
          display="flex"
          alignItems="center"
          justifyContent="center"
          sx={{ alignSelf: "center", boxShadow: "0 10px 22px rgba(0, 117, 255, 0.2)" }}
        >
          {icon}
        </VuiBox>
      </VuiBox>
    </Card>
  );
}

function MiniBar({ label, value, max = 100 }) {
  const width = `${Math.min(100, Math.round((value / max) * 100))}%`;

  return (
    <VuiBox mb={1.5}>
      <VuiBox display="flex" justifyContent="space-between" mb="6px">
        <VuiTypography color="text" variant="caption">
          {label}
        </VuiTypography>
        <VuiTypography color="white" variant="caption" fontWeight="bold">
          {value}
        </VuiTypography>
      </VuiBox>
      <VuiBox height="8px" borderRadius="8px" sx={{ background: "rgba(255,255,255,0.08)" }}>
        <VuiBox height="8px" borderRadius="8px" sx={{ width, background: "#0075ff" }} />
      </VuiBox>
    </VuiBox>
  );
}

function ControlCenter() {
  const backendEnabled = isBackendEnabled();
  const [selectedFile, setSelectedFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [processingStatus, setProcessingStatus] = useState(
    uiText.processing.idle
  );

  const activeRows = result?.rows?.length ? result.rows : sampleRecognitionRows;
  const analytics = useMemo(() => buildRecognitionAnalytics(activeRows), [activeRows]);
  const csvRows = result?.rows?.length ? result.rows : sampleRecognitionRows;

  useEffect(() => {
    if (!selectedFile && !result && !isProcessing && !error) {
      setProcessingStatus(uiText.processing.idle);
    }
  }, [error, isProcessing, result, selectedFile]);

  const kpis = [
    { label: uiText.kpi.shelfHealth, value: analytics.shelfHealthScore, icon: <IoCheckmark color="white" size="22px" /> },
    { label: uiText.kpi.priceTagsFound, value: analytics.priceTagsFound, icon: <IoDocumentText color="white" size="22px" /> },
    { label: uiText.kpi.qrSuccess, value: analytics.qrSuccessRate, icon: <IoGitNetwork color="white" size="22px" /> },
    { label: uiText.kpi.fieldCompleteness, value: analytics.fieldCompleteness, icon: <IoAnalytics color="white" size="22px" /> },
    { label: uiText.kpi.issuesFound, value: analytics.issuesFound, icon: <IoWarning color="white" size="22px" />, tone: "warning" },
    { label: uiText.kpi.averageConfidence, value: analytics.averageConfidenceLabel, icon: <IoBarChart color="white" size="22px" /> },
  ];

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] ?? null;

    if (file && !isSupportedVideoFile(file)) {
      setSelectedFile(null);
      setProgress(0);
      setResult(null);
      setError(uiText.upload.invalidFormat);
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
    setProgress(0);
    setResult(null);
    setProcessingStatus(file ? uiText.upload.selectedFile : uiText.processing.idle);
    setError("");
  };

  const handleReset = () => {
    setSelectedFile(null);
    setProgress(0);
    setResult(null);
    setProcessingStatus(uiText.processing.idle);
    setError("");
  };

  const handleProcessVideo = async () => {
    try {
      setIsProcessing(true);
      setError("");
      setResult(null);
      setProcessingStatus(backendEnabled ? uiText.processing.uploading : uiText.processing.mockProcessing);

      const nextResult = await processVideo(selectedFile, (nextProgress, meta = {}) => {
        setProgress(nextProgress);
        if (meta.stageLabel) {
          setProcessingStatus(meta.stageLabel);
        }
      });
      setResult(nextResult);
      setProcessingStatus(uiText.processing.completed);
    } catch (processError) {
      setProcessingStatus(uiText.processing.failed);
      setError(processError.message || "Не удалось выполнить обработку видео.");
      setProgress(0);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadCsv = async () => {
    await downloadCsv(
      csvRows,
      result?.rows?.length ? "recognition_result.csv" : "sample_recognition_result.csv",
      { backend: result?.backend }
    );
  };

  return (
    <DashboardLayout>
      <DashboardNavbar />
      <VuiBox py={3}>
        <VuiBox id="overview" mb={3}>
          <Card
            sx={{
              overflow: "hidden",
            }}
          >
            <Grid container spacing={4} alignItems="center">
              <Grid item xs={12} lg={10}>
                <VuiTypography color="info" variant="button" fontWeight="bold" mb="10px">
                  {uiText.hero.eyebrow}
                </VuiTypography>
                <VuiTypography color="white" variant="h1" fontWeight="bold" mb="12px">
                  {uiText.hero.title}
                </VuiTypography>
                <VuiTypography color="white" variant="h4" fontWeight="medium" mb="14px">
                  {uiText.hero.subtitle}
                </VuiTypography>
                <VuiTypography color="text" variant="button" display="block" mb={4}>
                  {uiText.hero.description}
                </VuiTypography>
                <VuiBox display="flex" flexWrap="wrap" gap={2}>
                  <VuiButton color="dark" onClick={handleDownloadCsv}>
                    <IoCloudDownload size="16px" style={{ marginRight: 8 }} />
                    {result?.rows?.length ? uiText.hero.csvButton : uiText.hero.sampleCsvButton}
                  </VuiButton>
                </VuiBox>
              </Grid>
            </Grid>
          </Card>
        </VuiBox>

        <Grid container spacing={3} mb={3}>
          {kpis.map((kpi) => (
            <Grid item xs={12} sm={6} xl={2} key={kpi.label}>
              <MetricCard {...kpi} />
            </Grid>
          ))}
        </Grid>

        <Grid container spacing={3} mb={3}>
          <Grid item xs={12} lg={5}>
            <Card id="scan-workspace" sx={{ height: "100%" }}>
              <VuiBox display="flex" alignItems="center" mb={3}>
                <VuiBox
                  bgColor="info"
                  width="48px"
                  height="48px"
                  borderRadius="lg"
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                  mr={2}
                >
                  <IoCloudUpload size="24px" color="white" />
                </VuiBox>
                <VuiBox>
                  <VuiTypography color="white" variant="lg" fontWeight="bold" display="block">
                    {uiText.upload.title}
                  </VuiTypography>
                  <VuiTypography color="text" variant="caption">
                    {uiText.upload.subtitle}
                  </VuiTypography>
                </VuiBox>
              </VuiBox>

              <VuiBox display="flex" flexWrap="wrap" gap={2} alignItems="center" mb={2}>
                <Button variant="contained" component="label" color="info" disabled={isProcessing}>
                  {uiText.upload.selectButton}
                  <input hidden type="file" accept=".mp4,.mov,.avi,.mkv,video/*" onChange={handleFileChange} />
                </Button>
                <VuiButton color="info" onClick={handleProcessVideo} disabled={!selectedFile || isProcessing}>
                  <IoPlay size="16px" style={{ marginRight: 8 }} />
                  {uiText.upload.startButton}
                </VuiButton>
                {(selectedFile || result || error) && (
                  <VuiButton color="dark" onClick={handleReset} disabled={isProcessing}>
                    <IoRefresh size="16px" style={{ marginRight: 8 }} />
                    {uiText.upload.resetButton}
                  </VuiButton>
                )}
              </VuiBox>

              {selectedFile ? (
                <VuiBox mb={2}>
                  <VuiTypography color="white" variant="button" fontWeight="bold">
                    {selectedFile.name}
                  </VuiTypography>
                  <VuiTypography color="text" variant="caption" display="block">
                    {uiText.upload.fileSize}: {formatFileSize(selectedFile.size)}
                  </VuiTypography>
                  <VuiTypography color="text" variant="caption" display="block">
                    {isProcessing ? uiText.upload.processing : result ? uiText.upload.completed : uiText.upload.selectedFile}
                  </VuiTypography>
                </VuiBox>
              ) : (
                <VuiTypography color="text" variant="caption" display="block" mb={2}>
                  {uiText.upload.noFile}
                </VuiTypography>
              )}

              {error && (
                <VuiBox p={2} borderRadius="md" sx={{ background: "rgba(255, 78, 78, 0.14)" }}>
                  <VuiTypography color="white" variant="button" fontWeight="bold">
                    {error}
                  </VuiTypography>
                </VuiBox>
              )}
            </Card>
          </Grid>

          <Grid item xs={12} lg={7}>
            <Card sx={{ height: "100%" }}>
              <VuiTypography color="white" variant="lg" fontWeight="bold" mb={2}>
                {uiText.processing.title}
              </VuiTypography>
              <LinearProgress variant="determinate" value={progress} color="info" />
              <VuiTypography color="text" variant="caption" display="block" mt={1} mb={3}>
                {progress}% · {progress === 0 ? uiText.processing.idle : progress >= 100 ? uiText.processing.completed : getProcessingStatusLabel(processingStatus)}
              </VuiTypography>
              <Grid container spacing={1.5}>
                {pipelineStages.map((stage) => {
                  const state = getPipelineStageState(progress, stage);
                  return (
                    <Grid item xs={12} md={6} key={stage.label}>
                      <VuiBox
                        p={1.75}
                        borderRadius="md"
                        sx={{
                          background:
                            state === "completed"
                              ? "rgba(0, 117, 255, 0.18)"
                              : state === "active"
                              ? "rgba(93, 188, 255, 0.16)"
                              : "rgba(255, 255, 255, 0.05)",
                          border: "1px solid rgba(93, 188, 255, 0.14)",
                        }}
                      >
                        <VuiTypography color="white" variant="button" fontWeight="bold">
                          {stage.label}
                        </VuiTypography>
                        <VuiTypography color="text" variant="caption" display="block">
                          {uiText.processing.statusLabels[state]} · {stage.status}
                        </VuiTypography>
                      </VuiBox>
                    </Grid>
                  );
                })}
              </Grid>
            </Card>
          </Grid>
        </Grid>

        <Grid container spacing={3} mb={3}>
          <Grid item xs={12}>
            <Card sx={{ height: "100%" }}>
              <VuiTypography color="white" variant="lg" fontWeight="bold" mb={2}>
                {uiText.summary.title}
              </VuiTypography>
              <Grid container spacing={2}>
                {[
                  [uiText.summary.framesProcessed, result?.summary?.framesProcessed ?? analytics.framesProcessed],
                  [uiText.summary.priceTagsFound, analytics.priceTagsFound],
                  [uiText.summary.qrDecoded, analytics.qrCodesDecoded],
                  [uiText.summary.fieldsFilled, analytics.fieldsFilled],
                  [uiText.summary.issuesFound, analytics.issuesFound],
                  [uiText.summary.averageConfidence, analytics.averageConfidenceLabel],
                  [uiText.summary.processingTime, `${result?.summary?.processingTimeSec ?? analytics.processingTimeSec} сек`],
                ].map(([label, value], index) => (
                  <Grid item xs={12} sm={6} lg={index === 6 ? 12 : 6} key={label}>
                    <VuiBox
                      p={2}
                      height="100%"
                      borderRadius="md"
                      display="flex"
                      alignItems="center"
                      justifyContent="space-between"
                      gap={2}
                      sx={{
                        background: "rgba(255, 255, 255, 0.045)",
                        border: "1px solid rgba(255, 255, 255, 0.07)",
                      }}
                    >
                      <VuiTypography
                        color="text"
                        variant="caption"
                        textTransform="uppercase"
                        sx={{ lineHeight: 1.35, maxWidth: "70%" }}
                      >
                        {label}
                      </VuiTypography>
                      <VuiTypography color="white" variant="h5" fontWeight="bold" lineHeight={1}>
                        {value}
                      </VuiTypography>
                    </VuiBox>
                  </Grid>
                ))}
              </Grid>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card id="results" sx={{ height: "100%" }}>
              <VuiBox display="flex" justifyContent="space-between" flexWrap="wrap" gap={2} mb={2}>
                <VuiBox>
                  <VuiTypography color="white" variant="lg" fontWeight="bold" display="block">
                    {uiText.results.title}
                  </VuiTypography>
                  <VuiTypography color="text" variant="caption">
                    {uiText.results.subtitle}
                  </VuiTypography>
                </VuiBox>
                <VuiButton color="info" onClick={handleDownloadCsv}>
                  <IoCloudDownload size="16px" style={{ marginRight: 8 }} />
                  {uiText.results.downloadButton}
                </VuiButton>
              </VuiBox>
              <TableContainer
                sx={{
                  overflowX: "auto",
                  borderRadius: "16px",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  background:
                    "linear-gradient(180deg, rgba(8, 18, 48, 0.96) 0%, rgba(5, 12, 34, 0.96) 100%)",
                  boxShadow: "inset 0 1px 0 rgba(255, 255, 255, 0.05)",
                  "&::-webkit-scrollbar": {
                    height: "8px",
                  },
                  "&::-webkit-scrollbar-track": {
                    background: "rgba(255, 255, 255, 0.04)",
                    borderRadius: "999px",
                  },
                  "&::-webkit-scrollbar-thumb": {
                    background: "rgba(0, 117, 255, 0.65)",
                    borderRadius: "999px",
                  },
                }}
              >
                <Table
                  size="small"
                  sx={{
                    width: "100%",
                    minWidth: 1120,
                    backgroundColor: "transparent",
                    borderCollapse: "separate",
                    borderSpacing: 0,
                    tableLayout: "fixed",
                    "& .MuiTableCell-root": {
                      fontFamily: "Inter, sans-serif",
                    },
                  }}
                >
                  <colgroup>
                    {resultColumns.map((column) => (
                      <col key={column.key} style={{ width: column.width }} />
                    ))}
                  </colgroup>
                  <TableHead
                    sx={{
                      display: "table-header-group !important",
                      background:
                        "linear-gradient(180deg, rgba(18, 41, 90, 0.98) 0%, rgba(9, 24, 61, 0.98) 100%)",
                    }}
                  >
                    <TableRow sx={{ display: "table-row !important" }}>
                      {resultColumns.map((column) => (
                        <TableCell
                          key={column.key}
                          align={column.align || "left"}
                          sx={{
                            borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
                            backgroundColor: "transparent !important",
                            color: "#A0AEC0",
                            py: 1.75,
                            px: 1.75,
                            whiteSpace: "nowrap",
                            verticalAlign: "middle",
                          }}
                        >
                          <VuiTypography
                            color="text"
                            variant="caption"
                            fontWeight="bold"
                            textTransform="uppercase"
                            sx={{
                              display: "block",
                              fontSize: "10px",
                              letterSpacing: "0.04em",
                              textAlign: column.align || "left",
                              marginLeft: column.key === "product_name" ? "18px" : 0,
                            }}
                          >
                            {column.label}
                          </VuiTypography>
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {activeRows.map((row, index) => (
                      <TableRow
                        key={`${row.frame_timestamp}-${index}`}
                        sx={{
                          backgroundColor:
                            index % 2 === 0 ? "rgba(255, 255, 255, 0.015)" : "rgba(0, 117, 255, 0.025)",
                          transition: "background-color 160ms ease, box-shadow 160ms ease",
                          "&:hover": {
                            backgroundColor: "rgba(0, 117, 255, 0.11)",
                            boxShadow: "inset 3px 0 0 rgba(0, 117, 255, 0.85)",
                          },
                        }}
                      >
                        {resultColumns.map((column) => (
                          <TableCell
                            key={column.key}
                            align={column.align || "left"}
                            sx={{
                              borderBottom: "1px solid rgba(255, 255, 255, 0.06)",
                              backgroundColor: "transparent !important",
                              height: "58px",
                              py: 1.3,
                              px: 1.75,
                              whiteSpace: "nowrap",
                              verticalAlign: "middle",
                            }}
                          >
                            {renderResultCell(row, column)}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Card>
          </Grid>
        </Grid>

        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card id="analytics" sx={{ height: "100%" }}>
              <VuiTypography color="white" variant="lg" fontWeight="bold" mb={2}>
                {uiText.analytics.title}
              </VuiTypography>
              <VuiTypography color="text" variant="caption" display="block" mb={2}>
                {uiText.analytics.subtitle}
              </VuiTypography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <MiniBar label={uiText.analytics.qrSuccess} value={Number(analytics.qrSuccessRate.replace("%", ""))} />
                  <MiniBar label={uiText.analytics.fieldCompleteness} value={Number(analytics.fieldCompleteness.replace("%", ""))} />
                  <MiniBar label={uiText.analytics.scanQuality} value={Number(analytics.scanQuality.replace("%", ""))} />
                </Grid>
                <Grid item xs={12} md={6}>
                  {Object.entries(analytics.issueBreakdown).map(([label, value]) => (
                    <MiniBar key={label} label={getIssueLabel(label)} value={value} max={analytics.priceTagsFound} />
                  ))}
                  {Object.entries(analytics.statusBreakdown).map(([label, value]) => (
                    <MiniBar key={label} label={getBreakdownLabel("status", label)} value={value} max={analytics.priceTagsFound} />
                  ))}
                  {Object.entries(analytics.priceTypeDistribution).map(([label, value]) => (
                    <MiniBar key={label} label={getBreakdownLabel("price", label)} value={value} max={analytics.priceTagsFound} />
                  ))}
                </Grid>
              </Grid>
              <VuiBox mt={2} p={2} borderRadius="md" sx={{ background: "rgba(1, 117, 255, 0.1)" }}>
                <VuiTypography color="text" variant="caption" display="block">
                  {uiText.analytics.insight}
                </VuiTypography>
                <VuiTypography color="text" variant="caption" display="block">
                  {uiText.analytics.recommendation}
                </VuiTypography>
              </VuiBox>
            </Card>
          </Grid>
        </Grid>

      </VuiBox>
    </DashboardLayout>
  );
}

export default ControlCenter;
