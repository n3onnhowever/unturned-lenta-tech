import recognitionColumns from "constants/recognitionColumns";

const normalizeLineEndings = (value) => value.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

export function parseCsv(csvText) {
  const text = normalizeLineEndings(csvText || "").replace(/^\uFEFF/, "");
  const records = [];
  let record = [];
  let field = "";
  let insideQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const nextChar = text[index + 1];

    if (char === '"') {
      if (insideQuotes && nextChar === '"') {
        field += '"';
        index += 1;
      } else {
        insideQuotes = !insideQuotes;
      }
      continue;
    }

    if (char === "," && !insideQuotes) {
      record.push(field);
      field = "";
      continue;
    }

    if (char === "\n" && !insideQuotes) {
      record.push(field);
      if (record.some((value) => value !== "")) {
        records.push(record);
      }
      record = [];
      field = "";
      continue;
    }

    field += char;
  }

  record.push(field);
  if (record.some((value) => value !== "")) {
    records.push(record);
  }

  if (!records.length) {
    return [];
  }

  const headers = records[0];

  return records.slice(1).map((values) =>
    headers.reduce(
      (row, header, index) => ({
        ...row,
        [header]: values[index] ?? "",
      }),
      {}
    )
  );
}

export function getCsvHeaders(csvText) {
  const firstRow = parseCsv(`${normalizeLineEndings(csvText || "").split("\n")[0]}\n`);
  if (firstRow.length) {
    return Object.keys(firstRow[0]);
  }

  return normalizeLineEndings(csvText || "")
    .replace(/^\uFEFF/, "")
    .split("\n")[0]
    .split(",")
    .map((header) => header.replace(/^"|"$/g, ""));
}

export function validateRecognitionCsvHeaders(csvText) {
  const headers = getCsvHeaders(csvText);
  const missingColumns = recognitionColumns.filter((column) => !headers.includes(column));
  const extraColumns = headers.filter((column) => !recognitionColumns.includes(column));
  const exactMatch =
    headers.length === recognitionColumns.length &&
    recognitionColumns.every((column, index) => headers[index] === column);

  return {
    headers,
    exactMatch,
    missingColumns,
    extraColumns,
  };
}
