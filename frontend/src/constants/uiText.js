const uiText = {
  nav: {
    sidebar: "Панель контроля",
    breadcrumbTitle: "Панель контроля Unturned",
    breadcrumbRoute: "Главная",
  },
  hero: {
    eyebrow: "Панель контроля Unturned",
    title: "Unturned",
    subtitle: "Контроль ценников с видеопотока робота",
    description:
      "Загрузите видео проезда вдоль стеллажа. Система найдет ценники, распознает QR/OCR-данные, отметит позиции для проверки и сформирует CSV для контроля полки.",
    uploadButton: "Загрузить видео",
    sampleCsvButton: "Скачать пример CSV",
    csvButton: "Скачать CSV",
  },
  kpi: {
    shelfHealth: "Индекс контроля полки",
    priceTagsFound: "Найдено ценников",
    qrSuccess: "QR распознано",
    fieldCompleteness: "Заполненность данных",
    issuesFound: "На проверку",
    averageConfidence: "Средняя уверенность",
  },
  upload: {
    title: "Загрузка видео",
    subtitle:
      "Поддерживаются MP4, MOV, AVI и MKV. Загрузите видео, снятое роботом вдоль стеллажа, чтобы запустить распознавание ценников.",
    selectButton: "Выбрать видео",
    startButton: "Запустить распознавание",
    resetButton: "Сбросить",
    noFile: "Файл не выбран. Загрузите видео с робота, чтобы начать обработку.",
    selectedFile: "Файл выбран. Можно запускать распознавание.",
    processing: "Видео обрабатывается. Дождитесь завершения pipeline.",
    completed: "Обработка завершена. Результаты доступны в таблице и CSV.",
    fileSize: "Размер",
    invalidFormat: "Выберите видеофайл в формате MP4, MOV, AVI или MKV.",
  },
  processing: {
    title: "Ход обработки",
    idle: "ожидает запуска",
    completed: "готово",
    failed: "ошибка",
    uploading: "Загрузка видео",
    mockProcessing: "Демо-обработка",
    stageLabels: {
      "Uploading video": "Загрузка видео",
      "Mock processing": "Демо-обработка",
      Done: "готово",
      Failed: "ошибка",
      "Video intake": "Прием видео",
      "Frame sampling / price tag detection": "Выбор кадров и поиск ценников",
      "Quality filtering / crop processing": "Фильтр качества и подготовка фрагментов",
      "QR/OCR recognition": "Чтение QR и OCR текста",
      "Deduplication / CSV export": "Объединение дублей и формирование CSV",
    },
    statusLabels: {
      pending: "Ожидает",
      active: "В процессе",
      completed: "Готово",
      failed: "Ошибка",
    },
    stages: [
      {
        label: "Прием видео",
        active: 1,
        complete: 12,
        status: "Загружаем файл и создаем задачу обработки.",
      },
      {
        label: "Выбор кадров",
        active: 12,
        complete: 25,
        status: "Берем информативные кадры из видеопотока.",
      },
      {
        label: "Фильтр качества",
        active: 25,
        complete: 38,
        status: "Отсеиваем смазанные и неинформативные кадры.",
      },
      {
        label: "Поиск ценников",
        active: 38,
        complete: 55,
        status: "Находим области, похожие на ценники.",
      },
      {
        label: "Чтение QR",
        active: 55,
        complete: 68,
        status: "Извлекаем данные из QR-кодов и штрихкодов.",
      },
      {
        label: "OCR текста",
        active: 68,
        complete: 82,
        status: "Распознаем текстовые поля на ценнике.",
      },
      {
        label: "Объединение дублей",
        active: 82,
        complete: 94,
        status: "Склеиваем один ценник из нескольких кадров.",
      },
      {
        label: "Формирование CSV",
        active: 94,
        complete: 100,
        status: "Собираем итоговый файл с результатами.",
      },
    ],
  },
  summary: {
    title: "Сводка обработки",
    framesProcessed: "Обработано кадров",
    priceTagsFound: "Найдено ценников",
    qrDecoded: "QR распознано",
    fieldsFilled: "Заполнено полей",
    issuesFound: "На проверку",
    averageConfidence: "Средняя уверенность",
    processingTime: "Время обработки",
  },
  results: {
    title: "Распознанные ценники",
    subtitle:
      "В таблице показаны ключевые поля для оператора. Полный CSV сохраняется в 29 колонках по заданию.",
    downloadButton: "Скачать CSV",
    columns: [
      { key: "product_name", label: "Товар", width: 240 },
      { key: "price_default", label: "Обычная цена", width: 112, align: "right" },
      { key: "price_card", label: "Цена по карте", width: 112, align: "right" },
      { key: "price_discount", label: "Акция", width: 92, align: "right" },
      { key: "barcode", label: "Штрихкод", width: 132 },
      { key: "color", label: "Цвет", width: 84 },
      { key: "frame_timestamp", label: "Время кадра", width: 106, align: "right" },
      { key: "status", label: "Статус", width: 112 },
      { key: "confidence", label: "Уверенность", width: 132 },
      { key: "issues", label: "Проверка", width: 220 },
    ],
  },
  statusLabels: {
    ok: "Готово",
    ready: "Готово",
    completed: "Готово",
    review: "Проверить",
    issue: "Проблема",
    failed: "Ошибка",
    error: "Ошибка",
  },
  issueLabels: {
    PRICE_QR_OCR_MISMATCH: "Расхождение цены QR/OCR",
    BARCODE_EMPTY: "Штрихкод не найден",
    QR_MISSING: "QR не распознан",
    LOW_CONFIDENCE: "Низкая уверенность",
    OCR_INCOMPLETE: "OCR неполный",
    PRICE_NOT_FOUND: "Цена не найдена",
    POSSIBLE_OCCLUSION: "Возможное перекрытие",
    PROMO_PRICE_FOUND: "Найдена акция",
    DUPLICATE_MERGED: "Дубль объединен",
  },
  analytics: {
    title: "Краткая аналитика",
    subtitle:
      "Показатели помогают понять качество сканирования и найти зоны, которые требуют внимания.",
    qrSuccess: "QR распознано",
    fieldCompleteness: "Заполненность данных",
    scanQuality: "Качество съемки",
    insight:
      "Главный риск: часть ценников требует проверки из-за отсутствующего QR или неполного OCR.",
    recommendation:
      "Рекомендация: повторно просканировать зону с низкой уверенностью распознавания.",
  },
  breakdownLabels: {
    "status: ok": "Готово",
    "status: ready": "Готово",
    "status: review": "Требуют проверки",
    "status: issue": "С проблемами",
    "price: default": "Обычная цена",
    "price: card": "Цена по карте",
    "price: discount": "Акционная цена",
  },
};

export const getStatusLabel = (status) =>
  uiText.statusLabels[String(status || "").toLowerCase()] || status || "Проверить";

export const getIssueLabel = (issue) => uiText.issueLabels[issue] || issue;


export const getBreakdownLabel = (prefix, label) =>
  uiText.breakdownLabels[`${prefix}: ${label}`] || label;

export const getProcessingStatusLabel = (label) => uiText.processing.stageLabels[label] || label;

export default uiText;
