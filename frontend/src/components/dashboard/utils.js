/**
 * Utility: Extract value from either structured or flat format
 * Handles both new Gemini format {value, page, bbox} and old Mistral format (plain string/number)
 * 
 * @param {*} field - The field data (can be object with {value, page, bbox} or plain value)
 * @returns {*} The extracted value or the field itself if already a plain value
 */
export const getValue = (field) => {
  let value = field;

  if (value && typeof value === 'object' && 'value' in value) {
    value = value.value; // New Gemini structured format
  }

  // Never return a raw object to React text nodes.
  if (value && typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }

  return value; // Old flat format or null/undefined
};

/**
 * Utility: Determine if a displayed value should be treated as "N/A" and hidden.
 * Handles common English/Vietnamese variants.
 */
export const isEffectivelyNA = (value) => {
  if (value === null || value === undefined) return true;
  if (typeof value === 'number') return false;

  const str = String(value).trim().toLowerCase();
  if (!str) return true;

  // Common NA variants
  const naValues = new Set([
    'n/a',
    'na',
    'not applicable',
    'none',
    'null',
    'undefined',
    '-',
    '--',
    'khong ap dung',
    'không áp dụng',
    'khong co',
    'không có',
    'khong',
    'không',
    'khong xac dinh',
    'không xác định',
  ]);

  // Remove diacritics for matching Vietnamese text.
  const normalized = str.normalize('NFD').replace(/\p{Diacritic}/gu, '');
  return naValues.has(str) || naValues.has(normalized);
};

/**
 * Utility: Get page and bbox info from structured field
 * @param {*} field - The field data
 * @returns {Object|null} {page, bbox} or null if not available
 */
export const getFieldInfo = (field) => {
  if (field && typeof field === 'object' && 'page' in field && 'bbox' in field) {
    return { page: field.page, bbox: field.bbox };
  }
  return null;
};

/**
 * Utility: Recursively find all bounding boxes for a specific page
 * This function traverses the entire JSON tree and finds all fields that have
 * value, page, and bbox properties matching the given page number.
 * 
 * @param {Object} data - The extracted data object
 * @param {number} pageNumber - The page number to filter by
 * @returns {Array} Array of highlight objects with {id, bbox, label, value}
 */
export const getHighlightsForPage = (data, pageNumber) => {
  const highlights = [];

  const traverse = (obj, keyName = '') => {
    if (!obj || typeof obj !== 'object') return;

    // Check if this object is a "Field with Location" (has value, page, bbox)
    if (obj.page === pageNumber && Array.isArray(obj.bbox) && obj.bbox.length === 4) {
      highlights.push({
        id: keyName + '_' + Math.random(), // Unique ID for React key
        bbox: obj.bbox,
        label: keyName, // e.g., "fund_name"
        value: obj.value
      });
    }

    // Recursively check children
    Object.keys(obj).forEach(key => {
      // Skip metadata keys or pure values
      if (key !== 'bbox' && key !== 'page' && key !== 'value') {
        traverse(obj[key], key);
      }
    });
  };

  traverse(data);
  return highlights;
};
