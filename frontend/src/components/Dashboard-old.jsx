import { useState, useEffect } from 'react';
import api from '../services/api';

/**
 * HighlightBox Component - Renders a highlighted bounding box overlay with tooltip
 * @param {Array} bbox - Bounding box [ymin, xmin, ymax, xmax] in 0-1000 scale
 * @param {string} label - Field name/label
 * @param {*} value - Extracted value to display in tooltip
 * @param {boolean} isHighlighted - Whether this box should be highlighted (from UI hover)
 */
const HighlightBox = ({ bbox, label, value, isHighlighted = false }) => {
  if (!bbox || bbox.length !== 4) return null;

  const [ymin, xmin, ymax, xmax] = bbox;

  // Convert 0-1000 scale to CSS percentages
  const style = {
    top: `${ymin / 10}%`,
    left: `${xmin / 10}%`,
    width: `${(xmax - xmin) / 10}%`,
    height: `${(ymax - ymin) / 10}%`,
  };

  return (
    <div
      className={`absolute border-2 transition-all cursor-pointer group z-10 ${
        isHighlighted 
          ? 'border-blue-600 bg-blue-400 bg-opacity-50 animate-pulse' 
          : 'border-yellow-500 bg-yellow-300 bg-opacity-20 hover:bg-opacity-40'
      }`}
      style={style}
    >
      {/* Tooltip on Hover */}
      <div className="hidden group-hover:block absolute bottom-full left-0 mb-1 bg-black text-white text-xs p-1 rounded whitespace-nowrap z-20 shadow-lg">
        <span className="font-bold">{label}:</span> {String(value)}
      </div>
    </div>
  );
};

/**
 * Utility: Extract value from either structured or flat format
 * Handles both new Gemini format {value, page, bbox} and old Mistral format (plain string/number)
 * 
 * @param {*} field - The field data (can be object with {value, page, bbox} or plain value)
 * @returns {*} The extracted value or the field itself if already a plain value
 */
const getValue = (field) => {
  if (field && typeof field === 'object' && 'value' in field) {
    return field.value; // New Gemini structured format
  }
  return field; // Old flat format or null/undefined
};

/**
 * Utility: Get page and bbox info from structured field
 * @param {*} field - The field data
 * @returns {Object|null} {page, bbox} or null if not available
 */
const getFieldInfo = (field) => {
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
const getHighlightsForPage = (data, pageNumber) => {
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

/**
 * DataField Component - Display/Edit field with hover functionality
 * Moved outside Dashboard to prevent losing focus on input
 */
const DataField = ({ label, field, fieldName, editable, isEditMode, editedData, hoveredField, setHoveredField, updateEditedField }) => {
  // Get the current value based on edit mode
  const getCurrentValue = () => {
    if (!isEditMode) return getValue(field);
    
    // Navigate through editedData using fieldName path
    const keys = fieldName.split('.');
    let current = editedData;
    for (const key of keys) {
      if (!current) return '';
      current = current[key];
    }
    return getValue(current);
  };
  
  const displayValue = getCurrentValue();
  const info = getFieldInfo(field);
  
  const handleMouseEnter = () => {
    if (info && !isEditMode) {
      setHoveredField({ fieldName, page: info.page, bbox: info.bbox });
    }
  };
  
  const handleMouseLeave = () => {
    setHoveredField(null);
  };
  
  const isHovered = hoveredField?.fieldName === fieldName;
  
  if (isEditMode && editable) {
    return (
      <div>
        <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</dt>
        <dd className="mt-1">
          <input
            type="text"
            value={displayValue || ''}
            onChange={(e) => updateEditedField(fieldName, e.target.value)}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </dd>
      </div>
    );
  }
  
  return (
    <div>
      <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</dt>
      <dd 
        className={`mt-1 text-sm text-gray-900 transition-all group ${
          info ? 'cursor-pointer hover:bg-yellow-100 hover:shadow-sm px-2 py-1 -mx-2 -my-1 rounded' : ''
        } ${isHovered ? 'bg-yellow-200 shadow-md' : ''}`}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <span>{displayValue || 'N/A'}</span>
        {info && (
          <span className="ml-2 text-xs text-gray-500 italic font-normal opacity-0 group-hover:opacity-100 transition-opacity">
            (Page {info.page})
          </span>
        )}
      </dd>
    </div>
  );
};

/**
 * Dashboard Component - Professional Design with Tailwind CSS
 * Displays processed documents and their extracted data
 */
function Dashboard({ refreshTrigger }) {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState(null);
  const [optimizedPages, setOptimizedPages] = useState(null);
  const [loadingPages, setLoadingPages] = useState(false);
  const [selectedPage, setSelectedPage] = useState(null);
  const [hoveredField, setHoveredField] = useState(null); // {fieldName, page, bbox}
  const [isEditMode, setIsEditMode] = useState(false);
  const [editedData, setEditedData] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [userComment, setUserComment] = useState('');
  const [changeLogs, setChangeLogs] = useState([]);
  const [showChangeLog, setShowChangeLog] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState(false);

  useEffect(() => {
    loadDocuments();
    loadStats();
  }, [refreshTrigger]);

  useEffect(() => {
    const processingDocs = documents.filter(
      doc => doc.status === 'pending' || doc.status === 'processing'
    );

    if (processingDocs.length > 0) {
      const interval = setInterval(() => {
        loadDocuments();
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [documents]);

  const loadDocuments = async () => {
    try {
      const response = await api.getDocuments();
      setDocuments(response.results || []);
    } catch (error) {
      console.error('Error loading documents:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const statsData = await api.getStats();
      setStats(statsData);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const loadChangeLogs = async (docId) => {
    setLoadingLogs(true);
    try {
      const logsData = await api.getChangeLogs(docId);
      setChangeLogs(logsData.results || []);
    } catch (error) {
      console.error('Error loading change logs:', error);
      setChangeLogs([]);
    } finally {
      setLoadingLogs(false);
    }
  };

  const handleDocumentClick = async (doc) => {
    try {
      const fullDoc = await api.getDocument(doc.id);
      setSelectedDoc(fullDoc);
      setOptimizedPages(null);
      setSelectedPage(null);
      
      // Load optimized pages if available
      if (fullDoc.optimized_file_url) {
        setLoadingPages(true);
        try {
          const pagesData = await api.getOptimizedPages(doc.id);
          setOptimizedPages(pagesData);
        } catch (error) {
          console.error('Error loading optimized pages:', error);
        } finally {
          setLoadingPages(false);
        }
      }
    } catch (error) {
      console.error('Error loading document details:', error);
    }
  };

  const handleReprocess = async (docId) => {
    try {
      await api.reprocessDocument(docId);
      loadDocuments();
      if (selectedDoc && selectedDoc.id === docId) {
        setSelectedDoc(null);
      }
    } catch (error) {
      console.error('Error reprocessing document:', error);
      alert('Failed to reprocess document');
    }
  };

  const handleDelete = async (docId) => {
    if (!confirm('Are you sure you want to delete this document?')) {
      return;
    }

    try {
      await api.deleteDocument(docId);
      loadDocuments();
      if (selectedDoc && selectedDoc.id === docId) {
        setSelectedDoc(null);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      alert('Failed to delete document');
    }
  };

  const handleEdit = () => {
    setIsEditMode(true);
    setEditedData(JSON.parse(JSON.stringify(selectedDoc.extracted_data))); // Deep copy
    setUserComment('');
  };

  const handleCancelEdit = () => {
    setIsEditMode(false);
    setEditedData(null);
    setUserComment('');
  };

  const handleSave = async () => {
    if (!editedData || !selectedDoc) return;

    setIsSaving(true);
    try {
      // Update the document with new extracted data and comment
      const response = await api.updateDocument(selectedDoc.id, {
        extracted_data: editedData,
        user_comment: userComment
      });
      
      // Update local state with new edit count and timestamp
      setSelectedDoc(response);
      setIsEditMode(false);
      setEditedData(null);
      setUserComment('');
      
      // Refresh document list and change logs
      loadDocuments();
      if (response.edit_count > 0) {
        loadChangeLogs(response.id);
      }
      
      alert('Changes saved successfully!');
    } catch (error) {
      console.error('Error saving changes:', error);
      alert('Failed to save changes. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const updateEditedField = (fieldPath, value) => {
    setEditedData(prevData => {
      const newData = JSON.parse(JSON.stringify(prevData)); // Deep copy
      const keys = fieldPath.split('.');
      let current = newData;
      
      // Navigate to the parent of the target field
      for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]]) current[keys[i]] = {};
        current = current[keys[i]];
      }
      
      // Update the field, preserving page and bbox if it's a structured field
      const lastKey = keys[keys.length - 1];
      if (current[lastKey] && typeof current[lastKey] === 'object' && 'value' in current[lastKey]) {
        current[lastKey].value = value;
      } else {
        current[lastKey] = value;
      }
      
      return newData;
    });
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      pending: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Chờ xử lý' },
      processing: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'Đang xử lý' },
      completed: { bg: 'bg-green-100', text: 'text-green-800', label: 'Hoàn thành' },
      failed: { bg: 'bg-red-100', text: 'text-red-800', label: 'Thất bại' },
    };

    const config = statusConfig[status] || statusConfig.pending;

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.bg} ${config.text}`}>
        {config.label}
      </span>
    );
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-96 gap-4">
        <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin"></div>
        <p className="text-gray-600">Đang tải tài liệu...</p>
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
            <div className="text-3xl font-bold text-gray-900">{stats.total}</div>
            <div className="text-sm text-gray-600 uppercase tracking-wide mt-1">Tổng số</div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
            <div className="text-3xl font-bold text-green-600">{stats.completed}</div>
            <div className="text-sm text-gray-600 uppercase tracking-wide mt-1">Hoàn thành</div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
            <div className="text-3xl font-bold text-blue-600">{stats.processing}</div>
            <div className="text-sm text-gray-600 uppercase tracking-wide mt-1">Đang xử lý</div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
            <div className="text-3xl font-bold text-red-600">{stats.failed}</div>
            <div className="text-sm text-gray-600 uppercase tracking-wide mt-1">Thất bại</div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Document List */}
        <div className="lg:col-span-1 bg-white rounded-lg shadow border border-gray-200 overflow-hidden flex flex-col max-h-[calc(100vh-280px)]">
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Tài liệu</h2>
          </div>
          
          <div className="flex-1 overflow-y-auto">
            {documents.length === 0 ? (
              <div className="flex items-center justify-center h-full p-8">
                <p className="text-gray-500 text-center">Chưa có tài liệu nào. Hãy tải lên PDF để bắt đầu!</p>
              </div>
            ) : (
              <div className="divide-y divide-gray-200">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className={`p-4 cursor-pointer transition-colors hover:bg-gray-50 ${
                      selectedDoc?.id === doc.id ? 'bg-blue-50 border-l-4 border-blue-600' : ''
                    }`}
                    onClick={() => handleDocumentClick(doc)}
                  >
                    <div className="flex justify-between items-start gap-3">
                      <div className="flex-1 min-w-0">
                        <h3 className="text-sm font-medium text-gray-900 truncate">{doc.file_name}</h3>
                        <p className="text-xs text-gray-500 mt-1">
                          {new Date(doc.uploaded_at).toLocaleString()}
                        </p>
                        {doc.fund_name && (
                          <p className="text-xs text-blue-600 font-medium mt-1">
                            {doc.fund_name}
                            {doc.fund_code && ` (${doc.fund_code})`}
                          </p>
                        )}
                        {doc.edit_count > 0 && (
                          <p 
                            className="text-xs text-orange-600 mt-1 flex items-center gap-1 cursor-pointer hover:text-orange-800 relative"
                            onMouseEnter={() => {
                              setShowChangeLog(doc.id);
                              loadChangeLogs(doc.id);
                            }}
                            onMouseLeave={() => setShowChangeLog(false)}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                              <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                            </svg>
                            Đã sửa {doc.edit_count} lần
                            
                            {/* Change Log Tooltip */}
                            {showChangeLog === doc.id && (
                              <div className="absolute left-0 top-full mt-2 w-96 bg-white border border-gray-300 rounded-lg shadow-xl z-50 p-3 max-h-80 overflow-y-auto">
                                <div className="text-sm font-semibold text-gray-900 mb-2 border-b pb-2">Lịch sử chỉnh sửa</div>
                                {loadingLogs ? (
                                  <div className="text-xs text-gray-500 text-center py-2">Đang tải...</div>
                                ) : changeLogs.length === 0 ? (
                                  <div className="text-xs text-gray-500 text-center py-2">Không có lịch sử</div>
                                ) : (
                                  <div className="space-y-3">
                                    {changeLogs.map((log, idx) => (
                                      <div key={log.id} className="text-xs border-b last:border-b-0 pb-2 last:pb-0">
                                        <div className="flex justify-between items-start mb-1">
                                          <span className="font-medium text-gray-700">Lần {changeLogs.length - idx}</span>
                                          <span className="text-gray-500">
                                            {new Date(log.changed_at).toLocaleString('vi-VN', {
                                              day: '2-digit',
                                              month: '2-digit',
                                              year: 'numeric',
                                              hour: '2-digit',
                                              minute: '2-digit'
                                            })}
                                          </span>
                                        </div>
                                        {log.user_comment && (
                                          <div className="bg-blue-50 rounded px-2 py-1 mb-1 italic text-blue-800">
                                            "{log.user_comment}"
                                          </div>
                                        )}
                                        <div className="text-gray-600 space-y-0.5">
                                          {Object.entries(log.changes).slice(0, 3).map(([field, change]) => (
                                            <div key={field} className="truncate">
                                              <span className="font-medium">{field}:</span> 
                                              <span className="line-through text-red-600 mx-1">
                                                {String(change.old || 'N/A').substring(0, 30)}{String(change.old || '').length > 30 ? '...' : ''}
                                              </span>
                                              → 
                                              <span className="text-green-600 mx-1">
                                                {String(change.new || 'N/A').substring(0, 30)}{String(change.new || '').length > 30 ? '...' : ''}
                                              </span>
                                            </div>
                                          ))}
                                          {Object.keys(log.changes).length > 3 && (
                                            <div className="text-gray-500 italic">... và {Object.keys(log.changes).length - 3} thay đổi khác</div>
                                          )}
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )}
                          </p>
                        )}
                      </div>
                      <div className="flex-shrink-0">
                        {getStatusBadge(doc.status)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Document Details */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow border border-gray-200 overflow-hidden flex flex-col max-h-[calc(100vh-280px)]">
          {selectedDoc ? (
            <>
              {/* Header */}
              <div className="p-4 border-b border-gray-200">
                <div className="flex justify-between items-center">
                  <div className="flex-1">
                    <h2 className="text-lg font-semibold text-gray-900">Dữ liệu trích xuất</h2>
                    <div className="flex items-center gap-4 mt-1">
                      <p className="text-xs text-gray-500">
                        Mô hình: {selectedDoc.ocr_model === 'gemini' ? 'Gemini 2.0 Flash' : 
                                selectedDoc.ocr_model === 'mistral' ? 'Mistral Large' : 'Mistral OCR + Small'}
                      </p>
                      {selectedDoc.edit_count > 0 && (
                        <p className="text-xs text-orange-600 flex items-center gap-1">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                          </svg>
                          Đã sửa {selectedDoc.edit_count} lần
                          {selectedDoc.last_edited_at && (
                            <span className="text-gray-400">
                              (Lần cuối: {new Date(selectedDoc.last_edited_at).toLocaleString('vi-VN', {
                                day: '2-digit',
                                month: '2-digit',
                                hour: '2-digit',
                                minute: '2-digit'
                              })})
                            </span>
                          )}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    {selectedDoc.status === 'completed' && !isEditMode && (
                      <button
                        className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors flex items-center gap-1"
                        onClick={handleEdit}
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                        Chỉnh sửa
                      </button>
                    )}
                    {isEditMode && (
                      <>
                        <button
                          className="px-3 py-1.5 text-sm font-medium text-white bg-green-600 rounded hover:bg-green-700 disabled:bg-gray-400 transition-colors flex items-center gap-1"
                          onClick={handleSave}
                          disabled={isSaving}
                        >
                          {isSaving ? (
                            <>
                              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                              Đang lưu...
                            </>
                          ) : (
                            <>
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                              Lưu
                            </>
                          )}
                        </button>
                        <button
                          className="px-3 py-1.5 text-sm font-medium text-white bg-gray-500 rounded hover:bg-gray-600 disabled:bg-gray-300 transition-colors flex items-center gap-1"
                          onClick={handleCancelEdit}
                          disabled={isSaving}
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                          Hủy
                        </button>
                      </>
                    )}
                    {selectedDoc.status === 'failed' && (
                      <button
                        className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors"
                        onClick={() => handleReprocess(selectedDoc.id)}
                      >
                        Xử lý lại
                      </button>
                    )}
                    <button
                      className="px-3 py-1.5 text-sm font-medium text-white bg-red-600 rounded hover:bg-red-700 transition-colors"
                      onClick={() => handleDelete(selectedDoc.id)}
                    >
                      Xóa
                    </button>
                    <a
                      href={api.getDownloadUrl(selectedDoc.id)}
                      className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded hover:bg-gray-200 transition-colors"
                      download
                    >
                      Tải PDF
                    </a>
                  </div>
                </div>
              </div>

              {/* Comment Input (only shown in edit mode) */}
              {isEditMode && (
                <div className="px-6 py-3 bg-blue-50 border-b border-blue-200">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Ghi chú về thay đổi (tùy chọn)
                  </label>
                  <input
                    type="text"
                    value={userComment}
                    onChange={(e) => setUserComment(e.target.value)}
                    placeholder="VD: Sửa số liệu phí quản lý, cập nhật tên công ty..."
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              )}

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-6">
                {selectedDoc.status === 'completed' && selectedDoc.extracted_data ? (
                  <div className="space-y-6">
                    {/* Optimized Pages Preview */}
                    {selectedDoc.optimized_file_url && (
                      <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                        <h3 className="text-base font-semibold text-gray-900 mb-4">Các trang đã tối ưu hóa</h3>
                        {loadingPages ? (
                          <div className="flex justify-center py-8">
                            <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin"></div>
                          </div>
                        ) : optimizedPages ? (
                          <>
                
                            <div className="grid grid-cols-4 gap-3 max-h-96 overflow-y-auto">
                              {optimizedPages.pages.map((page) => (
                                <div
                                  key={page.page_number}
                                  className="relative border border-gray-300 rounded cursor-pointer hover:border-blue-500 hover:shadow-md transition-all"
                                  onClick={() => setSelectedPage(page)}
                                >
                                  {/* Page Image */}
                                  <img
                                    src={page.image}
                                    alt={`Page ${page.page_number}`}
                                    className="w-full h-auto"
                                  />
                                  
                                  {/* Page Number Label */}
                                  <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-60 text-white text-xs py-1 text-center">
                                    Trang {page.page_number}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </>
                        ) : (
                          <p className="text-sm text-gray-500 italic">Không có dữ liệu trang đã tối ưu</p>
                        )}
                      </div>
                    )}

                    {/* Fund Information */}
                    <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                      <h3 className="text-base font-semibold text-gray-900 mb-4">Thông tin định danh</h3>
                      <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <DataField 
                          label="Tên quỹ" 
                          field={selectedDoc.extracted_data.fund_name} 
                          fieldName="fund_name"
                          editable={true}
                          isEditMode={isEditMode}
                          editedData={editedData}
                          hoveredField={hoveredField}
                          setHoveredField={setHoveredField}
                          updateEditedField={updateEditedField}
                        />
                        <DataField 
                          label="Mã giao dịch" 
                          field={selectedDoc.extracted_data.fund_code} 
                          fieldName="fund_code"
                          editable={true}
                          isEditMode={isEditMode}
                          editedData={editedData}
                          hoveredField={hoveredField}
                          setHoveredField={setHoveredField}
                          updateEditedField={updateEditedField}
                        />
                        <DataField 
                          label="Tên công ty quản lý quỹ" 
                          field={selectedDoc.extracted_data.management_company} 
                          fieldName="management_company"
                          editable={true}
                          isEditMode={isEditMode}
                          editedData={editedData}
                          hoveredField={hoveredField}
                          setHoveredField={setHoveredField}
                          updateEditedField={updateEditedField}
                        />
                        <DataField 
                          label="Ngân hàng giám sát" 
                          field={selectedDoc.extracted_data.custodian_bank} 
                          fieldName="custodian_bank"
                          editable={true}
                          isEditMode={isEditMode}
                          editedData={editedData}
                          hoveredField={hoveredField}
                          setHoveredField={setHoveredField}
                          updateEditedField={updateEditedField}
                        />
                      </dl>
                    </div>

                    {/* Fee Structure */}
                    {(selectedDoc.extracted_data.fees || selectedDoc.extracted_data.fees_detail) && (
                      <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                        <h3 className="text-base font-semibold text-gray-900 mb-4">Thông tin phí</h3>
                        <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <DataField 
                            label="Phí phát hành" 
                            field={selectedDoc.extracted_data.fees?.subscription_fee || selectedDoc.extracted_data.fees_detail?.issue} 
                            fieldName="fees.subscription_fee"
                            editable={true}
                            isEditMode={isEditMode}
                            editedData={editedData}
                            hoveredField={hoveredField}
                            setHoveredField={setHoveredField}
                            updateEditedField={updateEditedField}
                          />
                          <DataField 
                            label="Phí mua lại" 
                            field={selectedDoc.extracted_data.fees?.redemption_fee || selectedDoc.extracted_data.fees_detail?.redemption} 
                            fieldName="fees.redemption_fee"
                            editable={true}
                            isEditMode={isEditMode}
                            editedData={editedData}
                            hoveredField={hoveredField}
                            setHoveredField={setHoveredField}
                            updateEditedField={updateEditedField}
                          />
                          <DataField 
                            label="Phí quản lý thường niên" 
                            field={selectedDoc.extracted_data.fees?.management_fee || selectedDoc.extracted_data.management_fee} 
                            fieldName="fees.management_fee"
                            editable={true}
                            isEditMode={isEditMode}
                            editedData={editedData}
                            hoveredField={hoveredField}
                            setHoveredField={setHoveredField}
                            updateEditedField={updateEditedField}
                          />
                          <DataField 
                            label="Phí chuyển đổi" 
                            field={selectedDoc.extracted_data.fees?.switching_fee || selectedDoc.extracted_data.fees_detail?.conversion} 
                            fieldName="fees.switching_fee"
                            editable={true}
                            isEditMode={isEditMode}
                            editedData={editedData}
                            hoveredField={hoveredField}
                            setHoveredField={setHoveredField}
                            updateEditedField={updateEditedField}
                          />
                        </dl>
                      </div>
                    )}

                    {/* Portfolio Holdings */}
                    <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                      <h3 className="text-base font-semibold text-gray-900 mb-4">Danh mục đầu tư</h3>
                      {selectedDoc.extracted_data.portfolio && selectedDoc.extracted_data.portfolio.length > 0 ? (
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-100">
                              <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Tên tài sản
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Tỷ trọng
                                </th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {selectedDoc.extracted_data.portfolio.map((item, index) => (
                                <tr key={index} className="hover:bg-gray-50">
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {getValue(item.security_name) || getValue(item.asset_name) || 'N/A'}
                                  </td>
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {getValue(item.percentage) ? `${getValue(item.percentage)}%` : (getValue(item.weight) || 'N/A')}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="text-sm text-gray-500 italic">Không tìm thấy dữ liệu danh mục đầu tư</p>
                      )}
                    </div>

                    {/* NAV History */}
                    <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                      <h3 className="text-base font-semibold text-gray-900 mb-4">Giá trị tài sản ròng (NAV) qua các kỳ</h3>
                      {selectedDoc.extracted_data.nav_history && selectedDoc.extracted_data.nav_history.length > 0 ? (
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-100">
                              <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Kỳ
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Giá trị
                                </th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {selectedDoc.extracted_data.nav_history.map((item, index) => (
                                <tr key={index} className="hover:bg-gray-50">
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {getValue(item.date) || getValue(item.period) || 'N/A'}
                                  </td>
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {getValue(item.nav_per_unit) || getValue(item.value) || 'N/A'}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="text-sm text-gray-500 italic">Không tìm thấy dữ liệu lịch sử NAV</p>
                      )}
                    </div>

                    {/* Dividend History */}
                    <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                      <h3 className="text-base font-semibold text-gray-900 mb-4">Lịch sử chia cổ tức</h3>
                      {selectedDoc.extracted_data.dividend_history && selectedDoc.extracted_data.dividend_history.length > 0 ? (
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-100">
                              <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Ngày
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Giá trị cổ tức
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Ngày thanh toán
                                </th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {selectedDoc.extracted_data.dividend_history.map((item, index) => (
                                <tr key={index} className="hover:bg-gray-50">
                                  <td className="px-4 py-3 text-sm text-gray-900">{getValue(item.date) || 'N/A'}</td>
                                  <td className="px-4 py-3 text-sm text-gray-900">{getValue(item.dividend_per_unit) || 'N/A'}</td>
                                  <td className="px-4 py-3 text-sm text-gray-900">{getValue(item.payment_date) || 'N/A'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="text-sm text-gray-500 italic">Không tìm thấy dữ liệu lịch sử chia cổ tức</p>
                      )}
                    </div>

                    {/* Raw JSON */}
                    <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                      <h3 className="text-base font-semibold text-gray-900 mb-4">Dữ liệu thô (JSON)</h3>
                      <pre className="bg-white p-4 rounded border border-gray-200 overflow-x-auto text-xs text-gray-800">
                        {JSON.stringify(selectedDoc.extracted_data, null, 2)}
                      </pre>
                    </div>
                  </div>
                ) : selectedDoc.status === 'failed' ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <h3 className="text-lg font-semibold text-red-600 mb-2">Xử lý thất bại</h3>
                    <p className="text-gray-600">{selectedDoc.error_message || 'An error occurred during processing'}</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full gap-4">
                    <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin"></div>
                    <p className="text-gray-600">Đang xử lý tài liệu...</p>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500">Chọn một tài liệu để xem chi tiết</p>
            </div>
          )}
        </div>
      </div>

      {/* Page Preview Modal */}
      {selectedPage && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedPage(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh] bg-white rounded-lg overflow-hidden flex flex-col">
            
            {/* Header */}
            <div className="bg-white border-b border-gray-200 px-4 py-3 flex justify-between items-center z-20">
              <h3 className="text-lg font-semibold text-gray-900">
                Preview: Page {selectedPage.page_number}
              </h3>
              <button
                onClick={() => setSelectedPage(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Image Container with Relative Positioning */}
            <div className="relative overflow-auto flex-1 bg-gray-100 p-4 flex justify-center">
              
              {/* Wrapper div needs "relative" so absolute children align to it */}
              <div 
                className="relative inline-block shadow-lg"
                style={{ width: '85%' }}
                onClick={(e) => e.stopPropagation()}
              >
                {/* 1. The Page Image */}
                <img
                  src={selectedPage.image}
                  alt={`Page ${selectedPage.page_number}`}
                  className="w-full h-auto block"
                />

                {/* 2. Bbox-Based Annotation Overlay */}
                {selectedDoc?.extracted_data && 
                  getHighlightsForPage(selectedDoc.extracted_data, selectedPage.page_number)
                    .map((hl) => {
                      const isHighlighted = hoveredField && 
                                          hoveredField.page === selectedPage.page_number &&
                                          JSON.stringify(hoveredField.bbox) === JSON.stringify(hl.bbox);
                      return (
                        <HighlightBox 
                          key={hl.id} 
                          bbox={hl.bbox} 
                          label={hl.label}
                          value={hl.value}
                          isHighlighted={isHighlighted}
                        />
                      );
                    })
                }
              </div>

            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
