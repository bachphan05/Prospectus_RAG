import api from '../../services/api';

/**
 * DocumentHeader Component - Header with title, edit/save buttons, and actions
 */
const DocumentHeader = ({
  selectedDoc,
  isEditMode,
  isSaving,
  handleEdit,
  handleSave,
  handleCancelEdit,
  handleReprocess,
  handleDelete,
  handleChat
}) => {
  return (
    <div className="p-4 border-b border-gray-200">
      <div className="flex justify-between items-center">
        <div className="flex-1">
          <h2 className="text-lg font-semibold text-gray-900">Extracted data (Dữ liệu trích xuất)</h2>
          <div className="flex items-center gap-4 mt-1">
            <p className="text-xs text-gray-500">
              Model (Mô hình): {selectedDoc.ocr_model === 'gemini' ? 'Gemini 2.0 Flash' : 
                      selectedDoc.ocr_model === 'mistral' ? 'Mistral Large' : 'Mistral OCR + Small'}
            </p>
            {selectedDoc.edit_count > 0 && (
              <p className="text-xs text-orange-600 flex items-center gap-1">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                </svg>
                Edited {selectedDoc.edit_count} times (Đã sửa {selectedDoc.edit_count} lần)
                {selectedDoc.last_edited_at && (
                  <span className="text-gray-400">
                    (Last edit (Lần cuối): {new Date(selectedDoc.last_edited_at).toLocaleString('vi-VN', {
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
            <>
              <button
                className="px-3 py-1.5 text-sm font-medium text-white bg-purple-600 rounded hover:bg-purple-700 transition-colors flex items-center gap-1"
                onClick={handleChat}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
                Chat with AI (Chat với AI)
              </button>
              <button
                className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors flex items-center gap-1"
                onClick={handleEdit}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                Edit (Chỉnh sửa)
              </button>
            </>
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
                    Saving... (Đang lưu...)
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Save (Lưu)
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
                Cancel (Hủy)
              </button>
            </>
          )}
          {selectedDoc.status === 'failed' && (
            <button
              className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors"
              onClick={() => handleReprocess(selectedDoc.id)}
            >
              Reprocess (Xử lý lại)
            </button>
          )}
          <button
            className="px-3 py-1.5 text-sm font-medium text-white bg-red-600 rounded hover:bg-red-700 transition-colors"
            onClick={() => handleDelete(selectedDoc.id)}
          >
            Delete (Xóa)
          </button>
          
        </div>
      </div>
    </div>
  );
};

export default DocumentHeader;
