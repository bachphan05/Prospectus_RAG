/**
 * API Service for Document Processing
 */

// Use relative URL to leverage Vite proxy in development
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

class ApiService {
  /**
   * Upload a PDF document
   * @param {File} file - The PDF file to upload
   * @param {string} ocrModel - The OCR model to use ('gemini' or 'mistral')
   * @param {(pct:number)=>void} onProgress - Optional callback receiving upload percentage (0-100)
   * @returns {Promise} API response
   */
  async uploadDocument(file, ocrModel = 'gemini', onProgress) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_name', file.name);
    formData.append('ocr_model', ocrModel);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API_BASE_URL}/documents/`);

      if (xhr.upload && typeof onProgress === 'function') {
        xhr.upload.onprogress = (evt) => {
          if (!evt.lengthComputable) return;
          const pct = Math.round((evt.loaded / evt.total) * 100);
          onProgress(pct);
        };
      }

      xhr.onload = () => {
        const ok = xhr.status >= 200 && xhr.status < 300;
        let data = null;
        try {
          data = xhr.responseText ? JSON.parse(xhr.responseText) : null;
        } catch {
          data = null;
        }

        if (!ok) {
          const msg = (data && (data.message || data.error)) || 'Upload failed (Tải lên thất bại)';
          reject(new Error(msg));
          return;
        }

        resolve(data);
      };

      xhr.onerror = () => {
        reject(new Error('Upload failed (Network error)'));
      };

      xhr.send(formData);
    });
  }

  /**
   * Get all documents
   * @returns {Promise} List of documents
   */
  async getDocuments() {
    const response = await fetch(`${API_BASE_URL}/documents/`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch documents (Không thể tải danh sách tài liệu)');
    }

    return response.json();
  }

  /**
   * Get a specific document by ID
   * @param {number} id - Document ID
   * @returns {Promise} Document details
   */
  async getDocument(id) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch document (Không thể tải tài liệu)');
    }

    return response.json();
  }

  /**
   * Reprocess a document
   * @param {number} id - Document ID
   * @returns {Promise} Updated document
   */
  async reprocessDocument(id) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/reprocess/`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error('Failed to reprocess document (Xử lý lại tài liệu thất bại)');
    }

    return response.json();
  }

  /**
   * Update a document's extracted data
   * @param {number} id - Document ID
   * @param {Object} data - Updated document data
   * @returns {Promise} Updated document
   */
  async updateDocument(id, data) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Failed to update document (Cập nhật tài liệu thất bại)');
    }

    return response.json();
  }

  /**
   * Delete a document
   * @param {number} id - Document ID
   * @returns {Promise}
   */
  async deleteDocument(id) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error('Failed to delete document (Xóa tài liệu thất bại)');
    }

    return true;
  }

  /**
   * Get processing statistics
   * @returns {Promise} Statistics object
   */
  async getStats() {
    const response = await fetch(`${API_BASE_URL}/documents/stats/`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch stats (Không thể tải thống kê)');
    }

    return response.json();
  }

  /**
   * Get optimized PDF pages as images
   * @param {number} id - Document ID
   * @returns {Promise} Pages data with base64 images
   */
  async getOptimizedPages(id) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/optimized_pages/`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch optimized pages (Không thể tải các trang PDF đã tối ưu)');
    }

    return response.json();
  }

  /**
   * Get change logs for a document
   * @param {number} id - Document ID
   * @returns {Promise} Change logs data
   */
  async getChangeLogs(id) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/change_logs/`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch change logs (Không thể tải lịch sử chỉnh sửa)');
    }

    return response.json();
  }

  /**
   * Check RAG ingestion status
   * @param {number} id - Document ID
   * @returns {Promise} Status object
   */
  async ragStatus(id) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/rag_status/`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to check RAG status (Không thể kiểm tra trạng thái RAG)');
    }

    return response.json();
  }

  /**
   * Get persisted chat history for a document
   * @param {number} id - Document ID
   * @returns {Promise<{history: Array}>}
   */
  async getChatHistory(id) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/chat_history/`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || 'Failed to load chat history (Không thể tải lịch sử trò chuyện)');
    }

    return response.json();
  }

  /**
   * Save persisted chat history for a document
   * @param {number} id - Document ID
   * @param {Array} history - Array of message objects
   * @returns {Promise<{history: Array}>}
   */
  async saveChatHistory(id, history = []) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/chat_history/`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ history }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || 'Failed to save chat history (Không thể lưu lịch sử trò chuyện)');
    }

    return response.json();
  }

  /**
   * Get preview page with highlights burned in
   * @param {number} id - Document ID
   * @param {number} pageNum - Page number (1-based)
   * @returns {string} URL to the annotated image
   */
  getPreviewPageUrl(id, pageNum) {
    return `${API_BASE_URL}/documents/${id}/preview-page/${pageNum}/`;
  }

  /**
   * Get page image + matched-text highlight boxes for a citation
   * @param {number} id - Document ID
   * @param {number} pageNum - Raw 1-based page number
   * @param {string} quote - Text snippet to locate on the page
   * @returns {Promise} Page context payload
   */
  async getPageContext(id, pageNum, quote = '') {
    const base = `${API_BASE_URL}/documents/${id}/page-context/${pageNum}/`;
    const qs = quote ? `?quote=${encodeURIComponent(quote)}` : '';
    const response = await fetch(base + qs);
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || 'Failed to load page context');
    }
    return response.json();
  }

  /**
   * Ingest document for RAG (create vector embeddings)
   * @param {number} id - Document ID
   * @returns {Promise} Ingestion result
   */
  async ingestForRag(id) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/ingest_for_rag/`, {
      method: 'POST',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to ingest document');
    }

    return response.json();
  }

  /**
   * Chat with document using RAG
   * @param {number} id - Document ID
   * @param {string} query - User question
   * @param {Array} history - Chat history (optional)
   * @returns {Promise} Chat response
   */
  async chatWithDocument(id, query, history = []) {
    const response = await fetch(`${API_BASE_URL}/documents/${id}/chat/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, history }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to chat with document');
    }

    return response.json();
  }

  /**
   * Get download URL for a document
   * @param {number} id - Document ID
   * @returns {string} Download URL
   */
  getDownloadUrl(id) {
    return `${API_BASE_URL}/documents/${id}/download/`;
  }
}

export default new ApiService();
