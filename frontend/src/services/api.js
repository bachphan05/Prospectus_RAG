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
   * @returns {Promise} API response
   */
  async uploadDocument(file, ocrModel = 'gemini') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_name', file.name);
    formData.append('ocr_model', ocrModel);

    const response = await fetch(`${API_BASE_URL}/documents/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Upload failed');
    }

    return response.json();
  }

  /**
   * Get all documents
   * @returns {Promise} List of documents
   */
  async getDocuments() {
    const response = await fetch(`${API_BASE_URL}/documents/`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch documents');
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
      throw new Error('Failed to fetch document');
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
      throw new Error('Failed to reprocess document');
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
      throw new Error(error.message || 'Failed to update document');
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
      throw new Error('Failed to delete document');
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
      throw new Error('Failed to fetch stats');
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
      throw new Error('Failed to fetch optimized pages');
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
      throw new Error('Failed to fetch change logs');
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
      throw new Error(error.error || 'Failed to check RAG status');
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
      throw new Error(error.error || 'Failed to load chat history');
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
      throw new Error(error.error || 'Failed to save chat history');
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
