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
   * Get download URL for a document
   * @param {number} id - Document ID
   * @returns {string} Download URL
   */
  getDownloadUrl(id) {
    return `${API_BASE_URL}/documents/${id}/download/`;
  }
}

export default new ApiService();
