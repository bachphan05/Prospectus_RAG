import { useState } from 'react';
import api from '../services/api';

/**
 * File Upload Component - Professional Design with Tailwind CSS
 * Handles PDF file upload with drag-and-drop support
 */
function FileUpload({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');
  const [ocrModel, setOcrModel] = useState('gemini');

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setError('');

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      validateAndSetFile(droppedFile);
    }
  };

  const handleFileInput = (e) => {
    setError('');
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile) => {
    // Validate file type
    if (selectedFile.type !== 'application/pdf') {
      setError('Vui lòng tải lên tệp PDF');
      return;
    }

    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024;
    if (selectedFile.size > maxSize) {
      setError('Kích thước tệp phải nhỏ hơn 100MB');
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Vui lòng chọn một tệp');
      return;
    }

    setUploading(true);
    setError('');

    try {
      const response = await api.uploadDocument(file, ocrModel);
      console.log('Upload successful:', response);
      
      // Clear file selection
      setFile(null);
      
      // Notify parent component
      if (onUploadSuccess) {
        onUploadSuccess(response);
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.message || 'Tải lên thất bại');
    } finally {
      setUploading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setError('');
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-lg p-10 text-center transition-all duration-300 cursor-pointer ${
          dragActive 
            ? 'border-blue-500 bg-blue-50 scale-[1.02]' 
            : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-input"
          className="hidden"
          accept=".pdf"
          onChange={handleFileInput}
          disabled={uploading}
        />
        
        <label htmlFor="file-input" className="cursor-pointer block w-full h-full">
          {!file ? (
            <div className="flex flex-col items-center justify-center gap-3">
              <div className="text-gray-400 text-5xl mb-2">📄</div>
              <p className="text-gray-700 font-medium text-lg">
                Kéo và thả tệp PDF vào đây, hoặc nhấp để chọn
              </p>
              <p className="text-gray-500 text-sm">
                Kích thước tối đa: 100MB
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-3">
              <div className="text-green-500 text-5xl mb-2">✓</div>
              <p className="text-gray-800 font-semibold text-lg break-all">{file.name}</p>
              <p className="text-gray-500 text-sm">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          )}
        </label>
      </div>

      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2">
          <span className="text-lg">⚠️</span>
          {error}
        </div>
      )}

      {file && (
        <div className="mt-6">
          <div className="mb-4 flex justify-center">
            <div className="inline-flex rounded-md shadow-sm" role="group">
              <button
                type="button"
                onClick={() => setOcrModel('gemini')}
                className={`px-4 py-2 text-sm font-medium border rounded-l-lg ${
                  ocrModel === 'gemini'
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
                }`}
              >
                Gemini 2.0 Flash
              </button>
              <button
                type="button"
                onClick={() => setOcrModel('mistral')}
                className={`px-4 py-2 text-sm font-medium border rounded-r-lg ${
                  ocrModel === 'mistral'
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
                }`}
              >
                Mistral Large
              </button>
            </div>
          </div>

          <div className="flex gap-3 justify-center">
            <button
              className="px-6 py-2.5 bg-white border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleClear}
              disabled={uploading}
            >
              Hủy
            </button>
            <button
              className="px-6 py-2.5 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Đang tải lên...
                </>
              ) : (
                'Tải lên & Xử lý'
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default FileUpload;
