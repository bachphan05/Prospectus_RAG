import { useState, useEffect } from 'react';
import api from '../services/api';

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
                  <h2 className="text-lg font-semibold text-gray-900">Dữ liệu trích xuất</h2>
                  <div className="flex gap-2">
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
                            <p className="text-sm text-gray-600 mb-4">
                              Tổng số trang đã chọn: {optimizedPages.total_pages} / {optimizedPages.original_total || 'N/A'}
                            </p>
                            <div className="grid grid-cols-4 gap-3 max-h-96 overflow-y-auto">
                              {optimizedPages.pages.map((page) => (
                                <div
                                  key={page.page_number}
                                  className="relative border border-gray-300 rounded cursor-pointer hover:border-blue-500 hover:shadow-md transition-all"
                                  onClick={() => setSelectedPage(page)}
                                >
                                  <img
                                    src={page.image}
                                    alt={`Page ${page.page_number}`}
                                    className="w-full h-auto"
                                  />
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
                        <div>
                          <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">Tên quỹ</dt>
                          <dd className="mt-1 text-sm text-gray-900">{selectedDoc.extracted_data.fund_name || 'N/A'}</dd>
                        </div>
                        <div>
                          <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">Mã giao dịch</dt>
                          <dd className="mt-1 text-sm text-gray-900">{selectedDoc.extracted_data.fund_code || 'N/A'}</dd>
                        </div>
                        <div>
                          <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">Tên công ty quản lý quỹ</dt>
                          <dd className="mt-1 text-sm text-gray-900">{selectedDoc.extracted_data.management_company || 'N/A'}</dd>
                        </div>
                        <div>
                          <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">Ngân hàng giám sát</dt>
                          <dd className="mt-1 text-sm text-gray-900">{selectedDoc.extracted_data.custodian_bank || 'N/A'}</dd>
                        </div>
                      </dl>
                    </div>

                    {/* Fee Structure */}
                    {(selectedDoc.extracted_data.fees || selectedDoc.extracted_data.fees_detail) && (
                      <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                        <h3 className="text-base font-semibold text-gray-900 mb-4">Thông tin phí</h3>
                        <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {[
                            { label: 'Phí phát hành', value: selectedDoc.extracted_data.fees?.subscription_fee || selectedDoc.extracted_data.fees_detail?.issue },
                            { label: 'Phí mua lại', value: selectedDoc.extracted_data.fees?.redemption_fee || selectedDoc.extracted_data.fees_detail?.redemption },
                            { label: 'Phí quản lý thường niên', value: selectedDoc.extracted_data.fees?.management_fee || selectedDoc.extracted_data.management_fee },
                            { label: 'Phí chuyển đổi', value: selectedDoc.extracted_data.fees?.switching_fee || selectedDoc.extracted_data.fees_detail?.conversion }
                          ].map((fee, idx) => (
                            <div key={idx}>
                              <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">{fee.label}</dt>
                              <dd className="mt-1 text-sm text-gray-900">
                                {(() => {
                                  if (fee.value === null || fee.value === undefined || fee.value === '') return 'N/A';
                                  if (typeof fee.value === 'number') return `${fee.value}%`;
                                  if (typeof fee.value === 'string') {
                                    if (fee.value.includes('%')) return fee.value;
                                    // Check if it's a pure number string like "1.5"
                                    if (!isNaN(parseFloat(fee.value)) && isFinite(fee.value)) return `${fee.value}%`;
                                    return fee.value;
                                  }
                                  return fee.value;
                                })()}
                              </dd>
                            </div>
                          ))}
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
                                    {item.security_name || item.asset_name || 'N/A'}
                                  </td>
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {item.percentage ? `${item.percentage}%` : (item.weight || 'N/A')}
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
                                    {item.date || item.period || 'N/A'}
                                  </td>
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {item.nav_per_unit || item.value || 'N/A'}
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
                                  <td className="px-4 py-3 text-sm text-gray-900">{item.date || 'N/A'}</td>
                                  <td className="px-4 py-3 text-sm text-gray-900">{item.dividend_per_unit || 'N/A'}</td>
                                  <td className="px-4 py-3 text-sm text-gray-900">{item.payment_date || 'N/A'}</td>
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

                    {/* PDF Viewer */}
                    <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                      <h3 className="text-base font-semibold text-gray-900 mb-4">Xem trước tài liệu</h3>
                      <div className="flex gap-4 mb-4">
                        {selectedDoc.file_url && (
                          <a 
                            href={selectedDoc.file_url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:underline text-sm"
                          >
                            Xem file gốc
                          </a>
                        )}
                        {selectedDoc.optimized_file_url && (
                          <a 
                            href={selectedDoc.optimized_file_url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-green-600 hover:underline text-sm font-medium"
                          >
                            Xem file đã tối ưu (RapidOCR)
                          </a>
                        )}
                      </div>

                      {selectedDoc.optimized_file_url ? (
                        <iframe
                          src={selectedDoc.optimized_file_url}
                          title="Optimized PDF Preview"
                          className="w-full h-96 border border-gray-200 rounded"
                        />
                      ) : selectedDoc.file_url ? (
                        <iframe
                          src={selectedDoc.file_url}
                          title="PDF Preview"
                          className="w-full h-96 border border-gray-200 rounded"
                        />
                      ) : (
                        <p className="text-gray-500">Preview not available</p>
                      )}
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
          <div className="relative max-w-4xl max-h-[90vh] bg-white rounded-lg overflow-hidden">
            <div className="sticky top-0 bg-white border-b border-gray-200 px-4 py-3 flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900">Trang {selectedPage.page_number}</h3>
              <button
                onClick={() => setSelectedPage(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="overflow-auto max-h-[calc(90vh-60px)]">
              <img
                src={selectedPage.image}
                alt={`Page ${selectedPage.page_number}`}
                className="w-full h-auto"
                onClick={(e) => e.stopPropagation()}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
