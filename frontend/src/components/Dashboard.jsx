import { useState, useEffect } from 'react';
import api from '../services/api';
import ChatPanel from './ChatPanel';
import {
  DataField,
  DocumentListItem,
  DocumentHeader,
  CommentInput,
  StatsBar
} from './dashboard/index';

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
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [hoveredField, setHoveredField] = useState(null); // For highlighting data fields (hover effect)
  const [isEditMode, setIsEditMode] = useState(false);
  const [editedData, setEditedData] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [userComment, setUserComment] = useState('');
  const [changeLogs, setChangeLogs] = useState([]);
  const [showChangeLog, setShowChangeLog] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [showChatPanel, setShowChatPanel] = useState(false);
  const [chatDocument, setChatDocument] = useState(null);

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
      alert('Failed to reprocess document (Xử lý lại tài liệu thất bại)');
    }
  };

  const handleDelete = async (docId) => {
    if (!confirm('Are you sure you want to delete this document? (Bạn có chắc muốn xóa tài liệu này không?)')) {
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
      alert('Failed to delete document (Xóa tài liệu thất bại)');
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

  const handleChat = () => {
    if (selectedDoc && selectedDoc.status === 'completed') {
      setChatDocument(selectedDoc);
      setShowChatPanel(true);
    }
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
      
      alert('Changes saved successfully! (Đã lưu thay đổi thành công!)');
    } catch (error) {
      console.error('Error saving changes:', error);
      alert('Failed to save changes. Please try again. (Lưu thay đổi thất bại. Vui lòng thử lại.)');
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
      pending: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Pending (Chờ xử lý)' },
      processing: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'Processing (Đang xử lý)' },
      completed: { bg: 'bg-green-100', text: 'text-green-800', label: 'Completed (Hoàn thành)' },
      failed: { bg: 'bg-red-100', text: 'text-red-800', label: 'Failed (Thất bại)' },
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
        <p className="text-gray-600">Loading documents... (Đang tải tài liệu...)</p>
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* Statistics */}
      <StatsBar stats={stats} loading={!stats} />

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Document List */}
        <div className="lg:col-span-1 bg-white rounded-lg shadow border border-gray-200 overflow-hidden flex flex-col max-h-[calc(100vh-280px)]">
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Documents (Tài liệu)</h2>
          </div>
          
          <div className="flex-1 overflow-y-auto">
            {documents.length === 0 ? (
              <div className="flex items-center justify-center h-full p-8">
                <p className="text-gray-500 text-center">No documents yet. Upload a PDF to get started! (Chưa có tài liệu nào. Hãy tải lên PDF để bắt đầu!)</p>
              </div>
            ) : (
              <div className="divide-y divide-gray-200">
                {documents.map((doc) => (
                  <DocumentListItem
                    key={doc.id}
                    doc={doc}
                    isSelected={selectedDoc?.id === doc.id}
                    onSelect={() => handleDocumentClick(doc)}
                    getStatusBadge={getStatusBadge}
                    showChangeLog={showChangeLog}
                    setShowChangeLog={setShowChangeLog}
                    loadChangeLogs={loadChangeLogs}
                    changeLogs={changeLogs}
                    loadingLogs={loadingLogs}
                  />
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
              <DocumentHeader
                selectedDoc={selectedDoc}
                isEditMode={isEditMode}
                isSaving={isSaving}
                handleEdit={handleEdit}
                handleSave={handleSave}
                handleCancelEdit={handleCancelEdit}
                handleReprocess={handleReprocess}
                handleDelete={handleDelete}
                handleChat={handleChat}
              />

              {/* Comment Input (only shown in edit mode) */}
              {isEditMode && (
                <CommentInput userComment={userComment} setUserComment={setUserComment} />
              )}

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-6">
                {selectedDoc.status === 'completed' && selectedDoc.extracted_data ? (
                  <div className="space-y-6">
                    {/* Basic Information */}
                    <div>
                      <h3 className="text-md font-semibold text-gray-900 border-b pb-2 mb-4">Basic information (Thông tin cơ bản)</h3>
                      <dl className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
                        <DataField 
                          label="Fund name (Tên quỹ)" 
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
                          label="Fund code (Mã quỹ)" 
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
                          label="Management company (Công ty quản lý)" 
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
                          label="Custodian bank (Ngân hàng giám sát)" 
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

                    {/* Fee Information */}
                    {selectedDoc.extracted_data.fees && (
                      <div>
                        <h3 className="text-md font-semibold text-gray-900 border-b pb-2 mb-4">Fees (Phí dịch vụ)</h3>
                        <dl className="grid grid-cols-1 gap-y-4">
                          <DataField 
                            label="Management fee (Phí quản lý)" 
                            field={selectedDoc.extracted_data.fees.management_fee}
                            fieldName="fees.management_fee"
                            editable={true}
                            isEditMode={isEditMode}
                            editedData={editedData}
                            hoveredField={hoveredField}
                            setHoveredField={setHoveredField}
                            updateEditedField={updateEditedField}
                          />
                          <DataField 
                            label="Subscription fee (Phí giao dịch mua)" 
                            field={selectedDoc.extracted_data.fees.subscription_fee}
                            fieldName="fees.subscription_fee"
                            editable={true}
                            isEditMode={isEditMode}
                            editedData={editedData}
                            hoveredField={hoveredField}
                            setHoveredField={setHoveredField}
                            updateEditedField={updateEditedField}
                          />
                          <DataField 
                            label="Redemption fee (Phí giao dịch bán)" 
                            field={selectedDoc.extracted_data.fees.redemption_fee}
                            fieldName="fees.redemption_fee"
                            editable={true}
                            isEditMode={isEditMode}
                            editedData={editedData}
                            hoveredField={hoveredField}
                            setHoveredField={setHoveredField}
                            updateEditedField={updateEditedField}
                          />
                          <DataField 
                            label="Switching fee (Phí chuyển đổi)" 
                            field={selectedDoc.extracted_data.fees.switching_fee}
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

                    {/* Portfolio Information */}
                    {selectedDoc.extracted_data.portfolio && selectedDoc.extracted_data.portfolio.length > 0 && (
                      <div>
                        <h3 className="text-md font-semibold text-gray-900 border-b pb-2 mb-4">Portfolio (Danh mục đầu tư)</h3>
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200 text-sm">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">No. (STT)</th>
                                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Ticker (Mã CK)</th>
                                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Quantity (Số lượng)</th>
                                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Value (Giá trị)</th>
                                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">NAV % (Tỷ trọng NAV)</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {selectedDoc.extracted_data.portfolio.map((item, idx) => (
                                <tr key={idx} className="hover:bg-gray-50">
                                  <td className="px-3 py-2 whitespace-nowrap text-gray-900">{item.stt || idx + 1}</td>
                                  <td className="px-3 py-2 whitespace-nowrap font-medium text-gray-900">{item.ma_ck || 'N/A'}</td>
                                  <td className="px-3 py-2 whitespace-nowrap text-gray-900">{item.so_luong || 'N/A'}</td>
                                  <td className="px-3 py-2 whitespace-nowrap text-gray-900">{item.gia_tri || 'N/A'}</td>
                                  <td className="px-3 py-2 whitespace-nowrap text-gray-900">{item.ty_trong || 'N/A'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* NAV History */}
                    {selectedDoc.extracted_data.nav_history && selectedDoc.extracted_data.nav_history.length > 0 && (
                      <div>
                        <h3 className="text-md font-semibold text-gray-900 border-b pb-2 mb-4">NAV history (Lịch sử NAV)</h3>
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200 text-sm">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Date (Ngày)</th>
                                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">NAV (NAV)</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {selectedDoc.extracted_data.nav_history.map((item, idx) => (
                                <tr key={idx} className="hover:bg-gray-50">
                                  <td className="px-3 py-2 whitespace-nowrap text-gray-900">{item.ngay || 'N/A'}</td>
                                  <td className="px-3 py-2 whitespace-nowrap text-gray-900">{item.nav || 'N/A'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Dividend History */}
                    {selectedDoc.extracted_data.dividend_history && selectedDoc.extracted_data.dividend_history.length > 0 && (
                      <div>
                        <h3 className="text-md font-semibold text-gray-900 border-b pb-2 mb-4">Dividend payout history (Lịch sử chi trả cổ tức)</h3>
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200 text-sm">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Date (Ngày)</th>
                                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Amount (Giá trị)</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {selectedDoc.extracted_data.dividend_history.map((item, idx) => (
                                <tr key={idx} className="hover:bg-gray-50">
                                  <td className="px-3 py-2 whitespace-nowrap text-gray-900">{item.ngay || 'N/A'}</td>
                                  <td className="px-3 py-2 whitespace-nowrap text-gray-900">{item.gia_tri || 'N/A'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* PDF Page Thumbnails */}
                    {optimizedPages && optimizedPages.pages && optimizedPages.pages.length > 0 && (
                      <div>
                        <h3 className="text-md font-semibold text-gray-900 border-b pb-2 mb-4">PDF pages (Trang PDF)</h3>
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                          {optimizedPages.pages.map((page) => (
                            <div
                              key={page.page_number}
                              className="border border-gray-300 rounded-lg overflow-hidden cursor-pointer hover:shadow-lg transition-shadow"
                              onClick={() => setSelectedPage(page)}
                            >
                              <img
                                src={page.image}
                                alt={`Page ${page.page_number}`}
                                className="w-full h-auto"
                              />
                              <div className="p-2 bg-gray-50 text-center">
                                <span className="text-xs font-medium text-gray-600">Page {page.page_number} (Trang {page.page_number})</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : selectedDoc.status === 'failed' ? (
                  <div className="text-center py-12">
                    <svg className="mx-auto h-12 w-12 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h3 className="mt-2 text-sm font-medium text-gray-900">Processing failed (Xử lý thất bại)</h3>
                    <p className="mt-1 text-sm text-red-600">{selectedDoc.error_message || 'An error occurred while processing the document (Đã xảy ra lỗi khi xử lý tài liệu)'}</p>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-gray-600">Processing document... (Đang xử lý tài liệu...)</p>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-full p-8">
              <div className="text-center">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <h3 className="mt-2 text-sm font-medium text-gray-900">No document selected (Chưa chọn tài liệu)</h3>
                <p className="mt-1 text-sm text-gray-500">Select a document from the list to view details (Chọn một tài liệu từ danh sách để xem chi tiết)</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Full-Screen Page Modal */}
      {selectedPage && (
        <div 
          className="fixed inset-0 z-50 bg-black bg-opacity-90 flex flex-col"
          onClick={() => setSelectedPage(null)}
        >
          <div className="p-4 flex justify-between items-center">
            <h3 className="text-white text-lg font-semibold">Page {selectedPage.page_number} (Trang {selectedPage.page_number})</h3>
            <button 
              onClick={() => setSelectedPage(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Image Container with Annotated Preview */}
          <div className="relative overflow-auto flex-1 bg-gray-100 p-4 flex justify-center">
            <div 
              className="relative inline-block shadow-lg"
              style={{ width: '70%' }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Loading Spinner */}
              {loadingPreview && (
                <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
                  <div className="flex flex-col items-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                    <p className="mt-2 text-sm text-gray-600">Generating preview with highlights... (Đang tạo bản xem trước với đánh dấu...)</p>
                  </div>
                </div>
              )}
              
              {/* Annotated Page Image with Highlights Burned In */}
              <img
                src={api.getPreviewPageUrl(selectedDoc.id, selectedPage.page_number)}
                alt={`Page ${selectedPage.page_number} with highlights`}
                className="w-full h-auto block"
                onLoadStart={() => setLoadingPreview(true)}
                onLoad={() => setLoadingPreview(false)}
                onError={(e) => {
                  setLoadingPreview(false);
                  // Fallback to original image if preview generation fails
                  console.warn('Preview generation failed, falling back to original image');
                  e.target.src = selectedPage.image;
                }}
              />
            </div>

          </div>
        </div>
      )}

      {/* Chat Panel */}
      {showChatPanel && chatDocument && (
        <ChatPanel 
          document={chatDocument} 
          onClose={() => setShowChatPanel(false)} 
        />
      )}
    </div>
  );
}

export default Dashboard;
