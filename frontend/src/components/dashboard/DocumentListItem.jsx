import ChangeLogTooltip from './ChangeLogTooltip';

/**
 * DocumentListItem Component - Individual document in the list
 */
const DocumentListItem = ({ 
  doc, 
  isSelected, 
  onSelect, 
  getStatusBadge,
  showChangeLog,
  setShowChangeLog,
  loadChangeLogs,
  changeLogs,
  loadingLogs
}) => {
  return (
    <div
      className={`p-4 cursor-pointer transition-colors hover:bg-gray-50 ${
        isSelected ? 'bg-blue-50 border-l-4 border-blue-600' : ''
      }`}
      onClick={onSelect}
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
              Edited {doc.edit_count} times (Đã sửa {doc.edit_count} lần)
              
              {/* Change Log Tooltip */}
              {showChangeLog === doc.id && (
                <ChangeLogTooltip changeLogs={changeLogs} loadingLogs={loadingLogs} />
              )}
            </p>
          )}
        </div>
        <div className="flex-shrink-0">
          {getStatusBadge(doc.status)}
        </div>
      </div>
    </div>
  );
};

export default DocumentListItem;
