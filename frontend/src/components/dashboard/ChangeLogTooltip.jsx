/**
 * ChangeLogTooltip Component - Displays change history on hover
 */
const ChangeLogTooltip = ({ changeLogs, loadingLogs }) => {
  return (
    <div className="absolute left-0 top-full mt-2 w-96 bg-white border border-gray-300 rounded-lg shadow-xl z-50 p-3 max-h-80 overflow-y-auto">
      <div className="text-sm font-semibold text-gray-900 mb-2 border-b pb-2">Edit history (Lịch sử chỉnh sửa)</div>
      {loadingLogs ? (
        <div className="text-xs text-gray-500 text-center py-2">Loading... (Đang tải...)</div>
      ) : changeLogs.length === 0 ? (
        <div className="text-xs text-gray-500 text-center py-2">No history (Không có lịch sử)</div>
      ) : (
        <div className="space-y-3">
          {changeLogs.map((log, idx) => (
            <div key={log.id} className="text-xs border-b last:border-b-0 pb-2 last:pb-0">
              <div className="flex justify-between items-start mb-1">
                <span className="font-medium text-gray-700">Edit {changeLogs.length - idx} (Lần {changeLogs.length - idx})</span>
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
                  <div className="text-gray-500 italic">... and {Object.keys(log.changes).length - 3} more changes (... và {Object.keys(log.changes).length - 3} thay đổi khác)</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ChangeLogTooltip;
