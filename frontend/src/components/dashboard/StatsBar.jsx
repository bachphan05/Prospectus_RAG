/**
 * StatsBar Component - Display processing statistics
 */
const StatsBar = ({ stats, loading }) => {
  if (loading || !stats) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="bg-white p-4 rounded-lg shadow border border-gray-200 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-16 mb-2"></div>
            <div className="h-8 bg-gray-300 rounded w-12"></div>
          </div>
        ))}
      </div>
    );
  }

  const statItems = [
    { label: 'Total (Tổng số)', value: stats.total, color: 'text-gray-900', bgColor: 'bg-gray-50' },
    { label: 'Pending (Chờ xử lý)', value: stats.pending, color: 'text-yellow-600', bgColor: 'bg-yellow-50' },
    { label: 'Processing (Đang xử lý)', value: stats.processing, color: 'text-blue-600', bgColor: 'bg-blue-50' },
    { label: 'Completed (Hoàn thành)', value: stats.completed, color: 'text-green-600', bgColor: 'bg-green-50' },
    { label: 'Failed (Thất bại)', value: stats.failed, color: 'text-red-600', bgColor: 'bg-red-50' },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
      {statItems.map((stat, index) => (
        <div key={index} className={`${stat.bgColor} p-4 rounded-lg shadow border border-gray-200 transition-transform hover:scale-105`}>
          <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">{stat.label}</p>
          <p className={`text-2xl font-bold ${stat.color} mt-1`}>{stat.value}</p>
        </div>
      ))}
    </div>
  );
};

export default StatsBar;
