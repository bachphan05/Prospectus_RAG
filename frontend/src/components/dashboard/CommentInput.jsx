/**
 * CommentInput Component - Input field for user comments during edit
 */
const CommentInput = ({ userComment, setUserComment }) => {
  return (
    <div className="px-6 py-3 bg-blue-50 border-b border-blue-200">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        Note about changes (optional) (Ghi chú về thay đổi, tùy chọn)
      </label>
      <input
        type="text"
        value={userComment}
        onChange={(e) => setUserComment(e.target.value)}
        placeholder="Example: Update management fee values, correct company name... (VD: Sửa số liệu phí quản lý, cập nhật tên công ty...)"
        className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
    </div>
  );
};

export default CommentInput;
