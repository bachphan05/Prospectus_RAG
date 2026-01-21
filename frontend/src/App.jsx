import { useState } from 'react'
import FileUpload from './components/FileUpload'
import Dashboard from './components/Dashboard'

function App() {
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  const handleUploadSuccess = (response) => {
    console.log('Upload successful:', response)
    // Trigger dashboard refresh
    setRefreshTrigger(prev => prev + 1)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-2xl font-bold text-gray-900">Intelligent Document Processing System (Hệ thống Xử lý Tài liệu Thông minh)</h1>
          <p className="mt-1 text-sm text-gray-500">Extract financial data from fund prospectuses using AI (Trích xuất dữ liệu tài chính từ bản cáo bạch quỹ đầu tư bằng AI)</p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <section className="bg-white rounded-lg shadow border border-gray-200 p-6 mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-6 text-center">Upload document (Tải tài liệu lên)</h2>
          <FileUpload onUploadSuccess={handleUploadSuccess} />
        </section>

        <section className="w-full">
          <Dashboard refreshTrigger={refreshTrigger} />
        </section>
      </main>
    </div>
  )
}

export default App
