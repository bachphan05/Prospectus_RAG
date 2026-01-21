import { useState, useEffect, useRef } from 'react';
import api from '../services/api';

/**
 * ChatPanel Component - AI-powered chat interface for documents
 */
function ChatPanel({ document, onClose }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isIngested, setIsIngested] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const hasLoadedHistoryRef = useRef(false);
  const saveDebounceRef = useRef(null);

  useEffect(() => {
    // Load persisted chat history (if any) then check ingestion.
    hasLoadedHistoryRef.current = false;
    setMessages([]);
    setError(null);
    loadChatHistory();
    checkIngestionStatus();
  }, [document]);

  useEffect(() => {
    // Persist chat history (debounced) whenever messages change.
    if (!document?.id) return;
    if (!hasLoadedHistoryRef.current) return;

    if (saveDebounceRef.current) {
      clearTimeout(saveDebounceRef.current);
    }

    saveDebounceRef.current = setTimeout(() => {
      const historyToSave = (messages || []).map((m) => ({
        sender: m.sender,
        text: m.text,
        timestamp: m.timestamp ? new Date(m.timestamp).toISOString() : new Date().toISOString(),
        chunks_count: m.chunks_count,
      }));

      api.saveChatHistory(document.id, historyToSave).catch((err) => {
        console.warn('Failed to save chat history:', err);
      });
    }, 500);

    return () => {
      if (saveDebounceRef.current) {
        clearTimeout(saveDebounceRef.current);
      }
    };
  }, [messages, document?.id]);

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadChatHistory = async () => {
    if (!document?.id) return;

    try {
      const res = await api.getChatHistory(document.id);
      const saved = Array.isArray(res?.history) ? res.history : [];

      if (saved.length > 0) {
        setMessages(
          saved.map((m) => ({
            sender: m.sender,
            text: m.text,
            timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
            chunks_count: m.chunks_count,
          }))
        );
      }
    } catch (err) {
      console.warn('Failed to load chat history:', err);
    } finally {
      hasLoadedHistoryRef.current = true;
    }
  };

  const checkIngestionStatus = async () => {
    try {
      const result = await api.ragStatus(document.id);
      if (result.is_ingested) {
        setIsIngested(true);
        // Only inject the default system message if there's no existing conversation.
        setMessages((prev) => {
          if (Array.isArray(prev) && prev.length > 0) return prev;
          return [
            {
              sender: 'system',
              text: `Document ready (Tài liệu sẵn sàng)! ${result.chunks_count} knowledge chunks available (Có sẵn ${result.chunks_count} đoạn kiến thức). You can ask questions now (Bạn có thể đặt câu hỏi ngay).`,
              timestamp: new Date(),
            },
          ];
        });
      }
    } catch (err) {
      console.error('Failed to check ingestion status:', err);
      setIsIngested(false);
    }
  };

  const handleIngest = async () => {
    setIsIngesting(true);
    setError(null);

    try {
      const result = await api.ingestForRag(document.id);
      setIsIngested(true);
      setMessages((prev) => {
        if (Array.isArray(prev) && prev.length > 0) return prev;
        return [
          {
            sender: 'system',
            text: `Document processed (Đã xử lý tài liệu)! Created ${result.chunks_count} knowledge chunks (Đã tạo ${result.chunks_count} đoạn kiến thức). You can now ask questions about this document (Bạn có thể hỏi về tài liệu này).`,
            timestamp: new Date(),
          },
        ];
      });
    } catch (err) {
      console.error('Ingestion error:', err);
      setError(err.message || 'Failed to process document for chat (Xử lý tài liệu để trò chuyện thất bại)');
    } finally {
      setIsIngesting(false);
    }
  };

  const handleClose = async () => {
    try {
      if (document?.id && hasLoadedHistoryRef.current) {
        const historyToSave = (messages || []).map((m) => ({
          sender: m.sender,
          text: m.text,
          timestamp: m.timestamp ? new Date(m.timestamp).toISOString() : new Date().toISOString(),
          chunks_count: m.chunks_count,
        }));
        await api.saveChatHistory(document.id, historyToSave);
      }
    } catch (err) {
      console.warn('Failed to save chat history on close:', err);
    } finally {
      onClose();
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      sender: 'user',
      text: inputMessage,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      // Prepare chat history in the format backend expects
      const history = messages.map((msg) => ({
        sender: msg.sender,
        text: msg.text,
      }));

      const response = await api.chatWithDocument(document.id, inputMessage, history);

      const aiMessage = {
        sender: 'ai',
        text: response.answer,
        timestamp: new Date(),
        chunks_count: response.chunks_count,
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error('Chat error:', err);
      const errorMessage = {
        sender: 'system',
        text: `Error (Lỗi): ${err.message}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const suggestedQuestions = [
    'What is the management fee? (Phí quản lý là bao nhiêu?)',
    'Who is the custodian bank? (Ngân hàng giám sát là ai?)',
    'What are the fund\'s investment objectives? (Mục tiêu đầu tư của quỹ là gì?)',
    'What are the subscription and redemption fees? (Phí mua và phí bán là bao nhiêu?)',
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-4xl h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-blue-700">
          <div className="flex items-center gap-3">
            <svg
              className="w-6 h-6 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
              />
            </svg>
            <div>
              <h2 className="text-lg font-semibold text-white">Chat with document (Trò chuyện với tài liệu)</h2>
              <p className="text-xs text-blue-100">{document.file_name}</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="text-white hover:bg-blue-800 rounded-full p-2 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Ingestion Required */}
        {!isIngested && !isIngesting && (
          <div className="flex-1 flex flex-col items-center justify-center p-8 bg-gray-50">
            <svg
              className="w-20 h-20 text-blue-500 mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              Process document for chat (Xử lý tài liệu để trò chuyện)
            </h3>
            <p className="text-gray-600 text-center mb-6 max-w-md">
              To enable AI-powered chat, we need to process this document first (Để bật trò chuyện với AI, cần xử lý tài liệu trước). This creates a
              knowledge base that allows you to ask questions and get accurate answers (Điều này tạo cơ sở kiến thức để bạn đặt câu hỏi và nhận câu trả lời chính xác).
            </p>
            <button
              onClick={handleIngest}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium flex items-center gap-2"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
              Process document (Xử lý tài liệu)
            </button>
            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                {error}
              </div>
            )}
          </div>
        )}

        {/* Ingesting State */}
        {isIngesting && (
          <div className="flex-1 flex flex-col items-center justify-center p-8">
            <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Processing document... (Đang xử lý tài liệu...)</h3>
            <p className="text-gray-600 text-center">
              Creating knowledge chunks and generating embeddings (Đang tạo đoạn kiến thức và embeddings). This may take 1-2 minutes (Có thể mất 1-2 phút).
            </p>
          </div>
        )}

        {/* Chat Interface */}
        {isIngested && (
          <>
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
              {messages.length === 0 && (
                <div className="text-center py-8">
                  <p className="text-gray-600 mb-6">
                    Ask me anything about this document (Hỏi bất kỳ điều gì về tài liệu này). Try these questions (Thử các câu hỏi sau):
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto">
                    {suggestedQuestions.map((question, idx) => (
                      <button
                        key={idx}
                        onClick={() => setInputMessage(question)}
                        className="p-3 bg-white border border-gray-300 rounded-lg hover:bg-blue-50 hover:border-blue-400 transition-all text-left text-sm text-gray-700"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((message, idx) => (
                <div
                  key={idx}
                  className={`flex ${
                    message.sender === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`max-w-[70%] rounded-lg p-3 ${
                      message.sender === 'user'
                        ? 'bg-blue-600 text-white'
                        : message.sender === 'system'
                        ? 'bg-yellow-50 text-yellow-900 border border-yellow-200'
                        : 'bg-white text-gray-900 shadow-sm border border-gray-200'
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      {message.sender === 'ai' && (
                        <svg
                          className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                          <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
                        </svg>
                      )}
                      <div className="flex-1">
                        <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                        <p className="text-xs opacity-70 mt-1">
                          {message.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white rounded-lg p-3 shadow-sm border border-gray-200">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                        <div
                          className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"
                          style={{ animationDelay: '0.1s' }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"
                          style={{ animationDelay: '0.2s' }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500">AI is thinking... (AI đang suy nghĩ...)</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-gray-200 bg-white">
              <div className="flex gap-2">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask a question about this document... (Hỏi về tài liệu này...)"
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                  rows="2"
                  disabled={isLoading}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isLoading || !inputMessage.trim()}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {isLoading ? (
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  ) : (
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                      />
                    </svg>
                  )}
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Press Enter to send, Shift+Enter for new line (Nhấn Enter để gửi, Shift+Enter để xuống dòng)
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default ChatPanel;
