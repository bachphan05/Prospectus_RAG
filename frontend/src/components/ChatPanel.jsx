import { useState, useEffect, useRef, useCallback } from 'react';
import api from '../services/api';

/**
 * Parses AI answer text and turns "Trang X" / "Page X" / "[Trang X, Y]" patterns
 * into clickable span elements. All other text is rendered as-is.
 */
function parsePageRefs(text, onPageClick) {
  // Matches: Trang 12, trang 12, Page 12, [Trang 12, 34], [Trang 12]
  const pattern = /(?:\[Trang\s+([\d,\s]+)\]|(?:Trang|trang|Page|page)\s+(\d+))/g;
  const parts = [];
  let last = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const raw = match[1] || match[2];
    const pages = raw.split(',').map(s => parseInt(s.trim(), 10)).filter(Boolean);
    pages.forEach((p, i) => {
      if (i > 0) parts.push(', ');
      parts.push(
        <button
          key={`${match.index}-${p}`}
          type="button"
          onClick={() => onPageClick(p)}
          className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700 hover:bg-blue-200 border border-blue-200 transition-colors cursor-pointer"
          title={`Open page ${p}`}
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          {i === 0 && pages.length > 1 ? `Trang ${p}` : `Trang ${p}`}
        </button>
      );
    });
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

/**
 * ChatPanel Component - AI-powered chat interface for documents
 */
function ChatPanel({ document, onClose }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isHistoryLoaded, setIsHistoryLoaded] = useState(false);
  const [isCheckingRagStatus, setIsCheckingRagStatus] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isIngested, setIsIngested] = useState(false);
  const [ragStatus, setRagStatus] = useState(null);
  const [ragProgress, setRagProgress] = useState(0);
  const [ragErrorMessage, setRagErrorMessage] = useState(null);
  const [error, setError] = useState(null);

  // Side-by-side PDF page viewer state
  const [activeCitation, setActiveCitation] = useState(null);  // { page, quote }
  const [pageContext, setPageContext] = useState(null);         // response from getPageContext
  const [loadingPageCtx, setLoadingPageCtx] = useState(false);
  const messagesEndRef = useRef(null);
  const hasLoadedHistoryRef = useRef(false);
  const saveDebounceRef = useRef(null);
  const historyLoadTokenRef = useRef(0);
  const ragStatusTokenRef = useRef(0);

  const loadChatHistory = useCallback(async () => {
    if (!document?.id) return;

    historyLoadTokenRef.current += 1;
    const token = historyLoadTokenRef.current;
    setIsLoadingHistory(true);
    setIsHistoryLoaded(false);

    try {
      const res = await api.getChatHistory(document.id);
      const saved = Array.isArray(res?.history) ? res.history : [];

      if (token !== historyLoadTokenRef.current) return;

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
      if (token === historyLoadTokenRef.current) {
        hasLoadedHistoryRef.current = true;
        setIsLoadingHistory(false);
        setIsHistoryLoaded(true);
      }
    }
  }, [document?.id]);

  const checkIngestionStatus = useCallback(async () => {
    if (!document?.id) return;

    ragStatusTokenRef.current += 1;
    const token = ragStatusTokenRef.current;
    setIsCheckingRagStatus(true);

    try {
      const result = await api.ragStatus(document.id);

      if (token !== ragStatusTokenRef.current) return;

      setRagStatus(result?.rag_status ?? null);
      setRagProgress(typeof result?.rag_progress === 'number' ? result.rag_progress : 0);
      setRagErrorMessage(result?.rag_error_message ?? null);

      if (result.is_ingested) {
        setIsIngested(true);
        setIsIngesting(false);
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
        return;
      }

      // Not ingested yet: reflect background RAG status.
      const status = result?.rag_status;
      if (status === 'queued' || status === 'running') {
        setIsIngested(false);
        setIsIngesting(true);
      } else {
        setIsIngesting(false);
        setIsIngested(false);
      }
    } catch (err) {
      console.error('Failed to check ingestion status:', err);
      setIsIngested(false);
      setIsIngesting(false);
      setRagStatus(null);
      setRagProgress(0);
      setRagErrorMessage(null);
    } finally {
      if (token === ragStatusTokenRef.current) {
        setIsCheckingRagStatus(false);
      }
    }
  }, [document?.id]);

  useEffect(() => {
    // Load persisted chat history (if any) then check ingestion.
    hasLoadedHistoryRef.current = false;
    setMessages([]);
    setError(null);
    setIsHistoryLoaded(false);
    setIsCheckingRagStatus(false);
    loadChatHistory();
    checkIngestionStatus();
  }, [document?.id, loadChatHistory, checkIngestionStatus]);

  useEffect(() => {
    // Poll RAG status while it is queued/running.
    if (!document?.id) return;
    if (!(ragStatus === 'queued' || ragStatus === 'running' || isIngesting)) return;

    const id = setInterval(() => {
      checkIngestionStatus();
    }, 5000);

    return () => clearInterval(id);
  }, [document?.id, ragStatus, isIngesting, checkIngestionStatus]);

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

  

  const handleIngest = async () => {
    setIsIngesting(true);
    setError(null);

    try {
      const result = await api.ingestForRag(document.id);
      setIsIngested(true);
      setIsIngesting(false);
      setRagStatus('completed');
      setRagProgress(100);
      setRagErrorMessage(null);
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
      setRagStatus('failed');
      setRagErrorMessage(err.message || 'Failed to process document');
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
        citations: Array.isArray(response.citations) ? response.citations : [],
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

  const openPageFromChat = async (pageNum, quote = '') => {
    if (!document?.id) return;
    setActiveCitation({ page: pageNum, quote });
    setLoadingPageCtx(true);
    setPageContext(null);
    try {
      const ctx = await api.getPageContext(document.id, pageNum, quote);
      setPageContext(ctx);
    } catch (err) {
      console.error('Failed to load page context:', err);
    } finally {
      setLoadingPageCtx(false);
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
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-[96vw] h-[88vh] flex flex-col">
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

        {/* Checking RAG Status (avoid flashing the manual button) */}
        {!isIngested && !isIngesting && isCheckingRagStatus && (
          <div className="flex-1 flex flex-col items-center justify-center p-8 bg-gray-50">
            <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Checking document status... (Đang kiểm tra trạng thái tài liệu...)
            </h3>
            <p className="text-gray-600 text-center max-w-md">
              Loading RAG status for chat (Đang tải trạng thái RAG để trò chuyện).
            </p>
          </div>
        )}

        {/* Ingestion Required (only when auto RAG not started or failed) */}
        {!isIngested && !isIngesting && !isCheckingRagStatus && (ragStatus === 'not_started' || ragStatus === 'failed' || ragStatus === null) && (
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
              {ragStatus === 'failed'
                ? 'RAG processing failed (Xử lý RAG thất bại)'
                : 'Process document for chat (Xử lý tài liệu để trò chuyện)'}
            </h3>
            <p className="text-gray-600 text-center mb-6 max-w-md">
              {ragStatus === 'failed' ? (
                <>
                  {ragErrorMessage || 'The document could not be processed for chat.'}
                  <br />
                  You can retry processing (Bạn có thể thử xử lý lại).
                </>
              ) : (
                <>
                  To enable AI-powered chat, we need to process this document first (Để bật trò chuyện với AI, cần xử lý tài liệu trước). This creates a
                  knowledge base that allows you to ask questions and get accurate answers (Điều này tạo cơ sở kiến thức để bạn đặt câu hỏi và nhận câu trả lời chính xác).
                </>
              )}
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
              {ragStatus === 'failed' ? 'Retry processing (Thử lại)' : 'Process document (Xử lý tài liệu)'}
            </button>
            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                {error}
              </div>
            )}
          </div>
        )}

        {/* Ingesting/Queued State (auto or manual) */}
        {isIngesting && (
          <div className="flex-1 flex flex-col items-center justify-center p-8">
            <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              {ragStatus === 'queued'
                ? 'Queued for processing... (Đang xếp hàng xử lý...)'
                : 'Processing document... (Đang xử lý tài liệu...)'}
            </h3>
            <p className="text-gray-600 text-center max-w-md">
              Creating knowledge chunks and generating embeddings (Đang tạo đoạn kiến thức và embeddings). This may take several minutes (Có thể mất vài phút).
            </p>

            <div className="w-full max-w-md mt-6">
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span>RAG progress</span>
                <span>{Math.min(Math.max(ragProgress || 0, 0), 100)}%</span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-2 bg-blue-600 transition-all"
                  style={{ width: `${Math.min(Math.max(ragProgress || 0, 0), 100)}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Chat Interface */}
        {isIngested && (
          <div className="flex-1 min-h-0 flex overflow-hidden">
            {/* ── Left: messages + input ── */}
            <div className={`flex flex-col min-h-0 ${activeCitation ? 'w-1/2 border-r border-gray-200' : 'w-full'}`}>
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
              {isLoadingHistory && !isHistoryLoaded && messages.length === 0 && (
                <div className="flex flex-col items-center justify-center py-12">
                  <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-3"></div>
                  <p className="text-sm text-gray-600">Loading chat history... (Đang tải lịch sử trò chuyện...)</p>
                </div>
              )}

              {messages.length === 0 && !isLoadingHistory && isHistoryLoaded && (
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
                        {message.sender === 'ai' ? (
                          <p className="text-sm whitespace-pre-wrap leading-relaxed">
                            {parsePageRefs(message.text, (p) => openPageFromChat(p,
                              /* pass the chunk quote whose page matches, if any */
                              (message.citations || []).find(c => c.page === p)?.quote || ''
                            ))}
                          </p>
                        ) : (
                          <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                        )}
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
            </div>{/* end left pane */}

            {/* ── Right: PDF page viewer ── */}
            {activeCitation && (
              <div className="w-1/2 flex flex-col min-h-0 bg-gray-50">
                {/* Panel header */}
                <div className="flex items-center justify-between px-4 py-3 bg-white border-b border-gray-200 shrink-0">
                  <div>
                    <p className="text-sm font-semibold text-gray-900">
                      Trang {activeCitation.page}
                    </p>
                    <p className="text-xs text-gray-500">Source page</p>
                  </div>
                  <button
                    onClick={() => { setActiveCitation(null); setPageContext(null); }}
                    className="p-1 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-700"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {/* Page image + highlights */}
                <div className="flex-1 min-h-0 overflow-auto p-4">
                  {loadingPageCtx && (
                    <div className="h-full flex items-center justify-center">
                      <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
                    </div>
                  )}

                  {!loadingPageCtx && pageContext?.image && (
                    <div className="space-y-3">
                      {/* Page with highlight overlay */}
                      <div className="relative bg-white rounded border border-gray-200 overflow-hidden shadow">
                        <img
                          src={pageContext.image}
                          alt={`Page ${activeCitation.page}`}
                          className="w-full h-auto block"
                        />
                        {/* Yellow highlight boxes over matched text */}
                        {Array.isArray(pageContext.matched_bboxes) && pageContext.matched_bboxes.map((bbox, idx) => {
                          const [ymin, xmin, ymax, xmax] = bbox;
                          return (
                            <div
                              key={idx}
                              className="absolute border-2 border-yellow-400 bg-yellow-300 bg-opacity-40 pointer-events-none"
                              style={{
                                top: `${ymin / 10}%`,
                                left: `${xmin / 10}%`,
                                width: `${(xmax - xmin) / 10}%`,
                                height: `${(ymax - ymin) / 10}%`,
                              }}
                            />
                          );
                        })}
                      </div>

                      {/* Snippet card */}
                      {activeCitation.quote && (
                        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                          <p className="text-xs font-semibold text-yellow-900 mb-1">Đoạn trích nguồn</p>
                          <p className="text-sm text-yellow-900 whitespace-pre-wrap leading-relaxed">
                            {activeCitation.quote}
                          </p>
                          {Array.isArray(pageContext.matched_bboxes) && pageContext.matched_bboxes.length === 0 && (
                            <p className="text-xs text-yellow-700 mt-2 italic">
                              Không tìm thấy vị trí chính xác trên trang; hiển thị đoạn trích gốc.
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {!loadingPageCtx && !pageContext?.image && (
                    <div className="h-full flex items-center justify-center text-sm text-gray-500">
                      Không thể tải trang.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div> /* end flex chat+pdf */
        )}
      </div>
    </div>
  );
}

export default ChatPanel;
