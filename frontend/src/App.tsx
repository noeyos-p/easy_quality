import { useState, useRef, useEffect } from 'react'
import './App.css'

// ═══════════════════════════════════════════════════════════════════════════
// 타입 정의
// ═══════════════════════════════════════════════════════════════════════════

interface MetadataDisplay {
  doc_name?: string
  doc_title?: string
  sop_id?: string
  version?: string
  section?: string
  section_path?: string           // 🔥 "5 > 5.1 > 5.1.1"
  section_path_readable?: string  // 🔥 "5 절차 > 5.1 문서체계 > 5.1.1 Level 1"
  title?: string
  page?: string
}

interface Source {
  text: string
  similarity: number
  metadata: Record<string, any>
  metadata_display?: MetadataDisplay
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  timestamp: Date
}

interface DocumentInfo {
  doc_name: string
  doc_title?: string
  chunk_count: number
  chunk_method?: string
}

const API_URL = 'http://localhost:8000'

// ═══════════════════════════════════════════════════════════════════════════
// 유틸리티
// ═══════════════════════════════════════════════════════════════════════════

const getSimilarityColor = (score: number) => {
  if (score >= 0.7) return '#22c55e'
  if (score >= 0.5) return '#eab308'
  if (score >= 0.3) return '#f97316'
  return '#ef4444'
}

// ═══════════════════════════════════════════════════════════════════════════
// 메인 컴포넌트
// ═══════════════════════════════════════════════════════════════════════════

function App() {
  // 채팅 상태
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)

  // 문서 상태
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [uploadStatus, setUploadStatus] = useState('')
  const [uploadLoading, setUploadLoading] = useState(false)

  // 🔥 마크다운 미리보기 상태
  const [previewMarkdown, setPreviewMarkdown] = useState('')
  const [previewFilename, setPreviewFilename] = useState('')
  const [showPreview, setShowPreview] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)

  // 설정 상태
  const [showSettings, setShowSettings] = useState(false)
  const [showSources, setShowSources] = useState(true)
  const [embeddingModel, setEmbeddingModel] = useState('multilingual-e5-small')
  const [llmModel, setLlmModel] = useState('glm-4.7-flash')
  const [llmBackend, setLlmBackend] = useState('zai') // 🔥 백엔드 상태 추가
  const [chunkMethod, setChunkMethod] = useState('article')
  const [nResults, setNResults] = useState(7)  // 🔥 참고 문서 수 (3 -> 7 상향)
  const [agentMode, setAgentMode] = useState(false)  // 🤖 에이전트 모드

  // 소스 확장 상태
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())

  const chatEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchDocuments()
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ─────────────────────────────────────────────────────────────
  // API 호출
  // ─────────────────────────────────────────────────────────────

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/rag/documents?collection=documents`)
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
      }
    } catch {
      console.error('문서 목록 로드 실패')
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploadLoading(true)
    setUploadStatus('업로드 중...')

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('collection', 'documents')
      formData.append('chunk_method', chunkMethod)
      formData.append('model', embeddingModel)
      formData.append('exclude_intro', 'true')  // 🔥 v6.3: intro 블록 제외

      const response = await fetch(`${API_URL}/rag/upload`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        setUploadStatus(`✅ ${data.filename} 업로드 완료 (${data.chunks}개 청크)`)
        fetchDocuments()
      } else {
        const error = await response.json()
        setUploadStatus(`❌ 업로드 실패: ${error.detail}`)
      }
    } catch (error) {
      setUploadStatus(`❌ 업로드 오류: ${error}`)
    } finally {
      setUploadLoading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handlePreviewMarkdown = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setPreviewLoading(true)
    setPreviewFilename(file.name)
    setShowPreview(true)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/rag/preview-markdown`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        setPreviewMarkdown(data.markdown)
      } else {
        const error = await response.json()
        setPreviewMarkdown(`❌ 변환 실패: ${error.detail}`)
      }
    } catch (error) {
      setPreviewMarkdown(`❌ 오류: ${error}`)
    } finally {
      setPreviewLoading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleDeleteDocument = async (docName: string) => {
    if (!confirm(`"${docName}" 문서를 삭제하시겠습니까?`)) return

    try {
      const response = await fetch(`${API_URL}/rag/document`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_name: docName, collection: 'documents' }),
      })

      if (response.ok) {
        fetchDocuments()
      }
    } catch (error) {
      console.error('삭제 오류:', error)
    }
  }

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      // 🤖 에이전트 모드 vs 일반 RAG 분기
      const endpoint = agentMode ? `${API_URL}/agent/chat` : `${API_URL}/chat`
      const requestBody = agentMode
        ? {
          message: inputMessage,
          session_id: sessionId,
          llm_model: llmModel,
          use_langgraph: true,
        }
        : {
          message: inputMessage,
          session_id: sessionId,
          embedding_model: embeddingModel,
          llm_model: llmModel,
          llm_backend: llmBackend, // 🔥 백엔드 정보 포함
          include_sources: showSources,
          n_results: nResults,
        }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })

      if (response.ok) {
        const data = await response.json()

        if (!sessionId) {
          setSessionId(data.session_id)
        }

        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: data.answer,
          sources: agentMode ? [] : data.sources,  // 에이전트 모드는 sources 없음
          timestamp: new Date(),
        }

        setMessages(prev => [...prev, assistantMessage])
      } else {
        const error = await response.json()
        const errorMessage: ChatMessage = {
          role: 'assistant',
          content: `오류가 발생했습니다: ${error.detail}`,
          timestamp: new Date(),
        }
        setMessages(prev => [...prev, errorMessage])
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `네트워크 오류가 발생했습니다. 서버 연결을 확인해주세요.`,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const clearChat = async () => {
    if (sessionId) {
      try {
        await fetch(`${API_URL}/chat/history/${sessionId}`, { method: 'DELETE' })
      } catch { }
    }
    setMessages([])
    setSessionId(null)
    setExpandedSources(new Set())
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const toggleSourceExpand = (index: number) => {
    const newSet = new Set(expandedSources)
    if (newSet.has(index)) {
      newSet.delete(index)
    } else {
      newSet.add(index)
    }
    setExpandedSources(newSet)
  }

  // ─────────────────────────────────────────────────────────────
  // 렌더링 헬퍼
  // ─────────────────────────────────────────────────────────────

  const renderSource = (source: Source, index: number, messageIndex: number) => {
    const globalIndex = messageIndex * 100 + index
    const isExpanded = expandedSources.has(globalIndex)
    const meta = source.metadata_display || {}

    return (
      <div key={index} className="source-item">
        <div className="source-header" onClick={() => toggleSourceExpand(globalIndex)}>
          <div className="source-info">
            <span className="source-doc">📄 {meta.doc_name || '문서'}</span>
            {meta.sop_id && <span className="source-sop">{meta.sop_id}</span>}
            {meta.section && <span className="source-section">{meta.section}</span>}
          </div>
          <div className="source-meta">
            <span
              className="similarity-badge"
              style={{ backgroundColor: getSimilarityColor(source.similarity) }}
            >
              {(source.similarity * 100).toFixed(0)}%
            </span>
            <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
          </div>
        </div>

        {/* 🔥 section_path를 헤더 바로 아래에 항상 표시 (펼치지 않아도) */}
        {(meta.section_path_readable || meta.section_path) && (
          <div className="section-path-preview">
            <span className="path-icon">📍</span>
            <span className="path-text">{meta.section_path_readable || meta.section_path}</span>
          </div>
        )}

        {isExpanded && (
          <div className="source-details">
            {meta.title && (
              <div className="source-title">
                <strong>제목:</strong> {meta.title}
              </div>
            )}

            <div className="source-text">{source.text}</div>

            {/* 전체 메타데이터 */}
            <details className="metadata-details">
              <summary>전체 메타데이터</summary>
              <pre>{JSON.stringify(source.metadata, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>
    )
  }

  // ─────────────────────────────────────────────────────────────
  // 렌더링
  // ─────────────────────────────────────────────────────────────

  return (
    <div className="app">
      {/* 헤더 */}
      <header className="header">
        <div className="header-left">
          <h1>🤖 SOP 챗봇 <small>v11.0</small></h1>
          {agentMode && <span className="agent-badge">Agent</span>}
        </div>
        <div className="header-right">
          <button
            className="settings-btn"
            onClick={() => setShowSettings(!showSettings)}
          >
            ⚙️ 설정
          </button>
        </div>
      </header>

      <div className="main-container">
        {/* 사이드바 */}
        <aside className={`sidebar ${showSettings ? 'show' : ''}`}>
          {/* 문서 업로드 */}
          <section className="sidebar-section">
            <h3>📁 문서 업로드</h3>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.doc,.txt,.md,.html"
              onChange={handleFileUpload}
              disabled={uploadLoading}
              className="file-input"
            />
            {uploadStatus && <p className="upload-status">{uploadStatus}</p>}

            {/* 🔥 마크다운 미리보기 */}
            <div className="preview-section">
              <h4>🔍 마크다운 미리보기</h4>
              <input
                type="file"
                accept=".pdf,.docx,.doc,.txt,.md,.html"
                onChange={handlePreviewMarkdown}
                disabled={previewLoading}
                className="file-input"
              />
              {previewLoading && <p className="preview-status">변환 중...</p>}
            </div>
          </section>

          {/* 문서 목록 */}
          {documents.length > 0 && (
            <section className="sidebar-section">
              <h3>📚 문서 ({documents.length})</h3>
              <div className="doc-list">
                {documents.map((doc, i) => (
                  <div key={i} className="doc-item">
                    <div className="doc-info">
                      <span className="doc-name">{doc.doc_name}</span>
                      <span className="doc-chunks">{doc.chunk_count}청크</span>
                    </div>
                    <button
                      className="delete-btn"
                      onClick={() => handleDeleteDocument(doc.doc_name)}
                    >
                      🗑️
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 설정 */}
          <section className="sidebar-section">
            <h3>⚙️ 설정</h3>

            <div className="setting-group">
              <label>임베딩 모델</label>
              <select
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
              >
                <option value="multilingual-e5-small">E5-Small (경량)</option>
                <option value="ko-sbert">Ko-SBERT (한국어)</option>
                <option value="bge-m3">BGE-M3 (고성능)</option>
              </select>
            </div>

            <div className="setting-group">
              <label>LLM 모델</label>
              <select
                value={llmModel}
                onChange={(e) => {
                  setLlmModel(e.target.value);
                  // 모델에 따라 백엔드 자동 설정
                  if (e.target.value.includes('glm')) setLlmBackend('zai');
                  else if (e.target.value.includes(':')) setLlmBackend('ollama');
                  else setLlmBackend('hf');
                }}
              >
                <option value="glm-4.7-flash">GLM-4.7-Flash (Z.AI)</option>
                <option value="qwen2.5:3b">Qwen2.5-3B (Ollama)</option>
                <option value="qwen3:4b">Qwen3-4B (Ollama)</option>
              </select>
            </div>

            <div className="setting-group">
              <label>청킹 방식</label>
              <select
                value={chunkMethod}
                onChange={(e) => setChunkMethod(e.target.value)}
              >
                <option value="article">📜 조항 단위 (SOP 권장)</option>
                <option value="recursive">🔄 Recursive</option>
                <option value="sentence">📝 문장 단위</option>
                <option value="paragraph">📄 문단 단위</option>
              </select>
            </div>

            {!agentMode && (
              <div className="setting-group">
                <label>참고 문서 수</label>
                <select
                  value={nResults}
                  onChange={(e) => setNResults(Number(e.target.value))}
                >
                  <option value={1}>1개</option>
                  <option value={2}>2개</option>
                  <option value={3}>3개 (기본)</option>
                  <option value={5}>5개</option>
                  <option value={10}>10개</option>
                </select>
              </div>
            )}

            {!agentMode && (
              <div className="setting-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showSources}
                    onChange={(e) => setShowSources(e.target.checked)}
                  />
                  출처 표시
                </label>
              </div>
            )}

            {/* 🤖 에이전트 모드 토글 */}
            <div className="setting-group agent-toggle">
              <label className="toggle-label">
                <span className="toggle-text">
                  {agentMode ? '🤖 에이전트 모드' : '📄 일반 RAG'}
                </span>
                <div className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={agentMode}
                    onChange={(e) => setAgentMode(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </div>
              </label>
              <p className="toggle-description">
                {agentMode
                  ? 'LLM이 상황에 맞는 도구를 선택합니다 (LangSmith 추적)'
                  : '벡터 검색 → LLM 답변 (기본 방식)'}
              </p>
            </div>
          </section>
        </aside>

        {/* 채팅 영역 */}
        <main className="chat-area">
          {/* 채팅 메시지 */}
          <div className="messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <div className="welcome-icon">🤖</div>
                <h2>SOP 문서 챗봇에 오신 것을 환영합니다!</h2>
                <p>문서를 업로드하고 질문해보세요.</p>
                <div className="welcome-hints">
                  <div className="hint">📄 왼쪽에서 문서를 업로드하세요</div>
                  <div className="hint">💬 아래 입력창에 질문을 입력하세요</div>
                  <div className="hint">📍 section_path로 정확한 위치 확인!</div>
                </div>
              </div>
            ) : (
              messages.map((msg, msgIndex) => (
                <div key={msgIndex} className={`message ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? '👤' : '🤖'}
                  </div>
                  <div className="message-content">
                    <div className="message-text">{msg.content}</div>

                    {/* 출처 표시 */}
                    {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && showSources && (
                      <div className="sources">
                        <div className="sources-header">
                          📚 참고 문서 ({msg.sources.length})
                        </div>
                        {msg.sources.map((source, idx) =>
                          renderSource(source, idx, msgIndex)
                        )}
                      </div>
                    )}

                    <div className="message-time">
                      {msg.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className="message assistant loading">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* 입력 영역 */}
          <div className="input-area">
            <div className="input-container">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={documents.length > 0
                  ? "질문을 입력하세요... (Enter로 전송)"
                  : "먼저 문서를 업로드해주세요"}
                disabled={isLoading || documents.length === 0}
                rows={1}
              />
              <button
                className="send-btn"
                onClick={sendMessage}
                disabled={isLoading || !inputMessage.trim() || documents.length === 0}
              >
                {isLoading ? '⏳' : '📤'}
              </button>
            </div>

            <div className="input-actions">
              <button className="clear-btn" onClick={clearChat}>
                🗑️ 대화 초기화
              </button>
              {sessionId && (
                <span className="session-id">세션: {sessionId.slice(0, 8)}...</span>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* 🔥 마크다운 미리보기 모달 */}
      {showPreview && (
        <div className="modal-overlay" onClick={() => setShowPreview(false)}>
          <div className="modal-content markdown-preview" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>📄 마크다운 미리보기 - {previewFilename}</h2>
              <button className="close-btn" onClick={() => setShowPreview(false)}>✕</button>
            </div>
            <div className="modal-body">
              {previewLoading ? (
                <div className="loading">변환 중...</div>
              ) : (
                <pre className="markdown-content">{previewMarkdown}</pre>
              )}
            </div>
            <div className="modal-footer">
              <button
                className="download-btn"
                onClick={() => {
                  const blob = new Blob([previewMarkdown], { type: 'text/markdown' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `${previewFilename.replace(/\.[^/.]+$/, '')}.md`
                  a.click()
                  URL.revokeObjectURL(url)
                }}
                disabled={previewLoading || !previewMarkdown}
              >
                💾 다운로드
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App