import { useState, useRef, useEffect } from 'react'
import './App.css'

// ═══════════════════════════════════════════════════════════════════════════
// 타입 정의
// ═══════════════════════════════════════════════════════════════════════════

interface FileNode {
  name: string
  type: 'file' | 'folder'
  icon?: string
  children?: FileNode[]
  expanded?: boolean
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  thoughtProcess?: string
  thinkingTime?: number
}

const API_URL = 'http://localhost:8000'

// ═══════════════════════════════════════════════════════════════════════════
// 메인 컴포넌트
// ═══════════════════════════════════════════════════════════════════════════

function App() {
  // 채팅 상태
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [agentStatus, setAgentStatus] = useState<string>('연결 확인 중...')
  const [isConnected, setIsConnected] = useState(false)

  // UI 상태
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  // 파일 트리 상태 (데모 데이터)
  const [fileTree, setFileTree] = useState<FileNode[]>([
    {
      name: 'Uploaded Documents',
      type: 'folder',
      expanded: true,
      children: [], // 업로드된 파일이 여기 추가됨
    },
  ])

  const chatEndRef = useRef<HTMLDivElement>(null)

  // 백엔드 연결 상태 확인
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const healthResponse = await fetch(`${API_URL}/health`)
        if (healthResponse.ok) {
          setIsConnected(true)
          setAgentStatus('Agent Ready')
        } else {
          setIsConnected(false)
          setAgentStatus('Connection Failed')
        }
      } catch (error) {
        setIsConnected(false)
        setAgentStatus('Server Offline')
      }
    }

    checkBackendStatus()
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ─────────────────────────────────────────────────────────────
  // API 호출
  // ─────────────────────────────────────────────────────────────

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    const messageToSend = inputMessage
    setInputMessage('')
    setIsLoading(true)

    const startTime = Date.now()

    try {
      // 이제 RAG/일반 분기 없이 오직 Agent Chat만 호출
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageToSend,
          session_id: sessionId,
          llm_model: 'glm-4.7-flash', // 서브 에이전트용 기본값
        }),
      })

      const thinkingTime = Math.floor((Date.now() - startTime) / 1000)

      if (response.ok) {
        const data = await response.json()

        if (!sessionId) {
          setSessionId(data.session_id)
        }

        const answer = data.answer || "답변을 생성하지 못했습니다."

        // Agent 로그가 있으면 사고 과정으로 표시
        const thought = data.agent_log ? JSON.stringify(data.agent_log, null, 2) : "Agent reasoning..."

        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: answer,
          timestamp: new Date(),
          thoughtProcess: thought,
          thinkingTime: thinkingTime,
        }

        setMessages(prev => [...prev, assistantMessage])
      } else {
        const error = await response.json()
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `오류가 발생했습니다: ${error.detail}`,
          timestamp: new Date()
        }])
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `네트워크 오류: ${error}`,
        timestamp: new Date()
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleUpload = async () => {
    if (!uploadFile) return

    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', uploadFile)
    // 필요한 경우 추가 필드
    formData.append('chunk_size', '500')
    formData.append('use_langgraph', 'true')

    try {
      const response = await fetch(`${API_URL}/rag/upload`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        alert(`업로드 성공: ${data.filename} (${data.chunks} chunks)`)
        setIsUploadModalOpen(false)
        setUploadFile(null)

        // 파일 트리에 추가 (임시)
        setFileTree(prev => {
          const newTree = [...prev]
          if (newTree[0].children) {
            newTree[0].children.push({
              name: data.filename,
              type: 'file',
              icon: '📄'
            })
          }
          return newTree
        })
      } else {
        alert('업로드 실패')
      }
    } catch (error) {
      alert(`업로드 에러: ${error}`)
    } finally {
      setIsUploading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const toggleSection = (section: string) => {
    const newSet = new Set(expandedSections)
    if (newSet.has(section)) {
      newSet.delete(section)
    } else {
      newSet.add(section)
    }
    setExpandedSections(newSet)
  }

  // ─────────────────────────────────────────────────────────────
  // 렌더링 헬퍼
  // ─────────────────────────────────────────────────────────────

  const renderFileTree = (nodes: FileNode[], depth = 0) => {
    return nodes.map((node, index) => (
      <div key={index} className="tree-node">
        <div
          className="tree-item"
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
          onClick={() => {
            if (node.type === 'file') {
              setSelectedDocument(node.name)
            }
          }}
        >
          {node.type === 'folder' && (
            <span className="tree-chevron">{node.expanded ? '▼' : '▶'}</span>
          )}
          <span className="tree-icon">{node.icon || (node.type === 'folder' ? '📁' : '📄')}</span>
          <span className="tree-name">{node.name}</span>
        </div>
        {node.expanded && node.children && (
          <div className="tree-children">
            {renderFileTree(node.children, depth + 1)}
          </div>
        )}
      </div>
    ))
  }

  // ─────────────────────────────────────────────────────────────
  // 렌더링
  // ─────────────────────────────────────────────────────────────

  return (
    <div className="app">
      {/* 헤더 */}
      <header className="header">
        <div className="header-left">
          <span className="project-name">Orchestrator Agent</span>
        </div>
        <div className="header-right">
          <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢' : '🔴'} {agentStatus}
          </span>
        </div>
      </header>

      <div className="main-container">
        {/* 왼쪽: Explorer */}
        <aside className="explorer">
          <div className="explorer-header">
            <span className="explorer-title">Documents</span>
            <button
              className="icon-btn-small"
              onClick={() => setIsUploadModalOpen(true)}
              title="Upload Document"
            >
              ➕ Upload
            </button>
          </div>
          <div className="file-tree">
            {renderFileTree(fileTree)}
          </div>
        </aside>

        {/* 가운데: 문서 뷰어 (Optional) */}
        <main className="document-viewer">
          {selectedDocument ? (
            <div className="document-content">
              <div className="document-header">
                <h2>📄 {selectedDocument}</h2>
              </div>
              <div className="document-body">
                <p>문서 미리보기 기능은 준비 중입니다.</p>
                <p>선택된 파일: {selectedDocument}</p>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">📄</div>
              <h2>Select a document</h2>
            </div>
          )}
        </main>

        {/* 오른쪽: Agent 패널 */}
        <aside className="agent-panel">
          <div className="agent-header">
            <span className="agent-title">Agent Chat</span>
          </div>

          <div className="agent-content">
            {/* 채팅 영역 */}
            <div className="agent-messages-container">
              {messages.map((msg, index) => (
                <div key={index} className={`agent-conversation ${msg.role}`}>
                  {msg.role === 'user' ? (
                    <div className="user-input-display">
                      <span className="user-input-text">{msg.content}</span>
                    </div>
                  ) : (
                    <div className="assistant-response">
                      {/* Thought Process */}
                      {msg.thoughtProcess && (
                        <div className="thought-section">
                          <div
                            className="thought-header"
                            onClick={() => toggleSection(`thought-${index}`)}
                          >
                            <span className="chevron">
                              {expandedSections.has(`thought-${index}`) ? '▼' : '▶'}
                            </span>
                            <span className="thought-title">Show Reasoning</span>
                          </div>
                          {expandedSections.has(`thought-${index}`) && (
                            <pre className="thought-content">
                              {msg.thoughtProcess}
                            </pre>
                          )}
                        </div>
                      )}

                      {/* 답변 본문 */}
                      <div className="response-body">
                        {msg.content}
                      </div>

                      {msg.thinkingTime && (
                        <div className="meta-info">Time: {msg.thinkingTime}s</div>
                      )}
                    </div>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="agent-conversation assistant">
                  <div className="assistant-response">
                    <div className="typing-indicator">Processing request...</div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* 하단 입력 영역 */}
            <div className="agent-input-area">
              <div className="input-wrapper">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask the Agent..."
                  className="agent-input"
                  rows={1}
                />
                <button
                  className="send-btn"
                  onClick={sendMessage}
                  disabled={isLoading || !inputMessage.trim()}
                >
                  ➤
                </button>
              </div>
            </div>
          </div>
        </aside>
      </div>

      {/* 업로드 모달 */}
      {isUploadModalOpen && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h3>Upload Document</h3>
            <input
              type="file"
              onChange={(e) => setUploadFile(e.target.files ? e.target.files[0] : null)}
            />
            <div className="modal-actions">
              <button onClick={() => setIsUploadModalOpen(false)}>Cancel</button>
              <button onClick={handleUpload} disabled={!uploadFile || isUploading}>
                {isUploading ? 'Uploading...' : 'Upload'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
