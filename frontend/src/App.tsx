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

  // 파일 트리 상태
  const [fileTree] = useState<FileNode[]>([
    {
      name: 'easy_quality',
      type: 'folder',
      expanded: true,
      children: [
        { name: '__pycache__', type: 'folder', icon: '📁' },
        { name: '.vscode', type: 'folder', icon: '📁' },
        { name: 'chroma_db', type: 'folder', icon: '📁' },
        { name: 'frontend', type: 'folder', icon: '📁' },
        { name: 'rag', type: 'folder', icon: '📁' },
        { name: '.gitignore', type: 'file', icon: '📄' },
        { name: 'agent_logic_manual.md', type: 'file', icon: '📝' },
        { name: 'main.py', type: 'file', icon: '🐍' },
        { name: 'RAGLOGIC.md', type: 'file', icon: '📝' },
        { name: 'README.md', type: 'file', icon: '📝' },
        { name: 'requirements.txt', type: 'file', icon: '📄' },
      ],
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
          setAgentStatus('백엔드 연결됨')

          try {
            const agentResponse = await fetch(`${API_URL}/agent/status`)
            if (agentResponse.ok) {
              const data = await agentResponse.json()
              setAgentStatus(data.agent_available ? '에이전트 준비됨' : '일반 채팅 모드')
            }
          } catch {
            setAgentStatus('일반 채팅 모드')
          }
        } else {
          setIsConnected(false)
          setAgentStatus('백엔드 연결 실패')
        }
      } catch (error) {
        setIsConnected(false)
        setAgentStatus('백엔드 서버에 연결할 수 없습니다')
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
      let response = await fetch(`${API_URL}/agent/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageToSend,
          session_id: sessionId,
          llm_model: 'glm-4.7-flash',
          embedding_model: 'multilingual-e5-small',
          use_langgraph: true,
        }),
      })

      if (response.status === 404) {
        console.log('에이전트 API 없음, 테스트 에코 API 사용')
        response = await fetch(`${API_URL}/test/echo`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: messageToSend,
          }),
        })
      }

      const thinkingTime = Math.floor((Date.now() - startTime) / 1000)

      if (response.ok) {
        const data = await response.json()

        // 디버깅: 받은 데이터 확인
        console.log('백엔드 응답 데이터:', data)

        if (!sessionId) {
          setSessionId(data.session_id)
        }

        // tool_call 태그 제거 및 정리
        let cleanedAnswer = data.answer || data.message || '응답을 받았습니다.'

        console.log('원본 답변:', cleanedAnswer)

        // <tool_call>...</tool_call> 태그 제거
        cleanedAnswer = cleanedAnswer.replace(/<tool_call>.*?<\/tool_call>/gs, '')

        // <arg_key>, <arg_value> 등의 태그 제거
        cleanedAnswer = cleanedAnswer.replace(/<\/?[^>]+(>|$)/g, '')

        // 앞뒤 공백 제거
        cleanedAnswer = cleanedAnswer.trim()

        console.log('정리된 답변:', cleanedAnswer)

        // 답변이 비어있으면 기본 메시지 사용
        if (!cleanedAnswer) {
          cleanedAnswer = '답변을 처리하는 중 문제가 발생했습니다. 백엔드 로그를 확인해주세요.'
        }

        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: cleanedAnswer,
          timestamp: new Date(),
          thoughtProcess: '질문을 분석하고 관련 문서를 검색했습니다.',
          thinkingTime: thinkingTime,
        }

        setMessages(prev => [...prev, assistantMessage])
      } else {
        const error = await response.json()
        const errorMessage: ChatMessage = {
          role: 'assistant',
          content: `오류가 발생했습니다: ${error.detail || error.message}`,
          timestamp: new Date(),
        }
        setMessages(prev => [...prev, errorMessage])
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `네트워크 오류: ${error}. 백엔드가 http://localhost:8000에서 실행 중인지 확인해주세요.`,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
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
          <span className="project-name">easy_quality</span>
        </div>
        <div className="header-center">
          <span className="header-action">Open Agent Manager</span>
        </div>
        <div className="header-right">
          <button className="icon-btn">🔍</button>
          <button className="icon-btn">⚙️</button>
          <button className="icon-btn">🎨</button>
        </div>
      </header>

      <div className="main-container">
        {/* 왼쪽: Explorer */}
        <aside className="explorer">
          <div className="explorer-header">
            <span className="explorer-title">Explorer</span>
            <button className="icon-btn-small">⋯</button>
          </div>
          <div className="file-tree">
            {renderFileTree(fileTree)}
          </div>
          <div className="explorer-footer">
            <button className="footer-btn">📋 Outline</button>
            <button className="footer-btn">⏱️ Timeline</button>
          </div>
        </aside>

        {/* 가운데: 문서 뷰어 */}
        <main className="document-viewer">
          {selectedDocument ? (
            <div className="document-content">
              <div className="document-header">
                <h2>📄 {selectedDocument}</h2>
              </div>
              <div className="document-body">
                <p className="placeholder-text">문서 내용이 여기에 표시됩니다.</p>
                <p className="placeholder-text">선택한 파일: {selectedDocument}</p>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">📄</div>
              <h2>문서를 선택하세요</h2>
              <p>왼쪽 Explorer에서 파일을 선택하면 내용이 표시됩니다.</p>
            </div>
          )}
        </main>

        {/* 오른쪽: Agent 패널 */}
        <aside className="agent-panel">
          <div className="agent-header">
            <span className="agent-title">Agent</span>
            <div className="agent-controls">
              <button className="icon-btn-small">➕</button>
              <button className="icon-btn-small">🔄</button>
              <button className="icon-btn-small">⋯</button>
              <button className="icon-btn-small">✕</button>
            </div>
          </div>

          <div className="agent-content">
            {/* 상태 표시 */}
            <div className="agent-status-bar">
              <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
                {isConnected ? '🟢' : '🔴'}
              </span>
              <span className="status-text">{agentStatus}</span>
            </div>

            {/* 채팅 영역 */}
            <div className="agent-messages-container">
              {messages.map((msg, index) => (
                <div key={index} className={`agent-conversation ${msg.role}`}>
                  {msg.role === 'user' ? (
                    <div className="user-input-display">
                      <div className="user-input-header">
                        <span className="user-input-icon">💬</span>
                        <span className="user-input-text">{msg.content}</span>
                        <button className="undo-btn">↶</button>
                      </div>
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
                            <span className="thought-title">Thought Process</span>
                          </div>
                          {expandedSections.has(`thought-${index}`) && (
                            <div className="thought-content">
                              {msg.thoughtProcess}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Thinking Time */}
                      {msg.thinkingTime && (
                        <div className="thought-section">
                          <div
                            className="thought-header"
                            onClick={() => toggleSection(`time-${index}`)}
                          >
                            <span className="chevron">
                              {expandedSections.has(`time-${index}`) ? '▼' : '▶'}
                            </span>
                            <span className="thought-title">Thought for {msg.thinkingTime}s</span>
                          </div>
                        </div>
                      )}

                      {/* 답변 본문 */}
                      <div className="response-body">
                        {msg.content}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="agent-conversation assistant">
                  <div className="assistant-response">
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

            {/* 하단 입력 영역 */}
            <div className="agent-input-area">
              <div className="input-wrapper">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask anything (⌘L), @ to mention, / for workflow"
                  className="agent-input"
                  rows={1}
                />
              </div>
              <div className="input-actions">
                <button className="action-btn">➕</button>
                <button className="action-btn">📋 Planning</button>
                <button className="action-btn">⚡ Gemini 3 Flash</button>
                <button className="action-btn">🎤</button>
                <button
                  className="action-btn send-btn"
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

      {/* 하단 상태바 */}
      <footer className="statusbar">
        <div className="statusbar-left">
          <span className="status-item">⚡ main*</span>
          <span className="status-item">🔄</span>
          <span className="status-item">⊘ 0 ⚠ 0</span>
        </div>
        <div className="statusbar-right">
          <span className="status-item">Antigravity - Settings</span>
          <span className="status-item">🔔</span>
        </div>
      </footer>
    </div>
  )
}

export default App
