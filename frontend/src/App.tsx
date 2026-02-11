import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import MermaidRenderer from './components/MermaidRenderer'
import Sidebar from './components/Sidebar'
import DocumentManagementPanel from './components/DocumentManagementPanel'
import DocumentVisualizationPanel from './components/DocumentVisualizationPanel'
import './App.css'

// ═══════════════════════════════════════════════════════════════════════════
// 타입 정의
// ═══════════════════════════════════════════════════════════════════════════

interface RDBVerification {
  has_citations: boolean
  total_citations: number
  verified_citations: number
  incorrect_citations: string[]
  accuracy_rate: number
  verification_details: string
}

interface EvaluationScore {
  score: number
  reasoning: string
  rdb_verification?: RDBVerification
}

interface EvaluationScores {
  faithfulness?: EvaluationScore
  groundness?: EvaluationScore
  relevancy?: EvaluationScore
  correctness?: EvaluationScore
  average_score?: number
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  thoughtProcess?: string
  thinkingTime?: number
  evaluation_scores?: EvaluationScores
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
  const [documentContent, setDocumentContent] = useState<string | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())
  const [activePanel, setActivePanel] = useState<'documents' | 'visualization' | null>(null)

  // @멘션 상태
  const [docNames, setDocNames] = useState<{ id: number; name: string }[]>([])
  const [suggestions, setSuggestions] = useState<{ id: number; name: string }[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [suggestionIndex, setSuggestionIndex] = useState(0)
  const [mentionTriggerPos, setMentionTriggerPos] = useState<number | null>(null)

  // 파일 트리 상태 제거 (문서 관리 패널로 이동됨)

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

  // 문서 이름 목록 가져오기
  useEffect(() => {
    const fetchDocNames = async () => {
      try {
        const response = await fetch(`${API_URL}/rag/doc-names`)
        const data = await response.json()
        if (data.doc_names) {
          setDocNames(data.doc_names)
        }
      } catch (error) {
        console.error('Failed to fetch doc names:', error)
      }
    }
    fetchDocNames()
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
          llm_model: 'gpt-4o-mini', // OpenAI 모델
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

        // 디버깅: evaluation_scores 확인
        console.log('🔍 Evaluation Scores:', data.evaluation_scores)

        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: answer,
          timestamp: new Date(),
          thoughtProcess: thought,
          thinkingTime: thinkingTime,
          evaluation_scores: data.evaluation_scores,
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

  // handleUpload 제거 (DocumentManagementPanel로 이동됨)

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      if (showSuggestions && suggestions.length > 0) {
        e.preventDefault()
        selectSuggestion(suggestions[suggestionIndex].name)
      } else {
        e.preventDefault()
        sendMessage()
      }
    } else if (showSuggestions) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSuggestionIndex(prev => (prev + 1) % suggestions.length)
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSuggestionIndex(prev => (prev - 1 + suggestions.length) % suggestions.length)
      } else if (e.key === 'Escape') {
        setShowSuggestions(false)
      }
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value
    const cursorPos = e.target.selectionStart
    setInputMessage(value)

    const lastAtPos = value.lastIndexOf('@', cursorPos - 1)
    if (lastAtPos !== -1) {
      const textAfterAt = value.substring(lastAtPos + 1, cursorPos)
      if (!textAfterAt.includes(' ')) {
        const filtered = docNames.filter(doc =>
          doc.name.toLowerCase().includes(textAfterAt.toLowerCase())
        )
        setSuggestions(filtered)
        setShowSuggestions(filtered.length > 0)
        setSuggestionIndex(0)
        setMentionTriggerPos(lastAtPos)
        return
      }
    }
    setShowSuggestions(false)
  }

  const selectSuggestion = (name: string) => {
    if (mentionTriggerPos !== null) {
      const before = inputMessage.substring(0, mentionTriggerPos)
      const input = document.querySelector('.agent-input') as HTMLTextAreaElement
      const currentPos = input?.selectionStart || mentionTriggerPos + 1
      const afterAt = inputMessage.substring(currentPos)

      const newValue = before + name + ' ' + (afterAt.startsWith(' ') ? afterAt.substring(1) : afterAt)
      setInputMessage(newValue)
      setShowSuggestions(false)
      setMentionTriggerPos(null)
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

  // renderFileTree 제거 (필요 없음)

  // ─────────────────────────────────────────────────────────────
  // 문서 선택 핸들러
  // ─────────────────────────────────────────────────────────────

  const handleDocumentSelect = async (docId: string, content?: string) => {
    setSelectedDocument(docId)
    if (content) {
      setDocumentContent(content)
    } else {
      // 내용이 제공되지 않으면 API에서 가져오기
      try {
        const response = await fetch(`${API_URL}/rag/document/${docId}/content`)
        const data = await response.json()
        console.log('📄 [Document API Response]', data)

        // 원본 마크다운 content를 그대로 사용 (마크다운 렌더링은 JSX에서 처리)
        if (data.content) {
          setDocumentContent(data.content)
        } else {
          setDocumentContent('내용을 불러올 수 없습니다.')
        }
      } catch (error) {
        setDocumentContent('문서 내용을 가져오는 중 오류가 발생했습니다.')
      }
    }
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
            {isConnected ? '[OK]' : '[ERROR]'} {agentStatus}
          </span>
        </div>
      </header>

      <div className="main-container">
        {/* 왼쪽: 사이드바 아이콘 */}
        <Sidebar activePanel={activePanel} onPanelChange={setActivePanel} />

        {/* 문서 관리 패널 */}
        {activePanel === 'documents' && (
          <DocumentManagementPanel onDocumentSelect={handleDocumentSelect} />
        )}

        {/* 문서 시각화 패널 */}
        {activePanel === 'visualization' && <DocumentVisualizationPanel />}

        {/* 가운데: 문서 뷰어 */}
        <main className="document-viewer">
          {selectedDocument && documentContent ? (
            <div className="document-content">
              <div className="document-header">
                <h2>{selectedDocument}</h2>
              </div>
              <div className="document-body">
                <pre className="document-text-plain">{documentContent}</pre>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">[FILE]</div>
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
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p({ children }) {
                              // 문서 ID 패턴 (예: EQ-SOP-00001, EQ-WI-00012) 감지하여 클릭 가능한 링크로 변환
                              const docPattern = /(EQ-(?:SOP|WI)-\d{5}(?:\([\d.,\s]+\))?)/g;

                              const processText = (text: string) => {
                                const parts = text.split(docPattern);
                                return parts.map((part, i) => {
                                  if (docPattern.test(part)) {
                                    // 상세 번호(괄호 안) 제외하고 순수 ID만 추출
                                    const docId = part.split('(')[0];
                                    return (
                                      <span
                                        key={i}
                                        className="doc-link"
                                        onClick={() => handleDocumentSelect(docId)}
                                      >
                                        {part}
                                      </span>
                                    );
                                  }
                                  return part;
                                });
                              };

                              const recurse = (node: any): any => {
                                if (typeof node === 'string') return processText(node);
                                if (Array.isArray(node)) return node.map(recurse);
                                if (node?.props?.children) {
                                  return { ...node, props: { ...node.props, children: recurse(node.props.children) } };
                                }
                                return node;
                              };

                              return <p>{recurse(children)}</p>;
                            },
                            code({ node, inline, className, children, ...props }: any) {
                              const match = /language-(\w+)/.exec(className || '')
                              const language = match ? match[1] : ''

                              if (!inline && language === 'mermaid') {
                                return <MermaidRenderer chart={String(children).replace(/\n$/, '')} />
                              }

                              return !inline ? (
                                <pre className={className}>
                                  <code {...props}>{children}</code>
                                </pre>
                              ) : (
                                <code className={className} {...props}>
                                  {children}
                                </code>
                              )
                            }
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </div>

                      {msg.thinkingTime && (
                        <div className="meta-info">Time: {msg.thinkingTime}s</div>
                      )}

                      {/* 평가 점수 */}
                      {msg.evaluation_scores && (
                        <div className="evaluation-section">
                          <div
                            className="evaluation-header"
                            onClick={() => toggleSection(`eval-${index}`)}
                          >
                            <span className="chevron">
                              {expandedSections.has(`eval-${index}`) ? '▼' : '▶'}
                            </span>
                            <span className="evaluation-title">
                              🔍 평가 점수
                              {msg.evaluation_scores.average_score && (
                                <span className="eval-average"> ({msg.evaluation_scores.average_score.toFixed(1)}/5.0)</span>
                              )}
                            </span>
                          </div>
                          {expandedSections.has(`eval-${index}`) && (
                            <div className="evaluation-content">
                              {msg.evaluation_scores.faithfulness && (
                                <div className="eval-metric">
                                  <span className="eval-label">충실성 (Faithfulness):</span>
                                  <span className={`eval-score score-${msg.evaluation_scores.faithfulness.score}`}>
                                    {msg.evaluation_scores.faithfulness.score}/5
                                  </span>
                                  <div className="eval-reasoning">{msg.evaluation_scores.faithfulness.reasoning}</div>
                                  {msg.evaluation_scores.faithfulness.rdb_verification && (
                                    <div className="rdb-verification">
                                      <div className="rdb-header">📊 RDB 검증 결과</div>
                                      <div className="rdb-stats">
                                        <span className="rdb-stat">
                                          정확도: <strong>{msg.evaluation_scores.faithfulness.rdb_verification.accuracy_rate}%</strong>
                                        </span>
                                        <span className="rdb-stat">
                                          검증됨: {msg.evaluation_scores.faithfulness.rdb_verification.verified_citations}/{msg.evaluation_scores.faithfulness.rdb_verification.total_citations}
                                        </span>
                                      </div>
                                      {msg.evaluation_scores.faithfulness.rdb_verification.incorrect_citations.length > 0 && (
                                        <div className="rdb-errors">
                                          <strong>⚠️ 틀린 인용:</strong>
                                          <ul>
                                            {msg.evaluation_scores.faithfulness.rdb_verification.incorrect_citations.map((citation, i) => (
                                              <li key={i}>{citation}</li>
                                            ))}
                                          </ul>
                                        </div>
                                      )}
                                      <details className="rdb-details">
                                        <summary>상세 검증 결과</summary>
                                        <pre>{msg.evaluation_scores.faithfulness.rdb_verification.verification_details}</pre>
                                      </details>
                                    </div>
                                  )}
                                </div>
                              )}
                              {msg.evaluation_scores.groundness && (
                                <div className="eval-metric">
                                  <span className="eval-label">근거성 (Groundness):</span>
                                  <span className={`eval-score score-${msg.evaluation_scores.groundness.score}`}>
                                    {msg.evaluation_scores.groundness.score}/5
                                  </span>
                                  <div className="eval-reasoning">{msg.evaluation_scores.groundness.reasoning}</div>
                                  {msg.evaluation_scores.groundness.rdb_verification && (
                                    <div className="rdb-verification">
                                      <div className="rdb-stats">
                                        <span className="rdb-stat">
                                          정확도: <strong>{msg.evaluation_scores.groundness.rdb_verification.accuracy_rate}%</strong>
                                        </span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                              {msg.evaluation_scores.relevancy && (
                                <div className="eval-metric">
                                  <span className="eval-label">관련성 (Relevancy):</span>
                                  <span className={`eval-score score-${msg.evaluation_scores.relevancy.score}`}>
                                    {msg.evaluation_scores.relevancy.score}/5
                                  </span>
                                  <div className="eval-reasoning">{msg.evaluation_scores.relevancy.reasoning}</div>
                                </div>
                              )}
                              {msg.evaluation_scores.correctness && (
                                <div className="eval-metric">
                                  <span className="eval-label">정확성 (Correctness):</span>
                                  <span className={`eval-score score-${msg.evaluation_scores.correctness.score}`}>
                                    {msg.evaluation_scores.correctness.score}/5
                                  </span>
                                  <div className="eval-reasoning">{msg.evaluation_scores.correctness.reasoning}</div>
                                  {msg.evaluation_scores.correctness.rdb_verification && (
                                    <div className="rdb-verification">
                                      <div className="rdb-stats">
                                        <span className="rdb-stat">
                                          정확도: <strong>{msg.evaluation_scores.correctness.rdb_verification.accuracy_rate}%</strong>
                                        </span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
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
                  onChange={handleInputChange}
                  onKeyDown={handleKeyPress}
                  placeholder="Ask the Agent..."
                  className="agent-input"
                  rows={1}
                />
                {showSuggestions && (
                  <div className="suggestion-list">
                    {suggestions.map((doc, idx) => (
                      <div
                        key={doc.id}
                        className={`suggestion-item ${idx === suggestionIndex ? 'active' : ''}`}
                        onClick={() => selectSuggestion(doc.name)}
                      >
                        {doc.name}
                      </div>
                    ))}
                  </div>
                )}
                <button
                  className="send-btn"
                  onClick={sendMessage}
                  disabled={isLoading || !inputMessage.trim()}
                >
                  &gt;
                </button>
              </div>
            </div>
          </div>
        </aside>
      </div>

      {/* 업로드 모달 제거 (DocumentManagementPanel로 이동됨) */}
    </div>
  )
}

export default App
