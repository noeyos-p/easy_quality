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
  metadata?: DocumentMetadata
}

interface DocumentMetadata {
  doc_id?: string
  sop_id?: string
  title?: string
  version?: string
  effective_date?: string
  owning_dept?: string
  total_chunks?: number
  quality_score?: number
  conversion_method?: string
}

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
  const [selectedDocMetadata, setSelectedDocMetadata] = useState<DocumentMetadata | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<string>('')

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

  const handleUpload = async () => {
    if (!uploadFile) return

    setIsUploading(true)
    setUploadProgress('파일 업로드 중...')
    const formData = new FormData()
    formData.append('file', uploadFile)
    formData.append('chunk_size', '500')
    formData.append('chunk_overlap', '50')
    formData.append('use_langgraph', 'true')
    formData.append('use_llm_metadata', 'true')

    try {
      const response = await fetch(`${API_URL}/rag/upload`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()

        // 메타데이터 구성
        const metadata: DocumentMetadata = {
          doc_id: data.metadata?.doc_id || data.sop_id,
          sop_id: data.sop_id,
          title: data.doc_title || data.filename,
          version: data.metadata?.version,
          effective_date: data.metadata?.effective_date,
          owning_dept: data.metadata?.owning_dept,
          total_chunks: data.chunks,
          quality_score: data.quality_score,
          conversion_method: data.conversion_method,
        }

        setUploadProgress(`[OK] 업로드 완료!\n파일: ${data.filename}\n청크: ${data.chunks}개\n품질: ${(data.quality_score * 100).toFixed(0)}%`)

        // 파일 트리에 추가
        setFileTree(prev => {
          const newTree = [...prev]
          if (newTree[0].children) {
            newTree[0].children.push({
              name: data.filename,
              type: 'file',
              icon: '[FILE]',
              metadata: metadata
            })
          }
          return newTree
        })

        setTimeout(() => {
          setIsUploadModalOpen(false)
          setUploadFile(null)
          setUploadProgress('')
        }, 2000)
      } else {
        const error = await response.json()
        setUploadProgress(`[ERROR] 업로드 실패: ${error.detail}`)
      }
    } catch (error) {
      setUploadProgress(`[ERROR] 네트워크 오류: ${error}`)
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
              setSelectedDocMetadata(node.metadata || null)
            }
          }}
        >
          {node.type === 'folder' && (
            <span className="tree-chevron">{node.expanded ? '▼' : '▶'}</span>
          )}
          <span className="tree-icon">{node.icon || (node.type === 'folder' ? '[DIR]' : '[FILE]')}</span>
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
            {isConnected ? '[OK]' : '[ERROR]'} {agentStatus}
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
              + Upload
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
                <h2>[FILE] {selectedDocument}</h2>
              </div>
              <div className="document-body">
                {selectedDocMetadata ? (
                  <div className="metadata-section">
                    <h3>[METADATA] 문서 메타데이터</h3>
                    <table className="metadata-table">
                      <tbody>
                        {selectedDocMetadata.doc_id && (
                          <tr>
                            <td className="label">문서 ID:</td>
                            <td className="value">{selectedDocMetadata.doc_id}</td>
                          </tr>
                        )}
                        {selectedDocMetadata.title && (
                          <tr>
                            <td className="label">제목:</td>
                            <td className="value">{selectedDocMetadata.title}</td>
                          </tr>
                        )}
                        {selectedDocMetadata.version && (
                          <tr>
                            <td className="label">버전:</td>
                            <td className="value">{selectedDocMetadata.version}</td>
                          </tr>
                        )}
                        {selectedDocMetadata.effective_date && (
                          <tr>
                            <td className="label">시행일:</td>
                            <td className="value">{selectedDocMetadata.effective_date}</td>
                          </tr>
                        )}
                        {selectedDocMetadata.owning_dept && (
                          <tr>
                            <td className="label">담당부서:</td>
                            <td className="value">{selectedDocMetadata.owning_dept}</td>
                          </tr>
                        )}
                        {selectedDocMetadata.total_chunks && (
                          <tr>
                            <td className="label">총 청크 수:</td>
                            <td className="value">{selectedDocMetadata.total_chunks}</td>
                          </tr>
                        )}
                        {selectedDocMetadata.quality_score !== undefined && (
                          <tr>
                            <td className="label">품질 점수:</td>
                            <td className="value">{(selectedDocMetadata.quality_score * 100).toFixed(0)}%</td>
                          </tr>
                        )}
                        {selectedDocMetadata.conversion_method && (
                          <tr>
                            <td className="label">변환 방법:</td>
                            <td className="value">{selectedDocMetadata.conversion_method}</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <>
                    <p>문서 미리보기 기능은 준비 중입니다.</p>
                    <p>선택된 파일: {selectedDocument}</p>
                  </>
                )}
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
                        {msg.content}
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
                  &gt;
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
            <h3>[UPLOAD] Upload Document</h3>
            <div className="upload-form">
              <input
                type="file"
                onChange={(e) => setUploadFile(e.target.files ? e.target.files[0] : null)}
                accept=".pdf,.docx,.doc,.html,.md,.txt"
                className="file-input"
              />



              {uploadProgress && (
                <div className="upload-progress">
                  <pre>{uploadProgress}</pre>
                </div>
              )}
            </div>

            <div className="modal-actions">
              <button onClick={() => {
                setIsUploadModalOpen(false)
                setUploadFile(null)
                setUploadProgress('')
              }}>Cancel</button>
              <button onClick={handleUpload} disabled={!uploadFile || isUploading}>
                {isUploading ? '[WAIT] Uploading...' : '[OK] Upload'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
