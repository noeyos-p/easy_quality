import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import MermaidRenderer from './components/MermaidRenderer'
import Sidebar from './components/Sidebar'
import DocumentManagementPanel from './components/DocumentManagementPanel'
import ForceGraph2D from 'react-force-graph-2d'

const SCORE_COLORS: Record<number, string> = {
  5: 'bg-[#22D142] text-black',
  4: 'bg-[#85E89D] text-black',
  3: 'bg-[#FFD700] text-black',
  2: 'bg-[#FFA500] text-black',
  1: 'bg-[#FF4444] text-white',
}

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
  const [isSaving, setIsSaving] = useState(false) // 저장 중 상태 추가

  // UI 상태
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)
  const [documentContent, setDocumentContent] = useState<string | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())
  const [activePanel, setActivePanel] = useState<'documents' | 'visualization' | null>(null)
  const [isLeftVisible, setIsLeftVisible] = useState(true)
  const [isRightVisible, setIsRightVisible] = useState(true)
  const [isEditing, setIsEditing] = useState(false)
  const [editedContent, setEditedContent] = useState<string>('')
  const [isDownloadOpen, setIsDownloadOpen] = useState(false) // 다운로드 드롭다운 상태

  // @멘션 상태
  const [docNames, setDocNames] = useState<{ id: number; name: string }[]>([])
  const [suggestions, setSuggestions] = useState<{ id: number; name: string }[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [suggestionIndex, setSuggestionIndex] = useState(0)
  const [mentionTriggerPos, setMentionTriggerPos] = useState<number | null>(null)
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [isDraggingOver, setIsDraggingOver] = useState(false)
  // 그래프 시각화 상태
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [graphData, setGraphData] = useState<{ nodes: any[], links: any[] } | null>(null)
  const [isLoadingGraph, setIsLoadingGraph] = useState(false)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null)
  const [graphSize, setGraphSize] = useState({ width: 0, height: 0 })
  const graphContainerRef = useRef<HTMLDivElement>(null)

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
      } catch (_error) {
        setIsConnected(false)
        setAgentStatus('Server Offline')
      }
    }

    checkBackendStatus()
  }, [])

  const fetchDocumentContent = async (docName: string, version?: string) => {
    setSelectedDocument(docName)
    setIsLoading(false)
    setIsEditing(false) // 편집 모드 해제
    setEditedContent('') // 편집 내용 초기화

    try {
      const url = version
        ? `${API_URL}/rag/document/${encodeURIComponent(docName)}/content?version=${encodeURIComponent(version)}`
        : `${API_URL}/rag/document/${encodeURIComponent(docName)}/content`

      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setDocumentContent(data.content)
        setEditedContent(data.content) // 초기 편집 내용 설정
        setSelectedDocument(docName)
        setIsEditing(false) // 문서 변경 시 편집 모드 초기화
      }
    } catch (error) {
      console.error('문서 내용 조회 실패:', error)
    }
  }

  const handleSaveDocument = async () => {
    if (!selectedDocument) return

    setIsSaving(true) // 로딩 시작
    try {
      const response = await fetch(`${API_URL}/rag/document/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_name: selectedDocument,
          content: editedContent
        })
      })

      if (response.ok) {
        const data = await response.json()
        setDocumentContent(editedContent)
        setIsEditing(false)
        alert(`문서가 저장되었습니다. (새 버전: ${data.version})`)
      } else {
        const errorData = await response.json()
        alert(`저장 실패: ${errorData.detail || '알 수 없는 오류'}`)
      }
    } catch (error) {
      console.error('문서 저장 중 오류 발생:', error)
      alert('저장 중 오류가 발생했습니다.')
    } finally {
      setIsSaving(false) // 로딩 종료
    }
  }

  const handleDownload = async (format: 'pdf' | 'docx' | 'md') => {
    if (!selectedDocument) return

    try {
      const url = `${API_URL}/rag/document/download/${encodeURIComponent(selectedDocument)}?format=${format}`
      const response = await fetch(url)

      if (response.ok) {
        const blob = await response.blob()
        const downloadUrl = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = downloadUrl

        const contentDisposition = response.headers.get('Content-Disposition')
        let fileName = `${selectedDocument}.${format}`
        if (contentDisposition && contentDisposition.includes('filename=')) {
          fileName = contentDisposition.split('filename=')[1].replace(/"/g, '')
        }

        a.download = fileName
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(downloadUrl)
        document.body.removeChild(a)
        setIsDownloadOpen(false) // 다운로드 후 닫기
      } else {
        const errorData = await response.json()
        alert(`다운로드 실패: ${errorData.detail || '알 수 없는 오류'}`)
      }
    } catch (error) {
      console.error(`${format} 다운로드 중 오류 발생:`, error)
      alert('다운로드 중 오류가 발생했습니다.')
    }
  }

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

  // 그래프 데이터 로드
  useEffect(() => {
    if (activePanel === 'visualization') {
      fetchGraphData()
    }
  }, [activePanel])

  // 그래프 컨테이너 크기 측정 (ResizeObserver로 CSS transition 완료 후 감지)
  useEffect(() => {
    if (!graphContainerRef.current) return

    const updateSize = () => {
      if (graphContainerRef.current) {
        const { offsetWidth, offsetHeight } = graphContainerRef.current
        setGraphSize({ width: offsetWidth, height: offsetHeight })
      }
    }

    const ro = new ResizeObserver(() => {
      updateSize()
      // 크기 변경 후 그래프 중앙 정렬
      setTimeout(() => fgRef.current?.zoomToFit(400, 80), 50)
    })
    ro.observe(graphContainerRef.current)
    updateSize()

    return () => ro.disconnect()
  }, [activePanel])

  const fetchGraphData = async () => {
    setIsLoadingGraph(true)
    try {
      const response = await fetch(`${API_URL}/graph/visualization/all`)
      const data = await response.json()
      console.log('📊 [Graph Data]', data)

      if (data.success) {
        // 노드를 원형으로 배치하고 위치 고정
        const nodeCount = data.nodes.length
        const radius = Math.min(120, nodeCount * 12) // 화면에 맞게 반지름 더욱 줄임

        const nodesWithPosition = data.nodes.map((node: any, i: number) => {
          const angle = (i / nodeCount) * 2 * Math.PI
          const x = Math.cos(angle) * radius
          const y = Math.sin(angle) * radius - 40  // 위로 40px 이동 (가운데 정렬)

          return {
            id: node.id,
            name: node.id,
            title: node.title,
            version: node.version,
            doc_type: node.doc_type,
            type_name: node.type_name,
            x: x,
            y: y,
            fx: x,  // fixed x position
            fy: y   // fixed y position
          }
        })

        setGraphData({
          nodes: nodesWithPosition,
          links: data.links
        })
      }
    } catch (error) {
      console.error('그래프 데이터 로드 실패:', error)
    } finally {
      setIsLoadingGraph(false)
    }
  }

  // ─────────────────────────────────────────────────────────────
  // API 호출
  // ─────────────────────────────────────────────────────────────

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    // 상태 스냅샷을 만들어 동기적으로 사용 (레이스 컨디션 방지)
    const currentInput = inputMessage
    const currentDocs = [...selectedDocs]

    const formattedContent = currentDocs.length > 0
      ? `${currentDocs.map(d => `@${d}`).join(' ')} ${currentInput}`
      : currentInput

    const userMessage: ChatMessage = {
      role: 'user',
      content: formattedContent,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    const startTime = Date.now()

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: currentDocs.length > 0
            ? `[Selected Documents: ${currentDocs.join(', ')}]\n${currentInput}`
            : currentInput,
          session_id: sessionId,
          llm_model: 'gpt-4o-mini',
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
      setSelectedDocs([]) // 전송 후 선택된 문서 초기화
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

      // 입력된 텍스트에서 @멘션 부분을 제거하고 나머지만 유지
      const newValue = before + (afterAt.startsWith(' ') ? afterAt.substring(1) : afterAt)
      setInputMessage(newValue)

      // 선택된 문서 목록에 추가 (중복 방지)
      if (!selectedDocs.includes(name)) {
        setSelectedDocs(prev => [...prev, name])
      }

      setShowSuggestions(false)
      setMentionTriggerPos(null)
    }
  }

  const removeSelectedDoc = (docId: string) => {
    setSelectedDocs(prev => prev.filter(id => id !== docId))
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
      setEditedContent(content) // 편집 내용도 함께 설정
      setIsEditing(false) // 문서 선택 시 편집 모드 해제
    } else {
      // 내용이 제공되지 않으면 API에서 가져오기
      try {
        const response = await fetch(`${API_URL}/rag/document/${docId}/content`)
        const data = await response.json()
        console.log('📄 [Document API Response]', data)

        // 원본 마크다운 content를 그대로 사용 (마크다운 렌더링은 JSX에서 처리)
        if (data.content) {
          setDocumentContent(data.content)
          setEditedContent(data.content) // 편집 내용도 함께 설정
          setIsEditing(false) // 문서 선택 시 편집 모드 해제
        } else {
          setDocumentContent('내용을 불러올 수 없습니다.')
          setEditedContent('내용을 불러올 수 없습니다.')
          setIsEditing(false)
        }
      } catch (_error) {
        setDocumentContent('문서 내용을 가져오는 중 오류가 발생했습니다.')
        setEditedContent('문서 내용을 가져오는 중 오류가 발생했습니다.')
        setIsEditing(false)
      }
    }
  }


  // ─────────────────────────────────────────────────────────────
  // 렌더링
  // ─────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* 헤더 */}
      <header className="flex justify-between items-center h-[35px] bg-dark-deeper border-b border-dark-border px-4">
        <div className="flex items-center gap-3">
          <button
            className={`border-none py-1 px-2 text-[14px] rounded cursor-pointer flex items-center justify-center transition-all duration-200 ${isLeftVisible ? 'bg-transparent text-txt-secondary hover:bg-dark-hover hover:text-accent' : 'bg-accent/10 text-accent'}`}
            onClick={() => setIsLeftVisible(!isLeftVisible)}
            title={isLeftVisible ? "사이드바 접기" : "사이드바 펴기"}
          >
            {isLeftVisible ? '◀' : '▶'}
          </button>
          <span className="text-[13px] text-txt-primary">Orchestrator Agent</span>
          {selectedDocument && (
            <div className="flex gap-2 ml-4">
              {!isEditing ? (
                <button
                  className="bg-dark-hover border border-dark-border text-accent py-1 px-3 text-[11px] rounded cursor-pointer transition-all duration-200 hover:bg-dark-border hover:border-txt-secondary"
                  onClick={() => setIsEditing(true)}
                  title="문서 수정"
                >
                  수정
                </button>
              ) : (
                <>
                  <button
                    className="bg-dark-hover border border-dark-border text-[#f48fb1] py-1 px-3 text-[11px] rounded cursor-pointer transition-all duration-200 hover:bg-dark-border hover:border-txt-secondary"
                    onClick={() => {
                      setIsEditing(false)
                      setEditedContent(documentContent || '')
                    }}
                  >
                    취소
                  </button>
                  <button
                    className="bg-accent-blue text-white border-accent-blue py-1 px-3 text-[11px] rounded cursor-pointer transition-all duration-200 hover:bg-[#0062a3]"
                    onClick={handleSaveDocument}
                  >
                    저장
                  </button>
                </>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-[12px] ${isConnected ? 'text-accent' : 'text-[#f48771]'}`}>
            {isConnected ? '[OK]' : '[ERROR]'} {agentStatus}
          </span>
          <button
            className={`border-none py-1 px-2 text-[14px] rounded cursor-pointer flex items-center justify-center transition-all duration-200 ${isRightVisible ? 'bg-transparent text-txt-secondary hover:bg-dark-hover hover:text-accent' : 'bg-accent/10 text-accent'}`}
            onClick={() => setIsRightVisible(!isRightVisible)}
            title={isRightVisible ? "채팅 패널 접기" : "채팅 패널 펴기"}
          >
            {isRightVisible ? '▶' : '◀'}
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* 왼쪽: 사이드바 아이콘 */}
        <Sidebar activePanel={activePanel} onPanelChange={(panel) => {
          setActivePanel(panel);
          if (panel) setIsLeftVisible(true);
        }} />

        {/* 문서 관리 패널 (visualization 모드엔 표시 안 함) */}
        <div className={`flex-shrink-0 bg-dark-deeper border-r border-dark-border flex flex-col overflow-hidden transition-[width,opacity,border-color] duration-300 ease-in-out ${!isLeftVisible || !activePanel || activePanel === 'visualization' ? 'w-0 opacity-0 border-r-transparent pointer-events-none' : 'w-80'}`}>
          {activePanel === 'documents' && (
            <DocumentManagementPanel onDocumentSelect={handleDocumentSelect} />
          )}
        </div>

        {/* 가운데: 문서 뷰어 또는 그래프 시각화 */}
        <main
          className={`flex-1 bg-dark-bg overflow-y-auto flex flex-col transition-all duration-300 relative ${isDraggingOver ? 'outline outline-2 outline-accent-blue outline-offset-[-2px]' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
            if (!isDraggingOver) setIsDraggingOver(true);
          }}
          onDragLeave={() => setIsDraggingOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDraggingOver(false);
            const docId = e.dataTransfer.getData('text/plain');
            if (docId) handleDocumentSelect(docId);
          }}
        >
          {isDraggingOver && (
            <div className="absolute inset-0 bg-accent-blue/10 flex items-center justify-center z-50 pointer-events-none">
              <div className="flex flex-col items-center gap-3 text-txt-primary">
                <span className="text-[48px]">📄</span>
                <span className="text-[16px]">여기에 드롭하여 문서 열기</span>
              </div>
            </div>
          )}
          {activePanel === 'visualization' ? (
            // 전체 문서 그래프 시각화
            <div className="flex flex-col h-full overflow-hidden">
              <div className="flex justify-between items-center px-6 py-4 bg-dark-deeper border-b border-dark-border">
                <div className="flex flex-col gap-2">
                  <h2 className="text-[16px] font-medium m-0 text-txt-primary">전체 문서 관계 그래프</h2>
                  <div className="flex gap-4 text-[12px]">
                    <span className="flex items-center gap-1.5 text-txt-secondary">
                      <span className="w-3 h-3 rounded-full border border-[#333] bg-[#A8E6CF] inline-block"></span>
                      SOP (표준운영절차서)
                    </span>
                    <span className="flex items-center gap-1.5 text-txt-secondary">
                      <span className="w-3 h-3 rounded-full border border-[#333] bg-[#FFD3A5] inline-block"></span>
                      WI (작업지침서)
                    </span>
                    <span className="flex items-center gap-1.5 text-txt-secondary">
                      <span className="w-3 h-3 rounded-full border border-[#333] bg-[#FFB3BA] inline-block"></span>
                      FRM (양식)
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  {graphData && (
                    <span className="text-[12px] text-txt-secondary">
                      문서: {graphData.nodes.length}개 | 연결: {graphData.links.length}개
                    </span>
                  )}
                  <button
                    className="py-1.5 px-3 bg-dark-light text-txt-primary border border-dark-border rounded text-[12px] cursor-pointer transition-all duration-150 hover:bg-dark-hover hover:border-accent"
                    onClick={() => fgRef.current?.zoomToFit(400, 80)}
                    title="중앙으로"
                  >
                    중앙으로
                  </button>
                </div>
              </div>
              <div className="graph-container flex-1 relative overflow-hidden" ref={graphContainerRef}>
                {isLoadingGraph ? (
                  <div className="flex items-center justify-center h-full text-txt-secondary text-[14px]">데이터를 불러오는 중...</div>
                ) : graphData && graphData.nodes.length > 0 ? (
                  <ForceGraph2D
                    ref={fgRef}
                    graphData={graphData}
                    nodeLabel={(node: any) => `${node.id}\n${node.title || ''}`}
                    nodeRelSize={25}
                    onNodeClick={(node: any) => {
                      // 문서 관리 패널로 전환
                      setActivePanel('documents')
                      // 해당 문서 내용 표시
                      fetchDocumentContent(node.id)
                    }}
                    nodeCanvasObject={(node: any, ctx, globalScale) => {
                      const label = node.id
                      const fontSize = 11 / globalScale
                      ctx.font = `${fontSize}px Sans-Serif`

                      // 노드 색상 (파스텔 톤)
                      let color = '#A5D8FF'  // 기타 (파스텔 블루)
                      if (node.doc_type === 'SOP') color = '#A8E6CF'  // 파스텔 그린
                      else if (node.doc_type === 'WI') color = '#FFD3A5'  // 파스텔 오렌지
                      else if (node.doc_type === 'FRM') color = '#FFB3BA'  // 파스텔 핑크

                      // 원형 노드
                      const radius = 25 / globalScale
                      ctx.beginPath()
                      ctx.arc(node.x!, node.y!, radius, 0, 2 * Math.PI)
                      ctx.fillStyle = color
                      ctx.fill()
                      ctx.strokeStyle = '#555'
                      ctx.lineWidth = 2 / globalScale
                      ctx.stroke()

                      // 레이블 (원 밖 아래에 하얀색으로)
                      ctx.fillStyle = '#cccccc'
                      ctx.textAlign = 'center'
                      ctx.textBaseline = 'top'
                      ctx.fillText(label, node.x!, node.y! + radius + 5)
                    }}
                    linkColor={() => '#3e3e42'}
                    linkWidth={1}
                    backgroundColor="#1F1F1F"
                    width={graphSize.width || 600}
                    height={graphSize.height || 500}
                    enableNodeDrag={false}
                    enableZoomInteraction={true}
                    enablePanInteraction={false}
                    cooldownTicks={0}
                    minZoom={0.3}
                    maxZoom={5}
                    onEngineStop={() => fgRef.current?.zoomToFit(400, 80)}
                  />
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-txt-secondary">
                    <p>그래프 데이터가 없습니다.</p>
                  </div>
                )}
              </div>
            </div>
          ) : selectedDocument && documentContent ? (
            // 문서 내용 표시
            <div className="flex-1 overflow-y-auto">
              <div className="px-6 py-4 border-b border-dark-border bg-dark-deeper flex justify-between items-center">
                <h2 className="text-[16px] font-medium text-txt-primary">{selectedDocument}</h2>
                <div className="relative">
                  <button
                    className="bg-accent text-black border-none py-1.5 px-4 rounded text-[12px] font-bold cursor-pointer hover:bg-accent-hover transition-all duration-200 flex items-center gap-2 shadow-lg"
                    onClick={() => setIsDownloadOpen(!isDownloadOpen)}
                  >
                    📥 Download <span className="opacity-50">▼</span>
                  </button>

                  {isDownloadOpen && (
                    <div className="absolute right-0 mt-2 w-40 bg-dark-light border border-dark-border rounded shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
                      <button
                        className="w-full text-left px-4 py-2.5 text-[12px] text-txt-primary hover:bg-dark-hover transition-colors flex items-center gap-2"
                        onClick={() => handleDownload('pdf')}
                      >
                        <span className="text-red-400">📄</span> PDF Document
                      </button>
                      <button
                        className="w-full text-left px-4 py-2.5 text-[12px] text-txt-primary hover:bg-dark-hover border-t border-dark-border transition-colors flex items-center gap-2"
                        onClick={() => handleDownload('docx')}
                      >
                        <span className="text-blue-400">📝</span> Word (.docx)
                      </button>
                      <button
                        className="w-full text-left px-4 py-2.5 text-[12px] text-txt-primary hover:bg-dark-hover border-t border-dark-border transition-colors flex items-center gap-2"
                        onClick={() => handleDownload('md')}
                      >
                        <span className="text-green-400"> markdown </span> Markdown (.md)
                      </button>
                    </div>
                  )}
                </div>
              </div>
              <div className="py-10 px-5 bg-[#e0e0e0] flex flex-col items-center gap-[30px]">
                {isEditing ? (
                  <div className="w-full max-w-[1000px] h-[calc(100vh-120px)] bg-dark-deeper border border-dark-border rounded overflow-hidden shadow-[0_10px_30px_rgba(0,0,0,0.3)]">
                    <textarea
                      className="document-editor w-full h-full bg-transparent text-[#d4d4d4] border-none p-[30px] font-mono text-[14px] leading-[1.6] resize-none outline-none"
                      value={editedContent}
                      onChange={(e) => setEditedContent(e.target.value)}
                      placeholder="문서 내용을 수정하세요..."
                    />
                  </div>
                ) : (
                  (() => {
                    if (!documentContent) return (
                      <div className="flex-1 flex flex-col items-center justify-center text-txt-secondary">
                        <div className="text-[64px] mb-4 opacity-50">[FILE]</div>
                        <h2 className="text-[18px] font-medium mb-2 text-txt-primary">Select a document</h2>
                      </div>
                    );

                    // PAGE 마커를 기준으로 분할
                    const pages = documentContent.split(/<!-- PAGE:\d+ -->/);
                    const filteredPages = pages.filter((page, index) => index > 0 || page.trim() !== '');

                    return (
                      <div className="w-full max-w-[900px] flex flex-col gap-[40px]">
                        {filteredPages.map((page, index) => (
                          <div key={index} className="bg-white text-[#333] py-[80px] px-[70px] shadow-[0_10px_30px_rgba(0,0,0,0.15)] min-h-[1100px] flex flex-col relative rounded">
                            <div className="flex-1 whitespace-pre-wrap break-words text-[#2c3e50]">
                              {(() => {
                                let currentDepth = 0;
                                return page.split('\n').map((line, lineIdx) => {
                                  const trimmedLine = line.trim();
                                  if (trimmedLine === '') {
                                    return <div key={lineIdx} className="h-3" />;
                                  }

                                  const sectionMatch = trimmedLine.match(/^(\d+(?:\.\d+)*)\.?\s+/);
                                  if (sectionMatch) {
                                    const parts = sectionMatch[1].split('.');
                                    currentDepth = parts.length - 1;
                                  }

                                  const depthStyle = { paddingLeft: `${currentDepth * 32}px` };

                                  if (currentDepth === 0 && sectionMatch) {
                                    return <div key={lineIdx} className="text-[19px] font-bold mt-[40px] mb-[20px] text-[#1a1a1a] border-b-2 border-[#e9ecef] pb-[10px]" style={depthStyle}>{trimmedLine}</div>;
                                  }

                                  if (sectionMatch) {
                                    return <div key={lineIdx} className="text-[15px] font-normal mt-[18px] mb-[6px] text-[#2c3e50]" style={depthStyle}>{trimmedLine}</div>;
                                  }

                                  if (/^={10,}/.test(trimmedLine)) {
                                    return <div key={lineIdx} className="text-[#bdc3c7] tracking-[2px] my-4 font-mono">{trimmedLine}</div>;
                                  }

                                  return <div key={lineIdx} className="text-[15px] leading-[1.8] mb-[6px]" style={depthStyle}>{line}</div>;
                                });
                              })()}
                            </div>
                            <div className="mt-[60px] pt-5 border-t border-[#f8f9fa] flex justify-end">
                              <span className="text-[13px] text-[#95a5a6] font-medium">{index + 1} / {filteredPages.length}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    );
                  })()
                )}
              </div>
            </div>
          ) : (
            // 빈 상태
            <div className="flex-1 flex flex-col items-center justify-center text-txt-secondary">
              <div className="text-[64px] mb-4 opacity-50">[FILE]</div>
              <h2 className="text-[18px] font-medium mb-2 text-txt-primary">Select a document</h2>
            </div>
          )}
        </main>

        {/* 오른쪽: Agent 패널 */}
        <aside className={`flex-shrink-0 bg-dark-deeper border-l border-dark-border flex flex-col overflow-hidden transition-[width,opacity,border-color] duration-300 ease-in-out ${!isRightVisible ? 'w-0 opacity-0 border-l-transparent pointer-events-none' : 'w-[420px]'}`}>
          <div className="flex justify-between items-center px-4 py-2 h-[35px] border-b border-dark-border">
            <span className="text-[13px] font-medium text-txt-primary">Agent Chat</span>
          </div>

          <div className="flex-1 flex flex-col overflow-hidden">
            {/* 채팅 영역 */}
            <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
              {messages.map((msg, index) => (
                <div key={index} className="flex flex-col gap-2">
                  {msg.role === 'user' ? (
                    <div className="bg-dark-light rounded-lg p-3 border border-dark-border">
                      <div className="flex-1 text-[13px] text-txt-primary">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p({ children }) {
                              const docPattern = /(EQ-(?:SOP|WI)-\d{5}(?:\([\d.,\s]+\))?)/g;
                              const processText = (text: string) => {
                                const parts = text.split(docPattern);
                                return parts.map((part, i) => {
                                  if (docPattern.test(part)) {
                                    const docId = part.split('(')[0].replace(/^@/, '');
                                    return (
                                      <span
                                        key={i}
                                        className="text-accent underline cursor-pointer font-medium px-1 py-[1px] rounded transition-all duration-200 hover:bg-white/10 hover:text-accent-hover"
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
                            }
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-2">
                      {/* Thought Process */}
                      {msg.thoughtProcess && (
                        <div className="bg-dark-light rounded overflow-hidden border border-dark-border">
                          <div
                            className="flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors duration-200 select-none hover:bg-dark-hover"
                            onClick={() => toggleSection(`thought-${index}`)}
                          >
                            <span className="text-[10px] text-txt-muted w-3">
                              {expandedSections.has(`thought-${index}`) ? '▼' : '▶'}
                            </span>
                            <span className="text-[13px] text-txt-secondary font-medium">Show Reasoning</span>
                          </div>
                          {expandedSections.has(`thought-${index}`) && (
                            <pre className="p-3 border-t border-dark-border text-[13px] text-txt-primary leading-[1.6] bg-dark-deeper">
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
                                        className="text-accent underline cursor-pointer font-medium px-1 py-[1px] rounded transition-all duration-200 hover:bg-white/10 hover:text-accent-hover"
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
                        <div className="text-[11px] text-txt-muted mt-2">Time: {msg.thinkingTime}s</div>
                      )}

                      {/* 평가 점수 */}
                      {msg.evaluation_scores && (
                        <div className="mt-3 border-t border-dark-border pt-2">
                          <div
                            className="flex items-center gap-2 p-2 cursor-pointer rounded transition-colors duration-200 hover:bg-dark-hover"
                            onClick={() => toggleSection(`eval-${index}`)}
                          >
                            <span className="text-[10px] text-txt-muted w-3">
                              {expandedSections.has(`eval-${index}`) ? '▼' : '▶'}
                            </span>
                            <span className="text-[13px] font-semibold text-txt-secondary">
                              🔍 평가 점수
                              {msg.evaluation_scores.average_score && (
                                <span className="text-[12px] text-accent font-bold ml-2"> ({msg.evaluation_scores.average_score.toFixed(1)}/5.0)</span>
                              )}
                            </span>
                          </div>
                          {expandedSections.has(`eval-${index}`) && (
                            <div className="p-3 bg-dark-deeper rounded mt-2">
                              {msg.evaluation_scores.faithfulness && (
                                <div className="mb-4 pb-3 border-b border-dark-border last:mb-0 last:pb-0 last:border-b-0">
                                  <span className="text-[12px] font-semibold text-txt-primary mr-2">충실성 (Faithfulness):</span>
                                  <span className={`text-sm font-bold py-0.5 px-2 rounded ml-2 ${SCORE_COLORS[msg.evaluation_scores.faithfulness.score] ?? ''}`}>
                                    {msg.evaluation_scores.faithfulness.score}/5
                                  </span>
                                  <div className="text-[11px] text-txt-secondary mt-1.5 leading-[1.4] pl-1 border-l-2 border-dark-border">{msg.evaluation_scores.faithfulness.reasoning}</div>
                                  {msg.evaluation_scores.faithfulness.rdb_verification && (
                                    <div className="mt-2.5 p-2.5 bg-dark-deeper rounded border border-dark-border">
                                      <div className="text-[11px] font-bold text-accent mb-2">📊 RDB 검증 결과</div>
                                      <div className="flex gap-4 mb-2">
                                        <span className="text-[11px] text-txt-secondary">
                                          정확도: <strong className="text-accent text-[13px]">{msg.evaluation_scores.faithfulness.rdb_verification.accuracy_rate}%</strong>
                                        </span>
                                        <span className="text-[11px] text-txt-secondary">
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
                                <div className="mb-4 pb-3 border-b border-dark-border last:mb-0 last:pb-0 last:border-b-0">
                                  <span className="text-[12px] font-semibold text-txt-primary mr-2">근거성 (Groundness):</span>
                                  <span className={`text-sm font-bold py-0.5 px-2 rounded ml-2 ${SCORE_COLORS[msg.evaluation_scores.groundness.score] ?? ''}`}>
                                    {msg.evaluation_scores.groundness.score}/5
                                  </span>
                                  <div className="text-[11px] text-txt-secondary mt-1.5 leading-[1.4] pl-1 border-l-2 border-dark-border">{msg.evaluation_scores.groundness.reasoning}</div>
                                  {msg.evaluation_scores.groundness.rdb_verification && (
                                    <div className="mt-2.5 p-2.5 bg-dark-deeper rounded border border-dark-border">
                                      <div className="flex gap-4 mb-2">
                                        <span className="text-[11px] text-txt-secondary">
                                          정확도: <strong className="text-accent text-[13px]">{msg.evaluation_scores.groundness.rdb_verification.accuracy_rate}%</strong>
                                        </span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                              {msg.evaluation_scores.relevancy && (
                                <div className="mb-4 pb-3 border-b border-dark-border last:mb-0 last:pb-0 last:border-b-0">
                                  <span className="text-[12px] font-semibold text-txt-primary mr-2">관련성 (Relevancy):</span>
                                  <span className={`text-sm font-bold py-0.5 px-2 rounded ml-2 ${SCORE_COLORS[msg.evaluation_scores.relevancy.score] ?? ''}`}>
                                    {msg.evaluation_scores.relevancy.score}/5
                                  </span>
                                  <div className="text-[11px] text-txt-secondary mt-1.5 leading-[1.4] pl-1 border-l-2 border-dark-border">{msg.evaluation_scores.relevancy.reasoning}</div>
                                </div>
                              )}
                              {msg.evaluation_scores.correctness && (
                                <div className="mb-4 pb-3 border-b border-dark-border last:mb-0 last:pb-0 last:border-b-0">
                                  <span className="text-[12px] font-semibold text-txt-primary mr-2">정확성 (Correctness):</span>
                                  <span className={`text-sm font-bold py-0.5 px-2 rounded ml-2 ${SCORE_COLORS[msg.evaluation_scores.correctness.score] ?? ''}`}>
                                    {msg.evaluation_scores.correctness.score}/5
                                  </span>
                                  <div className="text-[11px] text-txt-secondary mt-1.5 leading-[1.4] pl-1 border-l-2 border-dark-border">{msg.evaluation_scores.correctness.reasoning}</div>
                                  {msg.evaluation_scores.correctness.rdb_verification && (
                                    <div className="mt-2.5 p-2.5 bg-dark-deeper rounded border border-dark-border">
                                      <div className="flex gap-4 mb-2">
                                        <span className="text-[11px] text-txt-secondary">
                                          정확도: <strong className="text-accent text-[13px]">{msg.evaluation_scores.correctness.rdb_verification.accuracy_rate}%</strong>
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
                <div className="flex flex-col gap-2">
                  <div className="flex flex-col gap-2">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                      Processing request...
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* 하단 입력 영역 */}
            <div className="border-t border-dark-border p-3 bg-dark-deeper">
              <div className="relative bg-dark-light border border-dark-border rounded-md px-2.5 py-1.5 transition-all duration-200 flex flex-row flex-wrap items-center gap-1.5 min-h-[40px] focus-within:border-accent focus-within:shadow-[0_0_0_1px_rgba(34,209,66,0.2)]">
                {selectedDocs.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {selectedDocs.map(docId => (
                      <div key={docId} className="flex items-center gap-1 bg-accent/10 border border-accent px-1.5 py-[1px] rounded">
                        <span className="text-[11px] text-accent font-medium">{docId}</span>
                        <button
                          className="bg-transparent border-none text-accent cursor-pointer text-[12px] p-0 flex items-center justify-center leading-none"
                          onClick={() => removeSelectedDoc(docId)}
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <textarea
                  value={inputMessage}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyPress}
                  placeholder={selectedDocs.length > 0 ? "" : "Ask the Agent...And Tag with @"}
                  className="agent-input flex-1 min-w-[120px] bg-transparent border-none py-1.5 text-txt-primary text-[13px] resize-none min-h-[24px] max-h-[120px] font-[inherit] focus:outline-none placeholder:text-[#6a6a6a]"
                  rows={1}
                />
                {showSuggestions && (
                  <div className="absolute bottom-full left-0 w-full max-h-[200px] overflow-y-auto bg-dark-light border border-dark-border rounded shadow-[0_-4px_12px_rgba(0,0,0,0.5)] z-[1000] mb-1">
                    {suggestions.map((doc, idx) => (
                      <div
                        key={doc.id}
                        className={`px-3 py-2 cursor-pointer text-[13px] transition-colors duration-200 ${idx === suggestionIndex ? 'bg-dark-hover text-accent' : 'text-txt-primary hover:bg-dark-hover hover:text-accent'}`}
                        onClick={() => selectSuggestion(doc.name)}
                      >
                        {doc.name}
                      </div>
                    ))}
                  </div>
                )}
                <button
                  className="bg-accent text-black border-none py-1.5 px-3 rounded font-semibold cursor-pointer transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:enabled:bg-accent-hover"
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

      {/* 저장 중 로딩 오버레이 */}
      {isSaving && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-[2000]">
          <div className="bg-[#2d2d2d] border border-dark-border rounded-lg p-8 flex flex-col items-center gap-4 text-center">
            <div className="w-10 h-10 border-4 border-dark-border border-t-accent-blue rounded-full animate-spin"></div>
            <p className="text-txt-primary text-[14px] m-0">문서를 분석하고 저장하는 중입니다...</p>
            <span className="text-txt-secondary text-[12px]">이 작업은 최대 1분 정도 소요될 수 있습니다.</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
