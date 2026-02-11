import { useState, useRef, useEffect } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { API_URL } from '../../types'

interface GraphVisualizationProps {
  onNodeClick: (docId: string) => void
  onSwitchToDocuments: () => void
}

export default function GraphVisualization({ onNodeClick, onSwitchToDocuments }: GraphVisualizationProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [graphData, setGraphData] = useState<{ nodes: any[], links: any[] } | null>(null)
  const [isLoadingGraph, setIsLoadingGraph] = useState(false)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null)
  const [graphSize, setGraphSize] = useState({ width: 0, height: 0 })
  const graphContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchGraphData()
  }, [])

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
      setTimeout(() => fgRef.current?.zoomToFit(400, 80), 50)
    })
    ro.observe(graphContainerRef.current)
    updateSize()

    return () => ro.disconnect()
  }, [])

  const fetchGraphData = async () => {
    setIsLoadingGraph(true)
    try {
      const response = await fetch(`${API_URL}/graph/visualization/all`)
      const data = await response.json()

      if (data.success) {
        const nodeCount = data.nodes.length
        const radius = Math.min(120, nodeCount * 12)

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const nodesWithPosition = data.nodes.map((node: any, i: number) => {
          const angle = (i / nodeCount) * 2 * Math.PI
          const x = Math.cos(angle) * radius
          const y = Math.sin(angle) * radius - 40

          return {
            id: node.id,
            name: node.id,
            title: node.title,
            version: node.version,
            doc_type: node.doc_type,
            type_name: node.type_name,
            x, y, fx: x, fy: y,
          }
        })

        setGraphData({ nodes: nodesWithPosition, links: data.links })
      }
    } catch (error) {
      console.error('그래프 데이터 로드 실패:', error)
    } finally {
      setIsLoadingGraph(false)
    }
  }

  return (
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
          >
            중앙으로
          </button>
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden" ref={graphContainerRef}>
        {isLoadingGraph ? (
          <div className="flex items-center justify-center h-full text-txt-secondary text-[14px]">
            데이터를 불러오는 중...
          </div>
        ) : graphData && graphData.nodes.length > 0 ? (
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            nodeLabel={(node: any) => `${node.id}\n${node.title || ''}`}
            nodeRelSize={25}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            onNodeClick={(node: any) => {
              onSwitchToDocuments()
              onNodeClick(node.id)
            }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            nodeCanvasObject={(node: any, ctx, globalScale) => {
              const label = node.id
              const fontSize = 11 / globalScale
              ctx.font = `${fontSize}px Sans-Serif`

              let color = '#A5D8FF'
              if (node.doc_type === 'SOP') color = '#A8E6CF'
              else if (node.doc_type === 'WI') color = '#FFD3A5'
              else if (node.doc_type === 'FRM') color = '#FFB3BA'

              const radius = 25 / globalScale
              ctx.beginPath()
              ctx.arc(node.x!, node.y!, radius, 0, 2 * Math.PI)
              ctx.fillStyle = color
              ctx.fill()
              ctx.strokeStyle = '#555'
              ctx.lineWidth = 2 / globalScale
              ctx.stroke()

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
  )
}
