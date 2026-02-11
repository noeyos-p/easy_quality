import { useState, useEffect } from 'react';
import MermaidRenderer from './MermaidRenderer';
import './DocumentVisualizationPanel.css';

const API_URL = 'http://localhost:8000';

interface Document {
  doc_id: string;
  title?: string;
}

interface ImpactSection {
  section_id: string;
  section_title: string;
  context: string;
}

interface Impact {
  doc_id: string;
  sections: ImpactSection[];
}

interface ImpactAnalysis {
  doc_id: string;
  impacts: Impact[];
  count: number;
  total_sections: number;
  message?: string;
}

export default function DocumentVisualizationPanel() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string>('');
  const [mermaidCode, setMermaidCode] = useState<string>('');
  const [impactAnalysis, setImpactAnalysis] = useState<ImpactAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // 문서 목록 로드
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/graph/documents`);
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('문서 목록 조회 실패:', error);
    }
  };

  // 문서 선택 시 시각화 및 영향 분석 로드
  const handleDocumentSelect = async (docId: string) => {
    if (!docId) {
      setMermaidCode('');
      setImpactAnalysis(null);
      return;
    }

    setSelectedDoc(docId);
    setIsLoading(true);

    try {
      // 시각화 데이터 로드 (Mermaid)
      const vizResponse = await fetch(`${API_URL}/graph/visualization/${docId}?format=mermaid`);
      const vizData = await vizResponse.json();
      setMermaidCode(vizData.code || '');

      // 영향 분석 로드
      const impactResponse = await fetch(`${API_URL}/graph/impact/${docId}`);
      const impactData = await impactResponse.json();
      setImpactAnalysis(impactData);
    } catch (error) {
      console.error('데이터 로드 실패:', error);
      setMermaidCode('');
      setImpactAnalysis(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="document-visualization-panel">
      <div className="panel-header">
        <h2>문서 시각화</h2>
      </div>

      <div className="panel-content">
        {/* 문서 선택 */}
        <div className="document-selector">
          <label>
            문서 선택
          </label>
          <select value={selectedDoc} onChange={(e) => handleDocumentSelect(e.target.value)}>
            <option value="">문서를 선택하세요</option>
            {documents.map((doc) => (
              <option key={doc.doc_id} value={doc.doc_id}>
                {doc.doc_id} {doc.title ? `- ${doc.title}` : ''}
              </option>
            ))}
          </select>
        </div>

        {isLoading && (
          <div className="loading-message">
            <p>데이터를 불러오는 중...</p>
          </div>
        )}

        {!isLoading && selectedDoc && (
          <>
            {/* 관계 그래프 */}
            {mermaidCode && (
              <div className="visualization-section">
                <h3>문서 관계 그래프</h3>
                <div className="mermaid-container">
                  <MermaidRenderer chart={mermaidCode} />
                </div>
              </div>
            )}

            {/* 영향 분석 */}
            {impactAnalysis && (
              <div className="visualization-section">
                <h3>
                  영향 분석
                </h3>

                {impactAnalysis.count === 0 ? (
                  <p className="info-message">{impactAnalysis.message}</p>
                ) : (
                  <>
                    <div className="impact-summary">
                      <span>영향받는 문서: {impactAnalysis.count}개</span>
                      <span>관련 조항: {impactAnalysis.total_sections}개</span>
                    </div>

                    <div className="impact-list">
                      {impactAnalysis.impacts.map((impact) => (
                        <div key={impact.doc_id} className="impact-item">
                          <div className="impact-doc-name">{impact.doc_id}</div>
                          <div className="impact-sections">
                            {impact.sections.map((section, idx) => (
                              <div key={idx} className="impact-section">
                                <span className="section-id">{section.section_id}</span>
                                <span className="section-title">{section.section_title}</span>
                                {section.context && (
                                  <p className="section-context">{section.context.substring(0, 100)}...</p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </>
        )}

        {!isLoading && !selectedDoc && (
          <div className="empty-state">
            <p>문서를 선택하여 관계 그래프와 영향 분석을 확인하세요.</p>
          </div>
        )}
      </div>
    </div>
  );
}
