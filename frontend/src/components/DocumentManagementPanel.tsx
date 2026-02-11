import { useState, useEffect } from 'react';
import './DocumentManagementPanel.css';

const API_URL = 'http://localhost:8000';

interface Document {
  doc_id: string;
  doc_name?: string;
  doc_title?: string;
  doc_category?: string;
  chunk_count?: number;
  model?: string;
  collection?: string;
}

interface DocumentGroup {
  category: string;
  documents: Document[];
  expanded: boolean;
}

interface Version {
  version: string;
  created_at: string;
}

interface DocumentContent {
  doc_name: string;
  version: string;
  content: string;
  chunk_count: number;
}

interface DocumentManagementPanelProps {
  onDocumentSelect?: (docId: string, content?: string) => void;
}

export default function DocumentManagementPanel({ onDocumentSelect }: DocumentManagementPanelProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [groupedDocuments, setGroupedDocuments] = useState<Map<string, DocumentGroup>>(new Map());
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [documentContent, setDocumentContent] = useState<DocumentContent | null>(null);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string>('');

  // 문서 목록 로드
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/rag/documents`);
      const data = await response.json();
      console.log('🔍 [Documents API Response]', data);
      const docs = data.documents || [];
      setDocuments(docs);

      // 문서를 카테고리별로 그룹화
      const groups = new Map<string, DocumentGroup>();
      docs.forEach((doc: Document) => {
        const category = doc.doc_category || '기타';
        if (!groups.has(category)) {
          groups.set(category, {
            category,
            documents: [],
            expanded: true, // 기본적으로 펼쳐진 상태
          });
        }
        groups.get(category)!.documents.push(doc);
      });

      // 카테고리 순서: SOP > WI > FRM > 기타
      const sortedGroups = new Map(
        Array.from(groups.entries()).sort((a, b) => {
          const order = ['SOP', 'WI', 'FRM', '기타'];
          return order.indexOf(a[0]) - order.indexOf(b[0]);
        })
      );

      setGroupedDocuments(sortedGroups);
    } catch (error) {
      console.error('문서 목록 조회 실패:', error);
    }
  };

  const toggleGroup = (category: string) => {
    setGroupedDocuments((prev) => {
      const newGroups = new Map(prev);
      const group = newGroups.get(category);
      if (group) {
        group.expanded = !group.expanded;
      }
      return newGroups;
    });
  };

  // 문서 선택 시 버전 목록 로드
  const handleDocumentSelect = async (docName: string) => {
    setSelectedDoc(docName);
    setDocumentContent(null);

    try {
      const response = await fetch(`${API_URL}/rag/document/${docName}/versions`);
      const data = await response.json();
      setVersions(data.versions || []);
    } catch (error) {
      console.error('버전 목록 조회 실패:', error);
      setVersions([]);
    }
  };

  // 문서 내용 보기
  const handleViewDocument = async (docName: string, version?: string) => {
    try {
      const url = version
        ? `${API_URL}/rag/document/${docName}/content?version=${version}`
        : `${API_URL}/rag/document/${docName}/content`;

      const response = await fetch(url);
      const data = await response.json();
      setDocumentContent(data);

      // App.tsx의 뷰어에도 표시 (content 전달하지 않아서 chunks 구조화 로직 실행)
      if (onDocumentSelect) {
        onDocumentSelect(docName);
      }
    } catch (error) {
      console.error('문서 내용 조회 실패:', error);
    }
  };

  // 문서 삭제
  const handleDeleteDocument = async (docName: string) => {
    if (!confirm(`"${docName}" 문서를 삭제하시겠습니까?`)) {
      return;
    }

    try {
      const response = await fetch(`${API_URL}/rag/document`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_name: docName, collection: 'documents' }),
      });

      if (response.ok) {
        alert('문서가 삭제되었습니다.');
        fetchDocuments();
        if (selectedDoc === docName) {
          setSelectedDoc(null);
          setVersions([]);
          setDocumentContent(null);
        }
      } else {
        alert('문서 삭제 실패');
      }
    } catch (error) {
      console.error('문서 삭제 실패:', error);
      alert('문서 삭제 중 오류 발생');
    }
  };

  // 문서 업로드
  const handleUpload = async () => {
    if (!uploadFile) {
      alert('파일을 선택해주세요.');
      return;
    }

    setIsUploading(true);
    setUploadProgress('업로드 중...');

    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('collection', 'documents');
    formData.append('use_langgraph', 'true');

    try {
      const response = await fetch(`${API_URL}/rag/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setUploadProgress('🟢 업로드 완료!');
        setTimeout(() => {
          setIsUploadModalOpen(false);
          setUploadFile(null);
          setUploadProgress('');
          fetchDocuments();
        }, 1500);
      } else {
        setUploadProgress('🔴 업로드 실패');
      }
    } catch (error) {
      console.error('업로드 실패:', error);
      setUploadProgress('🔴 업로드 중 오류 발생');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="document-management-panel">
      <div className="panel-header">
        <h2>문서 관리</h2>
        <button className="btn-upload" onClick={() => setIsUploadModalOpen(true)}>
          + 업로드
        </button>
      </div>

      <div className="panel-content">
        {/* 문서 목록 (폴더 구조) */}
        <div className="document-list">
          <h3>문서 목록</h3>
          {groupedDocuments.size === 0 ? (
            <p className="empty-message">문서가 없습니다.</p>
          ) : (
            Array.from(groupedDocuments.values()).map((group) => (
              <div key={group.category} className="document-group">
                {/* 폴더 헤더 */}
                <div className="folder-header" onClick={() => toggleGroup(group.category)}>
                  <span className="folder-icon">{group.expanded ? '📂' : '📁'}</span>
                  <span className="folder-name">{group.category}</span>
                  <span className="folder-count">({group.documents.length})</span>
                </div>

                {/* 폴더 내 문서들 */}
                {group.expanded && (
                  <div className="folder-content">
                    {group.documents.map((doc, idx) => (
                      <div
                        key={idx}
                        className={`document-item ${selectedDoc === doc.doc_id ? 'active' : ''}`}
                      >
                        <div className="document-info" onClick={() => handleDocumentSelect(doc.doc_id)}>
                          <span className="doc-icon">📄</span>
                          <span>{doc.doc_id}</span>
                          {doc.chunk_count && (
                            <span className="doc-chunk-count">({doc.chunk_count}개)</span>
                          )}
                        </div>
                        <div className="document-actions">
                          <button
                            className="btn-icon"
                            onClick={() => handleViewDocument(doc.doc_id)}
                            title="내용 보기"
                          >
                            보기
                          </button>
                          <button
                            className="btn-icon btn-delete"
                            onClick={() => handleDeleteDocument(doc.doc_id)}
                            title="삭제"
                          >
                            삭제
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* 버전 목록 */}
        {selectedDoc && versions.length > 0 && (
          <div className="version-list">
            <h3>버전 이력</h3>
            {versions.map((ver) => (
              <div key={ver.version} className="version-item">
                <div className="version-info">
                  <span>v{ver.version}</span>
                  <span className="version-date">{new Date(ver.created_at).toLocaleDateString()}</span>
                </div>
                <button
                  className="btn-icon"
                  onClick={() => handleViewDocument(selectedDoc, ver.version)}
                  title="이 버전 보기"
                >
                  보기
                </button>
              </div>
            ))}
          </div>
        )}

        {/* 문서 내용 */}
        {documentContent && (
          <div className="document-content">
            <h3>
              {documentContent.doc_name} (v{documentContent.version})
            </h3>
            <div className="content-stats">
              <span>청크: {documentContent.chunk_count}개</span>
            </div>
            <div className="content-text">
              <pre>{documentContent.content.substring(0, 2000)}...</pre>
            </div>
          </div>
        )}
      </div>

      {/* 업로드 모달 */}
      {isUploadModalOpen && (
        <div className="modal-overlay" onClick={() => setIsUploadModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>문서 업로드</h3>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
              disabled={isUploading}
            />
            {uploadProgress && <p className="upload-progress">{uploadProgress}</p>}
            <div className="modal-actions">
              <button onClick={handleUpload} disabled={isUploading || !uploadFile}>
                업로드
              </button>
              <button onClick={() => setIsUploadModalOpen(false)} disabled={isUploading}>
                취소
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
