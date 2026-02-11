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


interface DocumentManagementPanelProps {
  onDocumentSelect?: (docId: string, content?: string) => void;
}

export default function DocumentManagementPanel({ onDocumentSelect }: DocumentManagementPanelProps) {
  const [groupedDocuments, setGroupedDocuments] = useState<Map<string, DocumentGroup>>(new Map());
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string>('');
  const [isDeleting, setIsDeleting] = useState(false);

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
        newGroups.set(category, { ...group, expanded: !group.expanded });
      }
      return newGroups;
    });
  };

  // 문서 클릭 시 최신 버전 내용 바로 표시
  const handleDocumentSelect = async (docName: string) => {
    setSelectedDoc(docName);

    try {
      const versionResponse = await fetch(`${API_URL}/rag/document/${docName}/versions`);
      const versionData = await versionResponse.json();
      const fetchedVersions: Version[] = versionData.versions || [];
      setVersions(fetchedVersions);

      // 최신 버전(첫 번째) 내용 자동 표시
      const latestVersion = fetchedVersions[0]?.version;
      await handleViewDocument(docName, latestVersion);
    } catch (error) {
      console.error('문서 로드 실패:', error);
      setVersions([]);
      await handleViewDocument(docName);
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

      // App.tsx의 뷰어에 표시
      if (onDocumentSelect) {
        onDocumentSelect(docName);
      }
    } catch (error) {
      console.error('문서 내용 조회 실패:', error);
    }
  };


  // 문서 삭제 (RDB + Weaviate + Neo4j)
  const handleDeleteDocument = async () => {
    if (!selectedDoc) return;
    if (!confirm(`"${selectedDoc}" 문서를 모든 DB에서 삭제하시겠습니까?\n(RDB, VectorDB, GraphDB 전체 삭제)`)) {
      return;
    }

    setIsDeleting(true);
    try {
      const response = await fetch(`${API_URL}/rag/document`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_name: selectedDoc, collection: 'documents', delete_from_neo4j: true }),
      });

      if (response.ok) {
        alert(`"${selectedDoc}" 삭제 완료`);
        setSelectedDoc(null);
        setVersions([]);
        fetchDocuments();
      } else {
        alert('삭제 실패');
      }
    } catch (_error) {
      alert('삭제 중 오류 발생');
    } finally {
      setIsDeleting(false);
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
        await response.json();
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
        <div className="header-actions">
          <button
            className="btn-delete-doc"
            onClick={handleDeleteDocument}
            disabled={!selectedDoc || isDeleting}
            title={selectedDoc ? `"${selectedDoc}" 삭제` : '문서를 선택하세요'}
          >
            {isDeleting ? '삭제 중...' : '- 삭제'}
          </button>
          <button className="btn-upload" onClick={() => setIsUploadModalOpen(true)}>
            + 업로드
          </button>
        </div>
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
