import { useState, useEffect } from 'react';

const API_URL = 'http://localhost:8000';

interface ChangeRecord {
  id: string;
  doc_id: string;
  change_type: string;
  changed_at: string;
  changed_by?: string;
  description?: string;
}

export default function ChangeHistoryPanel() {
  const [changeHistory, setChangeHistory] = useState<ChangeRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchChangeHistory();
  }, []);

  const fetchChangeHistory = async () => {
    setIsLoading(true);
    try {
      // TODO: 실제 API 엔드포인트로 교체 필요
      const response = await fetch(`${API_URL}/rag/changes`);
      if (response.ok) {
        const data = await response.json();
        setChangeHistory(data.changes || []);
      }
    } catch (error) {
      console.error('변경 이력 조회 실패:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full bg-dark-light border-r border-dark-border flex flex-col h-full overflow-hidden">

      {/* panel-header */}
      <div className="px-4 py-3 border-b border-dark-border">
        <h2 className="text-[13px] font-semibold text-txt-primary m-0 uppercase tracking-[0.5px]">변경 이력</h2>
      </div>

      {/* panel-content */}
      <div className="flex-1 overflow-y-auto p-4">

        {isLoading ? (
          <div className="text-center text-txt-secondary py-10 px-5 text-[13px]">
            <p>데이터를 불러오는 중...</p>
          </div>
        ) : changeHistory.length === 0 ? (
          <div className="text-center text-txt-secondary py-10 px-5 text-[13px]">
            <p>변경 이력이 없습니다.</p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {changeHistory.map((record) => (
              <div key={record.id} className="bg-[#1e1e1e] rounded-md p-3 border-l-[3px] border-accent-blue">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[#4ec9b0] text-[13px] font-semibold">{record.doc_id}</span>
                  <span className="text-txt-secondary text-[11px]">
                    {new Date(record.changed_at).toLocaleString('ko-KR')}
                  </span>
                </div>
                <div className="text-txt-primary text-[12px] mb-1">
                  <span className="inline-block bg-[#3c3c3c] py-0.5 px-2 rounded text-[11px] mr-2">
                    {record.change_type}
                  </span>
                  {record.changed_by && (
                    <span className="text-txt-secondary text-[11px]">by {record.changed_by}</span>
                  )}
                </div>
                {record.description && (
                  <p className="text-txt-secondary text-[11px] mt-1.5 mb-0 leading-[1.5]">
                    {record.description}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
