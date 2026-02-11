import './Sidebar.css';

interface SidebarProps {
  activePanel: 'documents' | 'visualization' | null;
  onPanelChange: (panel: 'documents' | 'visualization') => void;
}

export default function Sidebar({ activePanel, onPanelChange }: SidebarProps) {
  return (
    <div className="sidebar">
      <div className="sidebar-icons">
        <button
          className={`sidebar-icon ${activePanel === 'documents' ? 'active' : ''}`}
          onClick={() => onPanelChange('documents')}
          title="문서 관리"
        >
          📄
        </button>

        <button
          className={`sidebar-icon ${activePanel === 'visualization' ? 'active' : ''}`}
          onClick={() => onPanelChange('visualization')}
          title="문서 시각화"
        >
          🔗
        </button>
      </div>
    </div>
  );
}
