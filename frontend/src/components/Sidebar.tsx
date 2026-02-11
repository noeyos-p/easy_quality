interface SidebarProps {
  activePanel: 'documents' | 'visualization' | null;
  onPanelChange: (panel: 'documents' | 'visualization') => void;
}

export default function Sidebar({ activePanel, onPanelChange }: SidebarProps) {
  return (
    <div className="w-12 bg-[#2d2d2d] border-r border-dark-border flex flex-col items-center py-2">
      <div className="flex flex-col gap-1">
        <button
          className={`sidebar-icon w-12 h-12 flex items-center justify-center bg-transparent border-none text-txt-primary cursor-pointer relative transition-all duration-200 hover:text-txt-white hover:bg-white/10 ${
            activePanel === 'documents'
              ? 'active text-txt-white border-l-2 border-accent-blue'
              : ''
          }`}
          onClick={() => onPanelChange('documents')}
          title="문서 관리"
        >
          📄
        </button>

        <button
          className={`sidebar-icon w-12 h-12 flex items-center justify-center bg-transparent border-none text-txt-primary cursor-pointer relative transition-all duration-200 hover:text-txt-white hover:bg-white/10 ${
            activePanel === 'visualization'
              ? 'active text-txt-white border-l-2 border-accent-blue'
              : ''
          }`}
          onClick={() => onPanelChange('visualization')}
          title="문서 시각화"
        >
          🔗
        </button>
      </div>
    </div>
  );
}
