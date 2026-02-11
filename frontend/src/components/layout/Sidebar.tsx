import documentManageIcon from '../../assets/icons/document-manage.svg';
import graphIcon from '../../assets/icons/graph.svg';
import updateIcon from '../../assets/icons/update.svg';

interface SidebarProps {
  activePanel: 'documents' | 'visualization' | 'history' | null;
  onPanelChange: (panel: 'documents' | 'visualization' | 'history') => void;
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
          <img src={documentManageIcon} alt="문서 관리" className="w-6 h-6" style={{ filter: activePanel === 'documents' ? 'brightness(0) invert(1)' : 'brightness(0) invert(0.6)' }} />
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
          <img src={graphIcon} alt="문서 시각화" className="w-9 h-9 ml-1 mt-1" style={{ filter: activePanel === 'visualization' ? 'brightness(0) invert(1)' : 'brightness(0) invert(0.6)' }} />
        </button>

        <button
          className={`sidebar-icon w-12 h-12 flex items-center justify-center bg-transparent border-none text-txt-primary cursor-pointer relative transition-all duration-200 hover:text-txt-white hover:bg-white/10 ${
            activePanel === 'history'
              ? 'active text-txt-white border-l-2 border-accent-blue'
              : ''
          }`}
          onClick={() => onPanelChange('history')}
          title="변경 이력"
        >
          <img src={updateIcon} alt="변경 이력" className="w-6 h-6 ml-1.5" style={{ filter: activePanel === 'history' ? 'brightness(0) invert(1)' : 'brightness(0) invert(0.6)' }} />
        </button>
      </div>
    </div>
  );
}
