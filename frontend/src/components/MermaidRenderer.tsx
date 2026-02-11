import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

// Mermaid 초기화 설정
mermaid.initialize({
    startOnLoad: true,
    theme: 'dark',
    securityLevel: 'loose',
    flowchart: { useMaxWidth: false, htmlLabels: false, padding: 16 },
    sequence: { useMaxWidth: false },
});

interface MermaidRendererProps {
    chart: string;
}

const MermaidRenderer: React.FC<MermaidRendererProps> = ({ chart }) => {
    const mermaidRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (mermaidRef.current && chart) {
            // 이전에 렌더링된 내용을 비우고 새로 렌더링
            mermaidRef.current.removeAttribute('data-processed');
            mermaid.contentLoaded();

            // 고유 ID 생성하여 렌더링 시 충돌 방지
            const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;

            try {
                mermaid.render(id, chart).then((result) => {
                    if (mermaidRef.current) {
                        mermaidRef.current.innerHTML = result.svg;
                        const svg = mermaidRef.current.querySelector('svg');
                        if (svg) {
                            svg.removeAttribute('width');
                            svg.removeAttribute('height');
                            svg.style.width = '100%';
                            svg.style.height = 'auto';
                            svg.style.maxWidth = 'none';
                        }
                    }
                });
            } catch (error) {
                console.error('Mermaid render error:', error);
            }
        }
    }, [chart]);

    return (
        <div className="mermaid-container" style={{ margin: '16px 0' }}>
            <div ref={mermaidRef} className="mermaid">
                {chart}
            </div>
        </div>
    );
};

export default MermaidRenderer;
