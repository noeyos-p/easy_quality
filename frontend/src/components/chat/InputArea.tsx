interface InputAreaProps {
  inputMessage: string
  isLoading: boolean
  selectedDocs: string[]
  suggestions: { id: number; name: string }[]
  showSuggestions: boolean
  suggestionIndex: number
  onInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void
  onKeyDown: (e: React.KeyboardEvent) => void
  onSend: () => void
  onSelectSuggestion: (name: string) => void
  onRemoveDoc: (docId: string) => void
}

export default function InputArea({
  inputMessage,
  isLoading,
  selectedDocs,
  suggestions,
  showSuggestions,
  suggestionIndex,
  onInputChange,
  onKeyDown,
  onSend,
  onSelectSuggestion,
  onRemoveDoc,
}: InputAreaProps) {
  return (
    <div className="border-t border-dark-border p-3 bg-dark-deeper">
      <div className="relative bg-dark-light border border-dark-border rounded-md px-2.5 py-1.5 transition-all duration-200 flex flex-row flex-wrap items-center gap-1.5 min-h-[40px] focus-within:border-accent focus-within:shadow-[0_0_0_1px_rgba(34,209,66,0.2)]">
        {selectedDocs.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {selectedDocs.map(docId => (
              <div key={docId} className="flex items-center gap-1 bg-accent/10 border border-accent px-1.5 py-[1px] rounded">
                <span className="text-[11px] text-accent font-medium">{docId}</span>
                <button
                  className="bg-transparent border-none text-accent cursor-pointer text-[12px] p-0 flex items-center justify-center leading-none"
                  onClick={() => onRemoveDoc(docId)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <textarea
          value={inputMessage}
          onChange={onInputChange}
          onKeyDown={onKeyDown}
          placeholder={selectedDocs.length > 0 ? '' : 'Ask the Agent...And Tag with @'}
          className="agent-input flex-1 min-w-[120px] bg-transparent border-none py-1.5 text-txt-primary text-[13px] resize-none min-h-[24px] max-h-[120px] font-[inherit] focus:outline-none placeholder:text-[#6a6a6a]"
          rows={1}
        />

        {showSuggestions && (
          <div className="absolute bottom-full left-0 w-full max-h-[200px] overflow-y-auto bg-dark-light border border-dark-border rounded shadow-[0_-4px_12px_rgba(0,0,0,0.5)] z-[1000] mb-1">
            {suggestions.map((doc, idx) => (
              <div
                key={doc.id}
                className={`px-3 py-2 cursor-pointer text-[13px] transition-colors duration-200 ${idx === suggestionIndex ? 'bg-dark-hover text-accent' : 'text-txt-primary hover:bg-dark-hover hover:text-accent'}`}
                onClick={() => onSelectSuggestion(doc.name)}
              >
                {doc.name}
              </div>
            ))}
          </div>
        )}

        <button
          className="bg-accent text-black border-none py-1.5 px-3 rounded font-semibold cursor-pointer transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:enabled:bg-accent-hover"
          onClick={onSend}
          disabled={isLoading || !inputMessage.trim()}
        >
          &gt;
        </button>
      </div>
    </div>
  )
}
