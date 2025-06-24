import React, { useState } from 'react';
import { Download, Save, AlertCircle, CheckCircle } from 'lucide-react';
import { useConversationStore } from '../store/conversationStore';

const Footer: React.FC = () => {
  const { markSaved } = useConversationStore();
  const activeProject = useConversationStore(state => state.activeProjectId ? state.projects[state.activeProjectId] : null);
  
  const conversation = activeProject?.data;
  const hasUnsavedChanges = activeProject?.hasUnsavedChanges;

  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string>('');

  const handleSaveAndDownload = async () => {
    if (!conversation) return;

    setIsSaving(true);
    setSaveStatus('Saving conversation...');

    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const editedData = {
        ...conversation,
        savedAt: new Date().toISOString(),
      };

      const blob = new Blob([JSON.stringify(editedData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `edited_conversation_${conversation.conversationId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      markSaved();
      setSaveStatus('Conversation saved and downloaded!');
      setTimeout(() => setSaveStatus(''), 3000);
    } catch (error) {
      setSaveStatus('Failed to save conversation');
      setTimeout(() => setSaveStatus(''), 3000);
    } finally {
      setIsSaving(false);
    }
  };

  const getEditSummary = () => {
    if (!conversation) return null;
    const editedCount = conversation.turns.filter(t => t.text !== t.originalText).length;
    const flaggedCount = conversation.turns.filter(t => t.isFlagged).length;
    return { editedCount, flaggedCount };
  };

  const summary = getEditSummary();

  return (
    <div className="bg-stone-950 border-t border-stone-700 px-4 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <span className="text-sm text-stone-400">KisaanVerse Editor</span>
          {summary && (
            <div className="flex items-center space-x-4 text-xs text-stone-500">
              <span>{summary.editedCount} edited</span>
              <span>{summary.flaggedCount} flagged</span>
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-4">
          {saveStatus && (
            <div className={`flex items-center space-x-2 text-sm ${saveStatus.includes('Failed') ? 'text-red-400' : 'text-emerald-400'}`}>
              {saveStatus.includes('Failed') ? <AlertCircle className="w-4 h-4" /> : <CheckCircle className="w-4 h-4" />}
              <span>{saveStatus}</span>
            </div>
          )}

          {hasUnsavedChanges && (
            <div className="flex items-center space-x-2 text-amber-400">
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm">Unsaved changes</span>
            </div>
          )}
          
          <button
            onClick={handleSaveAndDownload}
            disabled={!conversation || isSaving}
            className="flex items-center space-x-2 px-6 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-stone-600 disabled:cursor-not-allowed rounded-lg"
          >
            {isSaving ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span className="text-sm font-medium">Saving...</span>
              </>
            ) : (
              <>
                <Save className="w-4 h-4" />
                <Download className="w-4 h-4" />
                <span className="text-sm font-medium">Save & Download</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Footer;