import React, { useState } from 'react';
import { RotateCcw, Wheat, Cpu, PanelLeftClose, AlertCircle, CheckCircle, Trash2, Folder, FolderCheck } from 'lucide-react';
import { useConversationStore } from '../store/conversationStore';
import FileUploader from './FileUploader';

const Sidebar: React.FC = () => {
  // The Sidebar now gets the global state for visibility and projects
  const { projects, activeProjectId, setActiveProject, removeProject, resetStore, isSidebarOpen, toggleSidebar } = useConversationStore();
  const [scriptStatus, setScriptStatus] = useState<{ message: string, type: 'loading' | 'success' | 'error' } | null>(null);

  const handleReset = () => {
    if (Object.keys(projects).length === 0) return;
    if (window.confirm('This will remove all loaded projects. Are you sure you want to reset the editor?')) {
      resetStore();
    }
  };
  
  const handleRemoveProject = (e: React.MouseEvent, projectId: string) => {
    e.stopPropagation(); // Prevent setActiveProject from firing
    if (window.confirm('Are you sure you want to remove this project?')) {
      removeProject(projectId);
    }
  }

  const handleRunScript = async () => {
    setScriptStatus({ message: 'Executing analysis script...', type: 'loading' });
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      setScriptStatus({ message: 'Script executed successfully.', type: 'success' });
    } catch (error) {
      setScriptStatus({ message: 'Failed to run script. (Dummy)', type: 'error' });
    } finally {
      setTimeout(() => setScriptStatus(null), 5000);
    }
  };

  return (
    // This is the container that animates. It shrinks to w-0 and hides its content.
    <aside className={`
      flex-shrink-0 bg-stone-950 h-full
      transition-all duration-300 ease-in-out
      ${isSidebarOpen ? 'w-80' : 'w-0'}
      overflow-hidden
    `}>
      {/* This inner div has a fixed width, so the content doesn't get squished during animation. */}
      <div className="w-80 h-full flex flex-col border-r border-stone-700">
        <div className="p-4 border-b border-stone-700">
          <div className="flex justify-between items-center mb-1">
            <h2 className="text-lg font-bold text-stone-100 flex items-center space-x-2">
              <span className="text-2xl">🌾</span>
              <span>KisaanVerse Editor</span>
            </h2>
            <button onClick={toggleSidebar} className="p-1.5 rounded-full text-stone-400 hover:text-white hover:bg-stone-700" title="Close Sidebar">
              <PanelLeftClose className="w-5 h-5" />
            </button>
          </div>
          <p className="text-sm text-stone-400 mt-1">Alice & KisaanVerse Conversation Tool</p>
        </div>

        <div className="p-4 space-y-6 flex-1 overflow-y-auto">
          <FileUploader />
          
          {Object.keys(projects).length > 0 && (
            <div className="pt-4 border-t border-stone-700 space-y-2">
              <h3 className="text-sm font-semibold text-stone-200 mb-2">Loaded Projects</h3>
              {Object.values(projects).map(({ data }) => (
                <button
                  key={data.conversationId}
                  onClick={() => setActiveProject(data.conversationId)}
                  className={`w-full flex items-center justify-between p-3 rounded-lg text-left transition-colors group ${
                    activeProjectId === data.conversationId ? 'bg-emerald-900/50' : 'bg-stone-800/30 hover:bg-stone-700/50'
                  }`}
                >
                  <div className="flex items-center space-x-3 overflow-hidden">
                    {activeProjectId === data.conversationId ? <FolderCheck className="w-5 h-5 text-emerald-400 flex-shrink-0" /> : <Folder className="w-5 h-5 text-stone-400 flex-shrink-0" />}
                    <span className="text-sm text-stone-200 truncate" title={data.metadata?.originalFilename}>
                      {data.metadata?.originalFilename || data.conversationId}
                    </span>
                  </div>
                  <div onClick={(e) => handleRemoveProject(e, data.conversationId)} className="p-1 rounded-full opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-stone-500 hover:text-red-400" title="Remove Project">
                    <Trash2 className="w-4 h-4"/>
                  </div>
                </button>
              ))}
            </div>
          )}

          <div className="pt-4 border-t border-stone-700 space-y-3">
            <button
              onClick={handleRunScript}
              disabled={!activeProjectId || !!(scriptStatus && scriptStatus.type === 'loading')}
              className="w-full flex items-center justify-center space-x-2 p-3 bg-sky-700 hover:bg-sky-600 rounded-lg disabled:bg-stone-600 disabled:cursor-not-allowed"
            >
              <Cpu className="w-4 h-4 text-white" />
              <span className="text-sm font-medium text-white">Run Analysis Script</span>
            </button>
            <button
              onClick={handleReset}
              className="w-full flex items-center justify-center space-x-2 p-3 bg-red-700 hover:bg-red-600 rounded-lg"
            >
              <RotateCcw className="w-4 h-4 text-white" />
              <span className="text-sm font-medium text-white">Reset Editor</span>
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;