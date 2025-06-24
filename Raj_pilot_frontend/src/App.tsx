import React from 'react';
import { useConversationStore } from './store/conversationStore';
import AudioPlayer from './components/AudioPlayer';
import Sidebar from './components/Sidebar';
import EditableMessage from './components/EditableMessage';
import Footer from './components/Footer';
import { PanelLeftOpen } from 'lucide-react';

function App() {
  const { projects, activeProjectId, isLoading, error, isSidebarOpen, toggleSidebar } = useConversationStore();

  const activeProject = activeProjectId ? projects[activeProjectId] : null;
  const conversation = activeProject?.data;

  return (
    // The main container is a flex row and a positioning context for the button.
    // The problematic `overflow-hidden` has been REMOVED from this element.
    <div className="h-screen bg-stone-900 text-stone-100 flex flex-row relative">
      
      {/* Button to open sidebar: Positioned absolutely relative to the root container. */}
      {!isSidebarOpen && (
        <button 
          onClick={toggleSidebar}
          className="absolute top-4 left-4 z-30 p-2 bg-stone-800/80 backdrop-blur-sm rounded-full text-stone-300 hover:text-white hover:bg-stone-700"
          title="Open Sidebar"
        >
          <PanelLeftOpen className="w-5 h-5" />
        </button>
      )}

      {/* Sidebar Container: Animates width. */}
      <div className={`
        flex-shrink-0 
        transition-all duration-300 ease-in-out
        ${isSidebarOpen ? 'w-80' : 'w-0'}
      `}>
        {/* This inner div has a fixed width so the content doesn't get squished during animation. */}
        <div className="w-80 h-full overflow-hidden">
           <Sidebar />
        </div>
      </div>
      
      {/* Main Content Area: Takes up remaining space and handles its own internal scrolling. */}
      <div className="flex-1 flex flex-col overflow-hidden">
      
        <AudioPlayer />
        
        <main className="flex-1 overflow-y-auto p-4">
          <div className="max-w-4xl mx-auto">
            {error && (
              <div className="mb-6 p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
                <span className="text-red-400 text-sm">{error}</span>
              </div>
            )}

            {isLoading && (
              <div className="flex items-center justify-center py-12">
                <div className="flex items-center space-x-3">
                  <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-stone-400">Loading conversation...</span>
                </div>
              </div>
            )}

            {!activeProjectId && !isLoading && (
               <div className="flex flex-col items-center justify-center py-12 h-full text-center">
                <div className="text-6xl mb-4">🌾</div>
                <h2 className="text-xl font-semibold text-stone-200">
                  Welcome to KisaanVerse Conversation Editor
                </h2>
                <p className="text-stone-400 max-w-md mt-2">
                  To begin, add a project folder using the sidebar.
                </p>
              </div>
            )}

            {conversation && (
              <div className="space-y-4">
                <div className="text-center py-4 border-b border-stone-700">
                  <h2 className="text-lg font-semibold text-stone-200">
                    Editing: {conversation.metadata?.originalFilename || conversation.conversationId}
                  </h2>
                  <p className="text-sm text-stone-400 mt-1">
                    {conversation.turns.length} turns
                  </p>
                </div>

                {conversation.turns.map((turn) => (
                  <EditableMessage key={turn.id} turn={turn} />
                ))}
              </div>
            )}
          </div>
        </main>
         <Footer />
      </div>
    </div>
  );
}

export default App;