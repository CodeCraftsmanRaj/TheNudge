import { create } from 'zustand';
import { ConversationData, ConversationTurn } from '../types/conversation';

// Represents a single uploaded conversation project
interface ConversationProject {
  data: ConversationData;
  mainAudioUrl: string | null;
  hasUnsavedChanges: boolean;
}

interface ConversationStore {
  projects: Record<string, ConversationProject>;
  activeProjectId: string | null;
  isLoading: boolean;
  error: string | null;
  isSidebarOpen: boolean; // Global UI state

  // Actions
  addProject: (conversation: ConversationData, mainAudioUrl: string | null) => void;
  setActiveProject: (projectId: string) => void;
  removeProject: (projectId: string) => void;
  updateTurn: (turnId: string, updates: Partial<ConversationTurn>) => void;
  toggleFlag: (turnId: string) => void;
  updateFlagNote: (turnId: string, note: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  markSaved: () => void;
  resetStore: () => void;
  toggleSidebar: () => void;
}

export const useConversationStore = create<ConversationStore>((set, get) => ({
  projects: {},
  activeProjectId: null,
  isLoading: false,
  error: null,
  isSidebarOpen: true,

  addProject: (conversation, mainAudioUrl) => {
    const projectId = conversation.conversationId;
    const newProject: ConversationProject = {
      data: conversation,
      mainAudioUrl,
      hasUnsavedChanges: false,
    };

    set(state => ({
      projects: {
        ...state.projects,
        [projectId]: newProject,
      },
      activeProjectId: projectId,
      error: null,
    }));
  },

  setActiveProject: (projectId) => {
    if (get().projects[projectId]) {
      set({ activeProjectId: projectId });
    }
  },
  
  removeProject: (projectId) => {
    set(state => {
      const newProjects = { ...state.projects };
      delete newProjects[projectId];
      
      const remainingProjectIds = Object.keys(newProjects);
      const newActiveProjectId = state.activeProjectId === projectId
        ? remainingProjectIds.length > 0 ? remainingProjectIds[0] : null
        : state.activeProjectId;
        
      return {
        projects: newProjects,
        activeProjectId: newActiveProjectId
      };
    });
  },

  updateTurn: (turnId, updates) => {
    const { activeProjectId, projects } = get();
    if (!activeProjectId || !projects[activeProjectId]) return;

    const activeProject = projects[activeProjectId];
    const updatedTurns = activeProject.data.turns.map(turn =>
      turn.id === turnId ? { ...turn, ...updates } : turn
    );

    set(state => ({
      projects: {
        ...state.projects,
        [activeProjectId]: {
          ...activeProject,
          data: { ...activeProject.data, turns: updatedTurns },
          hasUnsavedChanges: true,
        },
      },
    }));
  },

  toggleFlag: (turnId) => {
    const { activeProjectId, projects } = get();
    if (!activeProjectId || !projects[activeProjectId]) return;

    const activeProject = projects[activeProjectId];
    const updatedTurns = activeProject.data.turns.map(t =>
      t.id === turnId ? { ...t, isFlagged: !t.isFlagged, flagNote: !t.isFlagged ? t.flagNote : '' } : t
    );

    set(state => ({
      projects: {
        ...state.projects,
        [activeProjectId]: {
          ...activeProject,
          data: { ...activeProject.data, turns: updatedTurns },
          hasUnsavedChanges: true,
        },
      },
    }));
  },

  updateFlagNote: (turnId, note) => {
    const { activeProjectId, projects } = get();
    if (!activeProjectId || !projects[activeProjectId]) return;
      
    const activeProject = projects[activeProjectId];
    const updatedTurns = activeProject.data.turns.map(turn =>
      turn.id === turnId ? { ...turn, flagNote: note } : turn
    );

    set(state => ({
      projects: {
        ...state.projects,
        [activeProjectId]: {
          ...activeProject,
          data: { ...activeProject.data, turns: updatedTurns },
          hasUnsavedChanges: true,
        },
      },
    }));
  },

  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  
  markSaved: () => {
    const { activeProjectId } = get();
    if (!activeProjectId) return;
    
    set(state => ({
      projects: {
        ...state.projects,
        [activeProjectId]: {
          ...state.projects[activeProjectId],
          hasUnsavedChanges: false,
        },
      },
    }));
  },

  resetStore: () => set({
    projects: {},
    activeProjectId: null,
    error: null,
    isSidebarOpen: true,
  }),

  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
}));