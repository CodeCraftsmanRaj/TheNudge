import { create } from 'zustand';
import { ConversationData, ConversationTurn, FunctionOption } from '../types/conversation';

interface ConversationProject {
  data: ConversationData;
  mainAudioUrl: string | null;
  hasUnsavedChanges: boolean;
}

type FunctionType = 'functionsCalled' | 'correctFunctions';

interface ConversationStore {
  projects: Record<string, ConversationProject>;
  activeProjectId: string | null;
  isLoading: boolean;
  error: string | null;
  isSidebarOpen: boolean;
  addProject: (conversation: ConversationData, mainAudioUrl: string | null) => void;
  setActiveProject: (projectId: string) => void;
  removeProject: (projectId: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  markSaved: () => void;
  resetStore: () => void;
  toggleSidebar: () => void;
  updateTurn: (turnId: string, updates: Partial<ConversationTurn>) => void;
  updateTurnFunctions: (turnId: string, type: FunctionType, option: FunctionOption) => void;
  toggleWrongFunctionCall: (turnId: string) => void;
}

export const useConversationStore = create<ConversationStore>((set, get) => ({
  projects: {},
  activeProjectId: null,
  isLoading: false,
  error: null,
  isSidebarOpen: true,

  addProject: (conversation, mainAudioUrl) => {
    const projectId = conversation.conversationId;
    const newProject: ConversationProject = { data: conversation, mainAudioUrl, hasUnsavedChanges: false };
    set(state => ({
      projects: { ...state.projects, [projectId]: newProject },
      activeProjectId: projectId,
      error: null,
    }));
  },

  setActiveProject: (projectId) => {
    if (get().projects[projectId]) set({ activeProjectId: projectId });
  },

  removeProject: (projectId) => {
    set(state => {
      const newProjects = { ...state.projects };
      delete newProjects[projectId];
      const remainingProjectIds = Object.keys(newProjects);
      const newActiveProjectId = state.activeProjectId === projectId ? (remainingProjectIds[0] || null) : state.activeProjectId;
      return { projects: newProjects, activeProjectId: newActiveProjectId };
    });
  },

  updateTurn: (turnId, updates) => {
    const { activeProjectId, projects } = get();
    if (!activeProjectId || !projects[activeProjectId]) return;
    set(state => {
      const activeProject = state.projects[activeProjectId];
      const updatedTurns = activeProject.data.turns.map(turn => turn.id === turnId ? { ...turn, ...updates } : turn);
      return { projects: { ...state.projects, [activeProjectId]: { ...activeProject, data: { ...activeProject.data, turns: updatedTurns }, hasUnsavedChanges: true } } };
    });
  },

  updateTurnFunctions: (turnId, type, option) => {
    const { activeProjectId, projects } = get();
    if (!activeProjectId || !projects[activeProjectId]) return;
    set(state => {
      const activeProject = state.projects[activeProjectId];
      const updatedTurns = activeProject.data.turns.map(turn => {
        if (turn.id !== turnId) return turn;
        const newTurn = { ...turn, [type]: [...turn[type]] };
        let currentValues = newTurn[type];
        if (option === 'None') {
          currentValues = ['None'];
        } else {
          currentValues = currentValues.filter(v => v !== 'None');
          const optionIndex = currentValues.indexOf(option);
          if (optionIndex > -1) currentValues.splice(optionIndex, 1);
          else currentValues.push(option);
          if (currentValues.length === 0) currentValues = ['None'];
        }
        newTurn[type] = currentValues;
        return newTurn;
      });
      return { projects: { ...state.projects, [activeProjectId]: { ...activeProject, data: { ...activeProject.data, turns: updatedTurns }, hasUnsavedChanges: true } } };
    });
  },

  toggleWrongFunctionCall: (turnId) => {
    const { activeProjectId, projects } = get();
    if (!activeProjectId || !projects[activeProjectId]) return;
    set(state => {
        const activeProject = state.projects[activeProjectId];
        const updatedTurns = activeProject.data.turns.map(turn => 
            turn.id === turnId 
            ? { ...turn, isFunctionCallWrong: !turn.isFunctionCallWrong } 
            : turn
        );
        return { projects: { ...state.projects, [activeProjectId]: { ...activeProject, data: { ...activeProject.data, turns: updatedTurns }, hasUnsavedChanges: true } } };
    });
  },

  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  markSaved: () => {
    const { activeProjectId } = get();
    if (!activeProjectId) return;
    set(state => ({ projects: { ...state.projects, [activeProjectId]: { ...state.projects[activeProjectId], hasUnsavedChanges: false } } }));
  },
  resetStore: () => set({ projects: {}, activeProjectId: null, error: null, isSidebarOpen: true }),
  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
}));