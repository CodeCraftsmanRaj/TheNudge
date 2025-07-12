export type FunctionName = 'Function1' | 'Function2' | 'Function3';
export type FunctionOption = FunctionName | 'None';

export interface ConversationTurn {
  id: string;
  speaker: 'kisaanverse' | 'alice';
  text: string;
  timestamp?: string;
  originalText: string;
  audioUrl?: string; 
  audioTimestamp?: string;

  // New system for annotation
  isFunctionCallWrong: boolean; // Main toggle for showing details
  functionsCalled: FunctionOption[];
  correctFunctions: FunctionOption[];
}

export interface ConversationData {
  conversationId: string;
  turns: ConversationTurn[];
  metadata?: {
    uploadedAt: string;
    originalFilename: string;
    mainAudioFilename?: string;
    totalAudioFiles?: number;
  };
}

export interface UploadResponse {
  conversationId: string;
  turns: ConversationTurn[];
  metadata?: any;
}

export interface SaveResponse {
  status: 'success' | 'error';
  downloadUrl?: string;
  message?: string;
}