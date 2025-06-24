export interface ConversationTurn {
  id: string;
  speaker: 'kisaanverse' | 'alice';
  text: string;
  timestamp?: string;
  isFlagged: boolean;
  flagNote: string;
  originalText: string;
  audioUrl?: string; // Individual audio file for each KisaanVerse response
  audioTimestamp?: string; // For audio sync
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