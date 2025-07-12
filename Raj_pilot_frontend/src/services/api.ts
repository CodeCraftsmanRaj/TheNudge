import { UploadResponse, SaveResponse, ConversationData, ConversationTurn, FunctionOption } from '../types/conversation';

const API_BASE = '/api';

// --- Real API functions ---
export const uploadTranscript = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('transcript', file);
  const response = await fetch(`${API_BASE}/upload/transcript`, { method: 'POST', body: formData });
  if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`);
  return response.json();
};
export const uploadAudio = async (file: File): Promise<{ audioUrl: string }> => {
  const formData = new FormData();
  formData.append('audio', file);
  const response = await fetch(`${API_BASE}/upload/audio`, { method: 'POST', body: formData });
  if (!response.ok) throw new Error(`Audio upload failed: ${response.statusText}`);
  return response.json();
};
export const saveConversation = async (conversationId: string, data: ConversationData): Promise<SaveResponse> => {
  const response = await fetch(`${API_BASE}/save/conversation/${conversationId}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
  if (!response.ok) throw new Error(`Save failed: ${response.statusText}`);
  return response.json();
};

// --- Mock API functions for development ---
export const processUploadedFolder = async (
  files: FileList
): Promise<{ conversation: ConversationData, mainAudioUrl: string | null }> => {
  const fileList = Array.from(files);
  const jsonFile = fileList.find(f => f.name.toLowerCase().endsWith('.json'));
  if (!jsonFile) throw new Error('No JSON file found in the selected folder.');

  const mainAudioFile = fileList.find(f => f.name.toLowerCase().startsWith('master_audio'));
  const mainAudioUrl = mainAudioFile ? URL.createObjectURL(mainAudioFile) : null;
  
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const jsonData = JSON.parse(e.target?.result as string);
        const turns = parseTranscriptData(jsonData, fileList);
        resolve({
          conversation: {
            conversationId: `conv_${Date.now()}`,
            turns,
            metadata: { uploadedAt: new Date().toISOString(), originalFilename: jsonFile.name, mainAudioFilename: mainAudioFile?.name, totalAudioFiles: fileList.filter(f => f.type.startsWith('audio/')).length }
          },
          mainAudioUrl
        });
      } catch (error) { reject(new Error('Failed to parse the JSON file.')); }
    };
    reader.onerror = () => reject(new Error('Failed to read the JSON file.'));
    reader.readAsText(jsonFile);
  });
};

const parseTranscriptData = (data: any, fileList: File[]): ConversationTurn[] => {
  const conversationArray = data.conversation || data.messages || data.turns || (Array.isArray(data) ? data : []);

  if (!Array.isArray(conversationArray) || conversationArray.length === 0) return [];

  return conversationArray.map((item, index) => {
    let speaker: 'alice' | 'kisaanverse' = 'kisaanverse';
    const speakerIdentifier = (item.speaker || '').toLowerCase();

    if (speakerIdentifier.includes('user') || speakerIdentifier.includes('alice')) {
      speaker = 'alice';
    }

    // Find audio for ANY speaker that has a path in the JSON
    let audioUrl: string | undefined = undefined;
    const audioPath = item.audio_segment_path;
    if (audioPath) {
        const audioFilename = audioPath.split(/[\\/]/).pop();
        if (audioFilename) {
            const audioFile = fileList.find(f => f.name.toLowerCase() === audioFilename.toLowerCase());
            if (audioFile) {
                audioUrl = URL.createObjectURL(audioFile);
            }
        }
    }
    
    // Read `functions_called` from JSON. If it's missing or an empty array, default to ['None'].
    const functionsCalled: FunctionOption[] = speaker === 'kisaanverse' 
      ? (Array.isArray(item.functions_called) && item.functions_called.length > 0 ? item.functions_called : ['None'])
      : [];

    return {
      id: `turn_${index}`,
      speaker: speaker,
      text: item.text || '',
      timestamp: item.timestamp,
      originalText: item.text || '',
      audioUrl: audioUrl,
      audioTimestamp: item.audioTimestamp,
      isFunctionCallWrong: false, // Always starts as false
      functionsCalled: functionsCalled,
      correctFunctions: speaker === 'kisaanverse' ? ['None'] : [],
    };
  });
};