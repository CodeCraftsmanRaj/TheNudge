import { UploadResponse, SaveResponse, ConversationData, ConversationTurn } from '../types/conversation';

const API_BASE = '/api';

// --- Real API functions ---
export const uploadTranscript = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('transcript', file);

  const response = await fetch(`${API_BASE}/upload/transcript`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return response.json();
};

export const uploadAudio = async (file: File): Promise<{ audioUrl: string }> => {
  const formData = new FormData();
  formData.append('audio', file);

  const response = await fetch(`${API_BASE}/upload/audio`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Audio upload failed: ${response.statusText}`);
  }

  return response.json();
};

export const saveConversation = async (conversationId: string, data: ConversationData): Promise<SaveResponse> => {
  const response = await fetch(`${API_BASE}/save/conversation/${conversationId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`Save failed: ${response.statusText}`);
  }

  return response.json();
};


// --- Mock API functions for development ---

/**
 * Processes an uploaded folder containing a master JSON, a master audio, and segment audio files.
 */
export const processUploadedFolder = async (
  files: FileList
): Promise<{ conversation: ConversationData, mainAudioUrl: string | null }> => {
  const fileList = Array.from(files);

  // Find the master JSON file (assuming one JSON file per folder)
  const jsonFile = fileList.find(f => f.name.toLowerCase().endsWith('.json'));
  if (!jsonFile) {
    throw new Error('No JSON file found in the selected folder.');
  }

  // Find the master audio file (e.g., master_audio.mp3, main.wav)
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
            metadata: {
              uploadedAt: new Date().toISOString(),
              originalFilename: jsonFile.name,
              mainAudioFilename: mainAudioFile?.name,
              totalAudioFiles: fileList.filter(f => f.type.startsWith('audio/')).length,
            }
          },
          mainAudioUrl
        });
      } catch (error) {
        console.error("JSON parsing error:", error);
        reject(new Error('Failed to parse the JSON file.'));
      }
    };
    reader.onerror = () => {
      reject(new Error('Failed to read the JSON file.'));
    };
    reader.readAsText(jsonFile);
  });
};

const parseTranscriptData = (data: any, fileList: File[]): ConversationTurn[] => {
  // Determine the array of turns from the JSON structure
  const conversationArray = data.conversation || data.messages || data.turns || (Array.isArray(data) ? data : []);

  if (!Array.isArray(conversationArray) || conversationArray.length === 0) {
    // If no valid conversation data is found, return sample data
    return generateSampleTurns();
  }
  
  let kisaanverseTurnCounter = 0;
  let aliceTurnCounter = 0;

  return conversationArray.map((item, index) => {
    // Determine speaker based on various possible field names
    let speaker: 'alice' | 'kisaanverse' = 'kisaanverse'; // default
    const speakerIdentifier = (item.speaker || item.role || item.name || '').toLowerCase();
    
    if (speakerIdentifier.includes('alice') || speakerIdentifier.includes('user')) {
      speaker = 'alice';
      aliceTurnCounter++;
    } else {
      speaker = 'kisaanverse';
      kisaanverseTurnCounter++;
    }

    // --- Find the corresponding audio file for this turn ---
    let audioUrl: string | undefined = undefined;
    let audioFile: File | undefined = undefined;

    // 1. Try to find audio based on the user's specific naming convention (e.g., audio1.mp3, alice1.mp3)
    let expectedFileNamePrefix: string | null = null;
    if (speaker === 'kisaanverse') {
        expectedFileNamePrefix = `audio${kisaanverseTurnCounter}.`;
        audioFile = fileList.find(f => f.name.toLowerCase().startsWith(expectedFileNamePrefix!));
    } else { // speaker is 'alice'
        // Try finding 'alice1.*' or 'user1.*'
        const alicePrefix = `alice${aliceTurnCounter}.`;
        const userPrefix = `user${aliceTurnCounter}.`;
        audioFile = fileList.find(f => 
            f.name.toLowerCase().startsWith(alicePrefix) || 
            f.name.toLowerCase().startsWith(userPrefix)
        );
    }
    
    // 2. If not found by convention, fall back to the path in the JSON
    if (!audioFile) {
        const audioPath = item.audio_segment_path || item.audioUrl;
        if (audioPath) {
          // Extract filename from the path (e.g., "/path/to/audio.mp3" -> "audio.mp3")
          const audioFilename = audioPath.split(/[\\/]/).pop();
          if (audioFilename) {
            audioFile = fileList.find(f => f.name.toLowerCase() === audioFilename.toLowerCase());
          }
        }
    }

    // 3. If a file was found (by either method), create its URL
    if (audioFile) {
        audioUrl = URL.createObjectURL(audioFile);
    }

    return {
      id: `turn_${index}`,
      speaker: speaker,
      text: item.text || item.content || item.message || '',
      timestamp: item.timestamp || item.time,
      isFlagged: false,
      flagNote: '',
      originalText: item.text || item.content || item.message || '',
      audioUrl: audioUrl,
      audioTimestamp: item.audioTimestamp || item.audio_start
    };
  });
};

const generateSampleTurns = () => [
  {
    id: 'turn_1',
    speaker: 'kisaanverse' as const,
    text: 'नमस्ते Alice! I\'m KisaanVerse, your AI farming assistant. I\'m here to help you with agricultural advice, crop management, weather updates, and sustainable farming practices. How can I assist you with your farming needs today?',
    timestamp: '10:30 AM',
    isFlagged: false,
    flagNote: '',
    originalText: 'नमस्ते Alice! I\'m KisaanVerse, your AI farming assistant. I\'m here to help you with agricultural advice, crop management, weather updates, and sustainable farming practices. How can I assist you with your farming needs today?',
    audioUrl: 'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav' // Sample audio URL
  },
  {
    id: 'turn_2',
    speaker: 'alice' as const,
    text: 'Hi KisaanVerse! I\'m having trouble with my wheat crop. The leaves are turning yellow and I\'m not sure what\'s causing it. Can you help me diagnose the problem?',
    timestamp: '10:31 AM',
    isFlagged: false,
    flagNote: '',
    originalText: 'Hi KisaanVerse! I\'m having trouble with my wheat crop. The leaves are turning yellow and I\'m not sure what\'s causing it. Can you help me diagnose the problem?'
  },
  {
    id: 'turn_3',
    speaker: 'kisaanverse' as const,
    text: 'I understand your concern about the yellowing wheat leaves, Alice. This is a common issue that can have several causes. The most likely reasons include:\n\n1. **Nitrogen deficiency** - This is the most common cause of yellowing leaves\n2. **Overwatering or poor drainage** - Can lead to root problems\n3. **Fungal diseases** like rust or septoria\n4. **Natural senescence** if the crop is nearing maturity\n\nCan you tell me more about your watering schedule, recent weather conditions, and which part of the plant is yellowing first?',
    timestamp: '10:32 AM',
    isFlagged: false,
    flagNote: '',
    originalText: 'I understand your concern about the yellowing wheat leaves, Alice. This is a common issue that can have several causes. The most likely reasons include:\n\n1. **Nitrogen deficiency** - This is the most common cause of yellowing leaves\n2. **Overwatering or poor drainage** - Can lead to root problems\n3. **Fungal diseases** like rust or septoria\n4. **Natural senescence** if the crop is nearing maturity\n\nCan you tell me more about your watering schedule, recent weather conditions, and which part of the plant is yellowing first?',
    audioUrl: 'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav' // Sample audio URL
  },
  {
    id: 'turn_4',
    speaker: 'alice' as const,
    text: 'The yellowing started from the bottom leaves and is moving upward. We\'ve had quite a bit of rain lately, and I haven\'t been fertilizing as much as usual due to the wet conditions.',
    timestamp: '10:35 AM',
    isFlagged: false,
    flagNote: '',
    originalText: 'The yellowing started from the bottom leaves and is moving upward. We\'ve had quite a bit of rain lately, and I haven\'t been fertilizing as much as usual due to the wet conditions.'
  },
  {
    id: 'turn_5',
    speaker: 'kisaanverse' as const,
    text: 'Based on your description, Alice, this sounds like a classic case of **nitrogen deficiency**. The pattern you\'re describing - yellowing starting from bottom leaves and moving upward - is a telltale sign.\n\nThe excessive rain has likely caused two problems:\n1. **Nutrient leaching** - Nitrogen gets washed away from the root zone\n2. **Reduced fertilizer application** - As you mentioned\n\n**Immediate recommendations:**\n- Apply a nitrogen-rich fertilizer (urea or ammonium sulfate)\n- Consider foliar feeding for quick uptake\n- Ensure proper drainage to prevent further nutrient loss\n- Monitor soil pH - it should be 6.0-7.0 for optimal nutrient uptake\n\nWould you like specific fertilizer recommendations based on your soil type and wheat variety?',
    timestamp: '10:37 AM',
    isFlagged: false,
    flagNote: '',
    originalText: 'Based on your description, Alice, this sounds like a classic case of **nitrogen deficiency**. The pattern you\'re describing - yellowing starting from bottom leaves and moving upward - is a telltale sign.\n\nThe excessive rain has likely caused two problems:\n1. **Nutrient leaching** - Nitrogen gets washed away from the root zone\n2. **Reduced fertilizer application** - As you mentioned\n\n**Immediate recommendations:**\n- Apply a nitrogen-rich fertilizer (urea or ammonium sulfate)\n- Consider foliar feeding for quick uptake\n- Ensure proper drainage to prevent further nutrient loss\n- Monitor soil pH - it should be 6.0-7.0 for optimal nutrient uptake\n\nWould you like specific fertilizer recommendations based on your soil type and wheat variety?',
    audioUrl: 'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav' // Sample audio URL
  }
];