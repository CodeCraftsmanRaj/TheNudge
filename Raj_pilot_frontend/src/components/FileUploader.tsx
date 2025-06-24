import React, { useRef, useState } from 'react';
import { FolderOpen, AlertCircle, CheckCircle } from 'lucide-react';
import { useConversationStore } from '../store/conversationStore';
import { processUploadedFolder } from '../services/api';

// This is a browser-specific property. We need to extend the type.
interface HTMLInputElementWithDirectory extends HTMLInputElement {
  webkitdirectory: boolean;
  directory: boolean;
}

const FileUploader: React.FC = () => {
  const folderInputRef = useRef<HTMLInputElementWithDirectory>(null);
  const [uploadStatus, setUploadStatus] = useState<{ message: string, type: 'success' | 'error' } | null>(null);
  
  const { addProject, setLoading, setError } = useConversationStore();

  const handleFolderUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setLoading(true);
    setError(null); // Clear previous errors
    setUploadStatus({ message: `Processing ${files.length} files...`, type: 'success' });
    
    try {
      const { conversation, mainAudioUrl } = await processUploadedFolder(files);
      
      addProject(conversation, mainAudioUrl);
      
      setUploadStatus({ message: `Project "${conversation.metadata?.originalFilename}" loaded!`, type: 'success' });
      setTimeout(() => setUploadStatus(null), 4000);
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to process folder.';
      setError(errorMessage);
      setUploadStatus({ message: errorMessage, type: 'error' });
      setTimeout(() => setUploadStatus(null), 5000);
    } finally {
      setLoading(false);
      // Reset file input to allow uploading the same folder again
      if (e.target) e.target.value = '';
    }
  };

  const handleButtonClick = () => {
    if (folderInputRef.current) {
        // These properties enable folder selection in the file dialog
        folderInputRef.current.webkitdirectory = true;
        folderInputRef.current.directory = true;
        folderInputRef.current.click();
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-stone-200 mb-3 flex items-center space-x-2">
          <FolderOpen className="w-4 h-4" />
          <span>Add Project</span>
        </h3>
        
        <div className="space-y-3">
          <button
            onClick={handleButtonClick}
            className="w-full p-4 border-2 border-dashed border-stone-600 rounded-lg hover:border-emerald-500 hover:bg-stone-800/50 transition-all duration-200 group"
          >
            <div className="flex flex-col items-center space-y-2">
              <FolderOpen className="w-8 h-8 text-stone-500 group-hover:text-emerald-500 transition-colors" />
              <span className="text-sm text-stone-400 group-hover:text-emerald-400 transition-colors font-medium">
                Add Project Folder
              </span>
              <span className="text-xs text-stone-500">
                Contains master JSON and all audio files
              </span>
            </div>
          </button>
          <input
            ref={folderInputRef}
            type="file"
            onChange={handleFolderUpload}
            className="hidden"
            multiple
          />
        </div>

        {uploadStatus && (
          <div className={`mt-3 p-3 rounded-lg ${
            uploadStatus.type === 'error'
              ? 'bg-red-900/20 border border-red-500/30'
              : 'bg-emerald-900/20 border border-emerald-500/30'
          }`}>
            <div className="flex items-center space-x-2">
              {uploadStatus.type === 'error' ? (
                <AlertCircle className="w-4 h-4 text-red-400" />
              ) : (
                <CheckCircle className="w-4 h-4 text-emerald-400" />
              )}
              <span className={`text-sm ${
                uploadStatus.type === 'error' ? 'text-red-300' : 'text-emerald-300'
              }`}>
                {uploadStatus.message}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUploader;