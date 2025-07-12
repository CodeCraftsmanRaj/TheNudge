import React, { useState, useRef, useEffect } from 'react';
import { Check, X, Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import { ConversationTurn, FunctionOption } from '../types/conversation';
import { useConversationStore } from '../store/conversationStore';

interface EditableMessageProps {
  turn: ConversationTurn;
}

const functionOptions: FunctionOption[] = ['None', 'Function1', 'Function2', 'Function3'];

const FunctionCheckboxGroup: React.FC<{
  title: string;
  turnId: string;
  type: 'functionsCalled' | 'correctFunctions';
  selectedOptions: FunctionOption[];
  disabled?: boolean;
}> = ({ title, turnId, type, selectedOptions, disabled = false }) => {
  const { updateTurnFunctions } = useConversationStore();
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-semibold text-stone-300">{title}</h4>
      <div className="flex items-center space-x-4 flex-wrap">
        {functionOptions.map(option => (
          <label key={option} className={`flex items-center space-x-2 p-1 ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}>
            <input
              type="checkbox"
              checked={selectedOptions.includes(option)}
              onChange={() => updateTurnFunctions(turnId, type, option)}
              disabled={disabled}
              className="w-4 h-4 text-emerald-500 bg-stone-700 border-stone-500 rounded focus:ring-emerald-500 focus:ring-2 disabled:cursor-not-allowed"
            />
            <span className="text-sm text-stone-300 capitalize">{option}</span>
          </label>
        ))}
      </div>
    </div>
  );
};

const EditableMessage: React.FC<EditableMessageProps> = ({ turn }) => {
  const { updateTurn, toggleWrongFunctionCall } = useConversationStore();
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(turn.text);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      const onTimeUpdate = () => setCurrentTime(audio.currentTime);
      const onLoadedMetadata = () => setDuration(audio.duration);
      const onEnded = () => setIsPlaying(false);
      audio.addEventListener('timeupdate', onTimeUpdate);
      audio.addEventListener('loadedmetadata', onLoadedMetadata);
      audio.addEventListener('ended', onEnded);
      return () => {
        audio.removeEventListener('timeupdate', onTimeUpdate);
        audio.removeEventListener('loadedmetadata', onLoadedMetadata);
        audio.removeEventListener('ended', onEnded);
      };
    }
  }, [turn.audioUrl]);

  const handleSaveEdit = () => {
    updateTurn(turn.id, { text: editText });
    setIsEditing(false);
  };
  const toggleAudioPlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!audioRef.current) return;
    isPlaying ? audioRef.current.pause() : audioRef.current.play().catch(console.error);
    setIsPlaying(!isPlaying);
  };
  const formatTime = (seconds: number) => {
    if (isNaN(seconds) || seconds === Infinity) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  const skip = (seconds: number) => {
    if (audioRef.current) audioRef.current.currentTime = Math.max(0, Math.min(duration, audioRef.current.currentTime + seconds));
  };
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (audioRef.current) audioRef.current.currentTime = parseFloat(e.target.value);
  };

  const isKisaanVerse = turn.speaker === 'kisaanverse';

  return (
    <div className={`flex ${isKisaanVerse ? 'justify-start' : 'justify-end'} mb-6 group`}>
      <div className={`flex max-w-[85%] ${isKisaanVerse ? 'flex-row' : 'flex-row-reverse'} items-start space-x-3`}>
        <div className="flex-shrink-0">
          <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold ${isKisaanVerse ? 'bg-emerald-700' : 'bg-amber-600'} text-white`}>
            {isKisaanVerse ? '🌾' : '👩‍🌾'}
          </div>
        </div>
        <div className={`flex flex-col ${isKisaanVerse ? 'items-start' : 'items-end'} w-full`}>
          <div className={`flex items-center space-x-2 mb-2 ${isKisaanVerse ? 'flex-row' : 'flex-row-reverse'}`}>
            <span className="text-sm text-stone-300 font-semibold">{isKisaanVerse ? 'KisaanVerse AI' : 'Alice'}</span>
            {turn.text !== turn.originalText && <span className="text-xs text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded-full">Edited</span>}
          </div>

          {turn.audioUrl && (
            <div className="mb-2 w-full max-w-md">
              <audio ref={audioRef} src={turn.audioUrl} preload="metadata" />
              <div className="bg-stone-800/50 rounded-lg p-3 border border-stone-600/30 space-y-2">
                <div className="flex items-center space-x-3">
                  <button onClick={() => skip(-5)} className="p-1.5 rounded-full text-stone-400 hover:text-white hover:bg-stone-700" title="Skip back 5s"><SkipBack className="w-4 h-4" /></button>
                  <button onClick={toggleAudioPlay} className="p-2 rounded-full bg-emerald-600 hover:bg-emerald-500" title={isPlaying ? 'Pause' : 'Play'}>{isPlaying ? <Pause className="w-5 h-5 text-white" /> : <Play className="w-5 h-5 text-white ml-0.5" />}</button>
                  <button onClick={() => skip(5)} className="p-1.5 rounded-full text-stone-400 hover:text-white hover:bg-stone-700" title="Skip forward 5s"><SkipForward className="w-4 h-4" /></button>
                  <span className="text-xs text-stone-400">{formatTime(currentTime)} / {formatTime(duration)}</span>
                </div>
                <input type="range" min="0" max={duration || 0} step="0.1" value={currentTime} onChange={handleSeek} className="w-full h-1.5 bg-stone-600 rounded-full appearance-none cursor-pointer slider"/>
              </div>
            </div>
          )}

          <div className={`relative p-4 rounded-2xl shadow-lg w-full border ${isKisaanVerse ? 'bg-stone-800 rounded-tl-sm border-stone-600/30' : 'bg-amber-700 text-white rounded-tr-sm border-amber-600/30'}`}>
            {isEditing ? (
              <div className="space-y-3">
                <textarea ref={textareaRef} value={editText} onChange={(e) => setEditText(e.target.value)} className="w-full bg-transparent resize-none border-none outline-none focus:ring-0 p-0 text-sm"/>
                <div className="flex justify-end space-x-2 pt-2 border-t border-current/20">
                  <button onClick={() => setIsEditing(false)} className="p-1 text-current/60 hover:text-current" title="Cancel"><X className="w-4 h-4" /></button>
                  <button onClick={handleSaveEdit} className="p-1 text-emerald-400 hover:text-emerald-300" title="Save"><Check className="w-4 h-4" /></button>
                </div>
              </div>
            ) : ( <p className="text-sm leading-relaxed whitespace-pre-wrap cursor-text" onClick={() => setIsEditing(true)} title="Click to edit">{turn.text}</p> )}
          </div>
          
          {isKisaanVerse && (
            <div className="mt-3 space-y-4 w-full p-4 bg-stone-800/40 border border-stone-700/50 rounded-lg">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={turn.isFunctionCallWrong}
                  onChange={() => toggleWrongFunctionCall(turn.id)}
                  className="w-4 h-4 text-red-500 bg-stone-700 border-stone-500 rounded focus:ring-red-500 focus:ring-2"
                />
                <span className="text-sm font-semibold text-red-400">Wrong Function Call</span>
              </label>

              {turn.isFunctionCallWrong && (
                <div className="pl-6 pt-4 border-t border-stone-700/50 space-y-4">
                  <FunctionCheckboxGroup
                    title="Function Called"
                    turnId={turn.id}
                    type="functionsCalled"
                    selectedOptions={turn.functionsCalled}
                    disabled={true}
                  />
                  <div className="border-t border-stone-700/50"></div>
                  <FunctionCheckboxGroup
                    title="Correct Functions to call"
                    turnId={turn.id}
                    type="correctFunctions"
                    selectedOptions={turn.correctFunctions}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EditableMessage;