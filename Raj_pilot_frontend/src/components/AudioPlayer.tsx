import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, SkipBack, SkipForward } from 'lucide-react';
import { useConversationStore } from '../store/conversationStore';

const AudioPlayer: React.FC = () => {
  const activeProject = useConversationStore(state => state.activeProjectId ? state.projects[state.activeProjectId] : null);
  const audioUrl = activeProject?.mainAudioUrl;

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.75);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    // When the audio URL changes (e.g., switching projects), reset the player state
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsPlaying(false);
    setCurrentTime(0);
  }, [audioUrl]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioUrl]);

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const togglePlay = () => {
    if (!audioRef.current || !audioUrl) return;
    
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return;
    const newTime = parseFloat(e.target.value);
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    if (audioRef.current) {
      audioRef.current.volume = newVolume;
    }
  };

  const skip = (seconds: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, Math.min(duration, currentTime + seconds));
  };

  if (!audioUrl) {
    return (
      <div className="bg-stone-800 border-b border-stone-700 px-4 py-3 sticky top-0 z-10">
        <div className="flex items-center justify-center max-w-7xl mx-auto">
          <span className="text-sm text-stone-400">No main audio file for this project</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-stone-800 border-b border-stone-700 px-4 py-3 sticky top-0 z-10">
      <audio ref={audioRef} src={audioUrl} key={audioUrl} />
      
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <button onClick={() => skip(-10)} className="p-2 rounded-full bg-stone-700 hover:bg-stone-600" title="Skip back 10s"><SkipBack className="w-4 h-4 text-stone-200" /></button>
            <button onClick={togglePlay} className="p-3 rounded-full bg-emerald-600 hover:bg-emerald-500" disabled={!audioUrl}>
              {isPlaying ? <Pause className="w-5 h-5 text-white" /> : <Play className="w-5 h-5 text-white ml-0.5" />}
            </button>
            <button onClick={() => skip(10)} className="p-2 rounded-full bg-stone-700 hover:bg-stone-600" title="Skip forward 10s"><SkipForward className="w-4 h-4 text-stone-200" /></button>
          </div>
          
          <div className="flex items-center space-x-3">
            <span className="text-sm text-stone-200 font-medium">Main Audio</span>
            <div className="text-xs text-stone-400">{formatTime(currentTime)} / {formatTime(duration)}</div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <input
            type="range"
            min="0"
            max={duration || 0}
            value={currentTime}
            onChange={handleSeek}
            className="w-48 h-1.5 bg-stone-600 rounded-full appearance-none cursor-pointer slider"
          />
          <div className="flex items-center space-x-2">
            <Volume2 className="w-4 h-4 text-stone-400" />
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={handleVolumeChange}
              className="w-20 h-1.5 bg-stone-600 rounded-full appearance-none cursor-pointer slider"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AudioPlayer;