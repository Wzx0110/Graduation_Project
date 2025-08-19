// src/App.tsx

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { ProgressCard } from './components/ProgressCard';
import { ChevronUpIcon, ChevronDownIcon } from './components/Icons';
import type { SubstepData, GenerationStrategy } from './types';

// 定義主要步驟
const MAJOR_STEPS = [
  "影像對齊",
  "故事腳本",
  "資源清理",
  "模型載入",
  "潛在空間插值",
  "圖像生成",
  "影片插幀",
  "影片合成"
];

// 進度數據的結構定義
interface ProgressData {
  [major_step: string]: {
    status: 'pending' | 'running' | 'completed' | 'failed';
    substeps: SubstepData[];
  };
}

// 生成策略選項的定義
const STRATEGIES: { id: GenerationStrategy; name: string }[] = [
  { id: 'basic', name: '基礎版' },
  { id: 'controlnet', name: 'ControlNet' },
  { id: 'lora', name: 'LoRA' },
  { id: 'controlnet_lora', name: 'CNet+LoRA' },
];

function App() {
  // --- 狀態管理 (State Management) ---
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [prompt, setPrompt] = useState<string>("A peaceful sunny day slowly turns into a dramatic thunderstorm");
  const [strength, setStrength] = useState<number>(0.3);
  const [numFrames, setNumFrames] = useState<number>(3); // 為了測試，預設值改小一點
  const [strategy, setStrategy] = useState<GenerationStrategy>('basic');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [progressData, setProgressData] = useState<ProgressData>({});
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isParamsOpen, setIsParamsOpen] = useState<boolean>(true);

  const socket = useRef<WebSocket | null>(null);

  // --- 邏輯函數與 Hooks ---
  const initializeProgress = () => {
    const initialData: ProgressData = {};
    MAJOR_STEPS.forEach(step => {
      initialData[step] = { status: 'pending', substeps: [] };
    });
    setProgressData(initialData);
  };

  useEffect(() => {
    initializeProgress();
  }, []);

  useEffect(() => {
    if (!isLoading) {
      if (socket.current?.readyState === WebSocket.OPEN) socket.current.close();
      return;
    }
    const ws = new WebSocket('ws://localhost:8000/ws');
    socket.current = ws;
    ws.onopen = () => console.log('WebSocket 連接成功');
    ws.onclose = () => console.log('WebSocket 連接斷開');
    ws.onerror = (error) => console.error('WebSocket 錯誤:', error);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('進度更新:', data);
        setProgressData(prev => {
          const newProgress = JSON.parse(JSON.stringify(prev));
          const majorStepKey = data.major_step;
          if (!newProgress[majorStepKey]) newProgress[majorStepKey] = { status: 'pending', substeps: [] };
          const majorStep = newProgress[majorStepKey];
          majorStep.status = data.status;
          if (data.substep) {
            const existingIndex = majorStep.substeps.findIndex((s: SubstepData) => s.name === data.substep.name);
            if (existingIndex > -1) majorStep.substeps[existingIndex] = data.substep;
            else majorStep.substeps.push(data.substep);
          }
          return newProgress;
        });
        const finalVideoUrl = data.substep?.video_url || data.video_url;
        if (finalVideoUrl) {
          setVideoUrl(finalVideoUrl);
        }
        if (data.major_step.includes("任務完成") || data.major_step.includes("任務失敗") || data.status === "failed") {
          setIsLoading(false);
        }
      } catch (error) {
        console.error("解析 WebSocket 消息失敗:", error);
      }
    };
    return () => {
      if (socket.current?.readyState === WebSocket.OPEN) socket.current.close();
    };
  }, [isLoading]);

  const handleSubmit = async () => {
    if (!image1 || !image2) {
      alert('請上傳兩張圖片');
      return;
    }
    setIsLoading(true);
    setIsParamsOpen(false);
    setVideoUrl(null);
    initializeProgress();
    const formData = new FormData();
    formData.append('image1', image1);
    formData.append('image2', image2);
    formData.append('prompt', prompt);
    formData.append('strength', strength.toString());
    formData.append('num_frames', numFrames.toString());
    formData.append('strategy', strategy);
    try {
      axios.post('http://localhost:8000/generate-video', formData);
    } catch (error) {
      console.error('提交請求錯誤:', error);
      setIsLoading(false);
      setProgressData(prev => ({
        ...prev,
        "任務失敗": { status: 'failed', substeps: [{ name: "連接錯誤", status: 'failed', text: "無法連接到後端服務。" }] }
      }));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, setImage: React.Dispatch<React.SetStateAction<File | null>>) => {
    if (e.target.files && e.target.files[0]) setImage(e.target.files[0]);
  };

  const renderImageContent = (file: File | null, side: string) => {
    if (file) return <img src={URL.createObjectURL(file)} alt="preview" className="absolute inset-0 w-full h-full object-cover rounded-xl" />;
    return (
      <div className="flex flex-col items-center justify-center text-sm" style={{ color: 'var(--muted-foreground)' }}>
        <svg className="w-10 h-10 mb-2" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5z" /></svg>
        <span>點擊或拖曳上傳</span>
        <span className="mt-1 text-xs">{side}圖片</span>
      </div>
    );
  };

  return (
    <div className="flex w-full h-screen font-sans overflow-hidden" style={{ color: 'var(--card-foreground)', backgroundColor: 'var(--background)' }}>
      <div className="flex-1 flex flex-col max-h-screen">
        <header className="px-8 py-5 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
          <h1 className="text-xl font-bold">ChronoWeaver</h1>
        </header>
        <main className="flex-1 p-8 flex flex-col gap-6 overflow-y-auto" style={{ minHeight: 0 }}>
          <div className={`transition-all duration-500 ease-in-out ${isLoading || videoUrl ? 'h-32' : 'flex-1'}`} style={{ flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
            <div className="grid grid-cols-2 gap-4 flex-1">
              <div className="space-y-2 flex flex-col">
                <span className="text-sm font-medium" style={{ color: 'var(--muted-foreground)' }}>起點</span>
                <label className="relative flex-1 w-full border border-dashed rounded-xl flex items-center justify-center p-2 cursor-pointer" style={{ backgroundColor: 'var(--card)', borderColor: 'var(--border)' }}>
                  {renderImageContent(image1, '起點')}
                  <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setImage1)} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
                </label>
              </div>
              <div className="space-y-2 flex flex-col">
                <span className="text-sm font-medium" style={{ color: 'var(--muted-foreground)' }}>終點</span>
                <label className="relative flex-1 w-full border border-dashed rounded-xl flex items-center justify-center p-2 cursor-pointer" style={{ backgroundColor: 'var(--card)', borderColor: 'var(--border)' }}>
                  {renderImageContent(image2, '終點')}
                  <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setImage2)} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
                </label>
              </div>
            </div>
          </div>
          <div className={`rounded-xl transition-all duration-300 ${!isLoading && !videoUrl ? 'flex-1 flex flex-col' : ''}`} style={{ backgroundColor: 'var(--card)', border: '1px solid var(--border)', minHeight: 0 }}>
            <button className="w-full p-6 flex justify-between items-center flex-shrink-0" onClick={() => setIsParamsOpen(!isParamsOpen)}>
              <h2 className="text-lg font-semibold">故事線與參數</h2>
              {isParamsOpen ? <ChevronUpIcon className="w-5 h-5" /> : <ChevronDownIcon className="w-5 h-5" />}
            </button>
            <div className="grid transition-all duration-300 ease-in-out" style={{ gridTemplateRows: isParamsOpen ? '1fr' : '0fr' }}>
              <div className={`overflow-hidden ${!isLoading && !videoUrl ? 'flex flex-col flex-1' : ''}`}>
                <div className={`p-6 pt-0 space-y-6 ${!isLoading && !videoUrl ? 'flex flex-col flex-1' : ''}`}>
                  <div className="space-y-2">
                    <label htmlFor="prompt" className="text-base font-semibold">故事線</label>
                    <textarea id="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} className="w-full p-3 rounded-lg text-base border transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[color:var(--accent)]" style={{ backgroundColor: 'var(--background)', borderColor: 'var(--border)', color: 'var(--foreground)' }} placeholder="描述一個從起點到終點的漸變故事..." />
                  </div>
                  <div className={`space-y-4 pt-2 ${!isLoading && !videoUrl ? 'flex-1 flex flex-col justify-end' : ''}`}>
                    <h3 className="text-base font-semibold">參數</h3>
                    <div className='space-y-4'>
                      <div className='grid grid-cols-[100px_1fr_auto] items-center gap-x-4'>
                        <label htmlFor="strength" className="text-sm font-medium" style={{ color: 'var(--muted-foreground)' }}>生成強度</label>
                        <input id="strength" type="range" min="0" max="1" step="0.05" value={strength} onChange={(e) => setStrength(parseFloat(e.target.value))} />
                        <span className="text-sm font-mono w-12 text-center py-1 rounded" style={{ backgroundColor: 'var(--background)' }}>{strength.toFixed(2)}</span>
                      </div>
                      <div className='grid grid-cols-[100px_1fr_auto] items-center gap-x-4'>
                        <label htmlFor="frames" className="text-sm font-medium" style={{ color: 'var(--muted-foreground)' }}>中間關鍵幀</label>
                        <input id="frames" type="range" min={3} max={10} step={1} value={numFrames} onChange={(e) => setNumFrames(parseInt(e.target.value))} />
                        <span className="text-sm font-mono w-12 text-center py-1 rounded" style={{ backgroundColor: 'var(--background)' }}>{numFrames}</span>
                      </div>
                      <div className='grid grid-cols-[100px_1fr] items-center gap-x-4'>
                        <label className="text-sm font-medium" style={{ color: 'var(--muted-foreground)' }}>生成策略</label>
                        <div className="p-1 rounded-lg flex items-center" style={{ backgroundColor: 'var(--background)' }}>
                          {STRATEGIES.map((s) => (<button key={s.id} onClick={() => setStrategy(s.id)} className={`flex-1 py-1.5 text-xs rounded-md transition-colors duration-200 ${strategy === s.id ? 'font-semibold' : ''}`} style={{ backgroundColor: strategy === s.id ? 'var(--secondary)' : 'transparent', color: strategy === s.id ? 'var(--foreground)' : 'var(--muted-foreground)' }}> {s.name} </button>))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          {(isLoading || videoUrl) && (
            <div className="p-6 rounded-xl space-y-4 flex flex-col flex-1" style={{ backgroundColor: 'var(--card)', border: '1px solid var(--border)', minHeight: 0 }}>
              <h2 className="text-lg font-semibold flex-shrink-0">生成結果</h2>
              <div className="w-full flex-1 rounded-xl flex items-center justify-center text-sm" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--muted-foreground)' }}>
                {isLoading && !videoUrl && <p>影片生成中，請稍候...</p>}
                {!isLoading && videoUrl === null && <p>任務已結束，但未生成影片。</p>}
                {videoUrl && (<video src={videoUrl} controls autoPlay muted loop className="w-full h-full object-contain rounded-xl" />)}
              </div>
            </div>
          )}
        </main>
        <footer className="p-4 border-t flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
          <button onClick={handleSubmit} disabled={isLoading} className="w-full py-3 px-5 text-base font-semibold text-center rounded-lg transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-50 hover:shadow-[0_0_20px_var(--accent)]" style={{ backgroundColor: 'var(--primary)', color: 'var(--primary-foreground)' }}>
            {isLoading ? '生成中...' : '開始生成影片'}
          </button>
        </footer>
      </div>
      <aside className="w-[500px] border-l flex flex-col flex-shrink-0 max-h-screen" style={{ backgroundColor: 'var(--background)', borderColor: 'var(--border)' }}>
        <header className="p-6 flex items-center justify-between border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
          <h2 className="text-lg font-semibold">執行進度</h2>
        </header>
        <div className="flex-grow overflow-y-auto p-6">
          <ol className="relative border-l pl-6" style={{ borderColor: 'var(--border)' }}>
            {Object.keys(progressData).length > 0 && MAJOR_STEPS.map((step) => (
              <li key={step} className="pb-8 last:pb-0">
                <ProgressCard title={step} status={progressData[step]?.status || 'pending'} substeps={progressData[step]?.substeps || []} />
              </li>
            ))}
            {progressData["任務失敗"]?.status === 'failed' && (<li className="mt-4"><ProgressCard title="任務失敗" status="failed" substeps={progressData["任務失敗"].substeps} /></li>)}
          </ol>
        </div>
      </aside>
    </div>
  );
}

export default App;