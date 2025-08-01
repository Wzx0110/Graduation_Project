// src/App.tsx

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { CloseIcon, SettingsIcon } from './components/Icons';
import { ProgressCard } from './components/ProgressCard';
import type { SubstepData } from './types';

interface ProgressData {
  [major_step: string]: {
    status: 'pending' | 'running' | 'completed' | 'failed';
    substeps: SubstepData[];
  };
}

// 定義專案的所有主要步驟，確保前端渲染順序
const MAJOR_STEPS = ["影像對齊", "故事腳本", "圖像生成", "影片合成"];

function App() {
  // --- 狀態管理 (State Management) ---
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [prompt, setPrompt] = useState<string>("A peaceful sunny day slowly turns into a dramatic thunderstorm");
  const [strength, setStrength] = useState<number>(0.7);
  const [numFrames, setNumFrames] = useState<number>(10);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);
  const [progressData, setProgressData] = useState<ProgressData>({});

  const socket = useRef<WebSocket | null>(null);

  // --- 函數與邏輯 (Functions & Logic) ---

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
      if (socket.current?.readyState === WebSocket.OPEN) {
        socket.current.close();
      }
      return;
    }

    socket.current = new WebSocket('ws://localhost:8000/ws');
    socket.current.onopen = () => console.log('WebSocket Connected');
    socket.current.onclose = () => console.log('WebSocket Disconnected');

    socket.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Progress Update:', data);

        // **使用函數式更新和深度拷貝來保證順序和狀態安全**
        setProgressData(prev => {
          const newProgress = JSON.parse(JSON.stringify(prev));
          const majorStepKey = data.major_step;

          if (!newProgress[majorStepKey]) {
            newProgress[majorStepKey] = { status: 'pending', substeps: [] };
          }

          const majorStep = newProgress[majorStepKey];
          majorStep.status = data.status;

          if (data.substep) {
            const existingIndex: number = majorStep.substeps.findIndex((s: SubstepData) => s.name === data.substep.name);
            if (existingIndex > -1) {
              majorStep.substeps[existingIndex] = data.substep;
            } else {
              majorStep.substeps.push(data.substep);
            }
          }

          return newProgress;
        });

        if (data.major_step.includes("任務完成") || data.major_step.includes("任務失敗") || data.status === "failed") {
          setIsLoading(false);
        }
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    return () => {
      if (socket.current?.readyState === WebSocket.OPEN) {
        socket.current.close();
      }
    };
  }, [isLoading]);

  const handleSubmit = async () => {
    if (!image1 || !image2) {
      alert('請上傳兩張圖片');
      return;
    }
    setIsLoading(true);
    setIsSidebarOpen(true);
    initializeProgress();

    const formData = new FormData();
    formData.append('image1', image1);
    formData.append('image2', image2);
    formData.append('prompt', prompt);
    formData.append('strength', strength.toString());
    formData.append('num_frames', numFrames.toString());

    try {
      await axios.post('http://localhost:8000/generate-video', formData);
    } catch (error) {
      console.error('Submit Error:', error);
      setIsLoading(false);
      setProgressData(prev => ({
        ...prev,
        "任務失敗": { status: 'failed', substeps: [{ name: "連接錯誤", status: 'failed', text: "無法連接到後端服務，請檢查後端是否已啟動，或查看瀏覽器開發者控制台的錯誤訊息。" }] }
      }));
    }
  };

  const renderImagePreview = (file: File | null, side: string) => {
    const content = file
      ? <img src={URL.createObjectURL(file)} alt="preview" className="w-full h-full object-contain" />
      : <div className="text-gray-400">點擊或拖曳上傳{side}圖片</div>;
    return <div className="w-full h-56 bg-gray-800 border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center p-2 hover:bg-gray-700 transition-colors">{content}</div>;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, setImage: React.Dispatch<React.SetStateAction<File | null>>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  return (
    <div className="bg-gray-900 min-h-screen text-white font-sans flex">
      <main
        className="flex-1 p-8 flex flex-col space-y-6 transition-all duration-300 ease-in-out overflow-y-auto"
        style={{ marginRight: isSidebarOpen ? '600px' : '0' }}
      >
        <header className='flex items-center space-x-3'>
          <svg className="w-10 h-10 text-cyan-400" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M6.115 5.19l.319 1.913A6 6 0 0012 12h0a6 6 0 005.566-4.897l.319-1.913A2.25 2.25 0 0015.385 3H8.615a2.25 2.25 0 00-2.5 2.19z" /><path strokeLinecap="round" strokeLinejoin="round" d="M12 12l-3 7.5m3-7.5l3 7.5M12 12l-3-7.5m3 7.5l3-7.5" /></svg>
          <h1 className="text-4xl font-bold text-gray-100">ChronoWeaver</h1>
        </header>
        <p className="text-gray-400">從兩張圖片和一個故事，編織時光的影片。</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="cursor-pointer">
            <span className="block mb-2 text-sm font-medium">起點圖片</span>
            {renderImagePreview(image1, '起點')}
            <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setImage1)} className="hidden" />
          </label>
          <label className="cursor-pointer">
            <span className="block mb-2 text-sm font-medium">終點圖片</span>
            {renderImagePreview(image2, '終點')}
            <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setImage2)} className="hidden" />
          </label>
        </div>

        <div>
          <label htmlFor="prompt" className="block mb-2 text-sm font-medium">故事線 (Story Prompt)</label>
          <textarea id="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} className="w-full p-2.5 bg-gray-800 border border-gray-600 rounded-lg focus:ring-cyan-500 focus:border-cyan-500" placeholder="e.g., A peaceful sunny day slowly turns into a dramatic thunderstorm..."></textarea>
        </div>

        <div className='grid grid-cols-1 md:grid-cols-2 gap-6 pt-4'>
          <div>
            <label htmlFor="strength" className="block mb-2 text-sm font-medium">生成強度: {strength}</label>
            <input id="strength" type="range" min="0" max="1" step="0.05" value={strength} onChange={(e) => setStrength(parseFloat(e.target.value))} className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer" />
          </div>
          <div>
            <label htmlFor="frames" className="block mb-2 text-sm font-medium">中間關鍵幀: {numFrames}</label>
            <input id="frames" type="range" min="10" max="50" step="10" value={numFrames} onChange={(e) => setNumFrames(parseInt(e.target.value))} className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer" />
          </div>
        </div>

        <div className="pt-4">
          <button onClick={handleSubmit} disabled={isLoading} className="w-full py-3 px-5 text-base font-medium text-center text-white bg-cyan-600 rounded-lg hover:bg-cyan-700 focus:ring-4 focus:outline-none focus:ring-cyan-800 disabled:bg-gray-600 disabled:cursor-not-allowed">
            {isLoading ? '正在編織中...' : '開始編織影片'}
          </button>
        </div>
      </main>

      <div
        className={`fixed top-0 right-0 h-full w-[600px] bg-gray-800/90 backdrop-blur-md border-l border-gray-700 shadow-2xl transition-transform duration-300 ease-in-out z-10 
                   ${isSidebarOpen ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="p-6 h-full flex flex-col">
          <div className="flex items-center justify-between mb-6 flex-shrink-0">
            <h2 className="text-2xl font-semibold">執行進度</h2>
            <button onClick={() => setIsSidebarOpen(false)} className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded-full transition-colors">
              <CloseIcon className="w-6 h-6" />
            </button>
          </div>

          <div className="flex-grow overflow-y-auto pr-4">
            <ol className="relative border-l-2 border-gray-700 pl-4">
              {Object.keys(progressData).length > 0 && MAJOR_STEPS.map((step) => (
                <li key={step} className="pb-8 last:pb-0">
                  <ProgressCard
                    title={step}
                    status={progressData[step]?.status || 'pending'}
                    substeps={progressData[step]?.substeps || []}
                  />
                </li>
              ))}
              {progressData["任務失敗"] && (
                <li className="ml-4 mt-4">
                  <ProgressCard title="任務失敗" status="failed" substeps={progressData["任務失敗"].substeps} />
                </li>
              )}
            </ol>
          </div>
        </div>
      </div>

      {!isSidebarOpen && (
        <button
          onClick={() => setIsSidebarOpen(true)}
          className="fixed top-6 right-6 z-20 p-2 bg-gray-700/50 rounded-full hover:bg-gray-600 transition-opacity"
        >
          <SettingsIcon className="w-6 h-6 text-white" />
        </button>
      )}
    </div>
  );
}

export default App;