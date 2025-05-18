import { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import ResultDisplay from "./components/ResultDisplay";
import OptionalTextInput from "./components/OptionalTextInput";
import {
  ArrowUpTrayIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowPathIcon,
  FilmIcon
} from '@heroicons/react/24/outline'; // 使用 outline 風格圖標

// 定義處理結果的介面
interface ProcessResult {
  outputUrl?: string;
  text: string;
  status: 'idle' | 'processing' | 'success' | 'error';
}

// 定義預設提示文字
const DEFAULT_PROMPT = ""; // 預設提示文字

function App() {
  const [files, setFiles] = useState<{ [key: number]: File | null }>({ 0: null, 1: null });
  const [result, setResult] = useState<ProcessResult>({ text: '請上傳圖片以開始', status: 'idle', outputUrl: undefined });
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [optionalPrompt, setOptionalPrompt] = useState<string>(DEFAULT_PROMPT);

  const handleUploadAll = async () => {
    if (!files[0] || !files[1]) {
      setError("請確保已上傳兩張圖片");
      setResult(prev => ({ ...prev, status: 'error' }));
      return;
    }

    setIsProcessing(true);
    setResult({ text: '正在處理圖像...', status: 'processing', outputUrl: undefined });
    setError(null);

    const formData = new FormData();
    if (files[0]) formData.append("image1", files[0]);
    if (files[1]) formData.append("image2", files[1]);
    formData.append("prompt", optionalPrompt);

    try {
      const response = await fetch("http://127.0.0.1:8000/Process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorData: any = null;
        try {
          errorData = await response.json();
        } catch (e) { }
        console.error("Server Error:", response.status, errorData);
        throw new Error(errorData?.detail || `影片處理失敗 (${response.status})`); // 錯誤訊息
      }

      // 後端回應的是影片檔案 (mp4)
      const blob = await response.blob();
      const outputUrl = URL.createObjectURL(blob);
      setResult({
        outputUrl: outputUrl,
        text: "影片處理完成！",
        status: 'success'
      });
    } catch (error: any) {
      console.error("上傳或處理錯誤", error);
      const errorMessage = error instanceof Error ? error.message : "發生未知錯誤，請重試";
      setError(errorMessage);
      setResult({ text: '處理失敗', status: 'error', outputUrl: undefined });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleIndividualUpload = (id: number, file: File | null) => {
    setFiles((prev) => ({ ...prev, [id]: file }));
    if (error) {
      setError(null);
      if (result.status === 'error') {
        setResult({ text: '請上傳圖片以開始', status: 'idle', outputUrl: undefined });
      }
    }
    // 當用戶重新上傳圖片時，可以重置結果狀態回 idle
    if (result.status === 'success' || result.status === 'error') {
      // 重置 outputUrl
      setResult({ text: '準備就緒，可以開始處理', status: 'idle', outputUrl: undefined });
    }
  };
  const handlePromptChange = (newValue: string) => { // 更新提示文字
    setOptionalPrompt(newValue);
  };
  // 判斷按鈕是否應該被禁用
  const isButtonDisabled = isProcessing || !files[0] || !files[1];

  return (
    // 頁面背景和整體佈局
    <div className="min-h-screen bg-gray-100 py-10 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">

        {/* 頁面標題 */}
        <header className="mb-10 text-center">
          <h1 className="text-4xl font-bold text-gray-800 tracking-tight">
            圖像轉影片工具 {/* 標題 */}
          </h1>
          <p className="mt-2 text-lg text-gray-600">
            上傳兩張圖片，生成它們之間的影片 {/* 描述 */}
          </p>
        </header>

        {/* 主要內容區域：左右佈局 */}
        <main className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">

          {/* 左側：輸入和控制區域 */}
          <section className="bg-white rounded-xl shadow-lg border border-gray-200 p-6 md:p-8 flex flex-col space-y-6">
            <h2 className="text-2xl font-semibold text-gray-700 border-b pb-3 mb-4">
              輸入圖片
            </h2>

            {/* 兩個圖片上傳元件 */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <ImageUploader id={0} title="圖片 A" onUpload={handleIndividualUpload} />
              <ImageUploader id={1} title="圖片 B" onUpload={handleIndividualUpload} />
            </div>

            {/* */}
            <OptionalTextInput
              id="prompt-input"
              label="提示文字 (Optional Prompt)"
              value={optionalPrompt}
              onChange={handlePromptChange}
              placeholder="例如：製作平滑的過渡影片(可選)"
            />

            {/* 錯誤提示區域 */}
            {error && (
              <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-400 text-red-700 rounded-md flex items-center gap-3 text-sm shadow-sm">
                <ExclamationTriangleIcon className="h-5 w-5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* 處理按鈕 */}
            <div className="mt-auto pt-6"> {/* 將按鈕推到底部 */}
              <button
                onClick={handleUploadAll}
                disabled={isButtonDisabled}
                className={`
                  w-full px-6 py-3 flex items-center justify-center gap-2
                  text-lg font-semibold text-white rounded-lg shadow-md
                  focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500
                  transition duration-200 ease-in-out
                  ${isButtonDisabled
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800'
                  }
                `}
              >
                {isProcessing ? (
                  <>
                    <ArrowPathIcon className="animate-spin h-5 w-5" />
                    正在生成影片... {/* 按下後按鈕文字 */}
                  </>
                ) : (
                  <>
                    <ArrowUpTrayIcon className="h-5 w-5" />
                    生成過渡影片 {/* 按鈕文字 */}
                  </>
                )}
              </button>
            </div>
          </section>

          {/* 右側：輸出結果區域 */}
          <section className="bg-white rounded-xl shadow-lg border border-gray-200 p-6 md:p-8 flex flex-col">
            <h2 className="text-2xl font-semibold text-gray-700 border-b pb-3 mb-4">
              輸出結果
            </h2>
            {/* 狀態提示文字 */}
            <div className="mb-4 text-center text-gray-600">
              {/* 狀態提示文字的 JSX */}
              {result.status === 'idle' && !error && (
                <p>{result.text}</p>
              )}
              {result.status === 'processing' && (
                <div className="flex items-center justify-center gap-2 text-indigo-600">
                  <ArrowPathIcon className="animate-spin h-5 w-5" />
                  <span>{result.text}</span>
                </div>
              )}
              {result.status === 'success' && (
                <div className="flex items-center justify-center gap-2 text-green-600">
                  <CheckCircleIcon className="h-5 w-5" />
                  <span>{result.text}</span>
                </div>
              )}
            </div>

            {/* 結果顯示元件容器 */}
            <div className="flex-grow flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200 shadow-inner overflow-hidden min-h-[300px] md:min-h-[400px]">

              {/* 優先判斷：如果有影片 URL 或正在處理中，則顯示 ResultDisplay */}
              {(result.outputUrl || isProcessing) ? (
                <ResultDisplay result={result} isProcessing={isProcessing} />
              ) : (
                <>
                  {/* 其次判斷：如果是錯誤狀態，顯示錯誤圖標 */}
                  {result.status === 'error' ? (
                    <div className="text-center text-red-500">
                      <ExclamationTriangleIcon className="h-16 w-16 mx-auto mb-2" />
                      <p>無法生成結果</p>
                      {/* 顯示 error state 中的詳細錯誤訊息 */}
                      {error && <p className="mt-1 text-sm">{error}</p>}
                    </div>
                  ) : (
                    /* 最後：如果是閒置狀態 (或其他未處理情況)，顯示預設圖標 */
                    <div className="text-center text-gray-400">
                      <FilmIcon className="h-16 w-16 mx-auto mb-2" />
                      <p>處理結果 (影片) 將顯示於此</p>
                    </div>
                  )}
                </>
              )}
            </div>
          </section>

        </main>
      </div>
    </div>
  );
}

export default App;