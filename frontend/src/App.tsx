import { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import ResultDisplay from "./components/ResultDisplay";

// 定義處理結果的介面
interface ProcessResult {
  image?: string; // 處理後的圖片 URL
  text: string;  // 處理後的文字結果
}

function App() {
  // 狀態管理
  const [uploading, setUploading] = useState([false, false]); // 追蹤兩個圖片的上傳狀態
  const [result, setResult] = useState<ProcessResult>({ text: '' }); // 處理結果
  const [error, setError] = useState<string | null>(null); // 錯誤訊息
  const [files, setFiles] = useState<{ [key: number]: File | null }>({ 0: null, 1: null }); // 儲存上傳的圖片檔案

  // 處理「上傳並處理」按鈕點擊事件
  const handleUploadAll = async () => {
    // 檢查是否上傳了兩張圖片
    if (!files[0] || !files[1]) {
      setError("請上傳兩張圖片"); // 設置錯誤訊息
      return; // 停止執行
    }

    setUploading([true, true]); // 設置上傳狀態為 true
    setResult({ text: '處理中...' }); // 顯示處理中的訊息
    setError(null); // 重置錯誤訊息

    const formData = new FormData();
    formData.append("image1", files[0]); // 添加第一張圖片
    formData.append("image2", files[1]); // 添加第二張圖片

    try {
      const response = await fetch("https://your-backend-endpoint/process-images", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("圖片處理失敗");
      }

      const data = await response.json(); // 解析後端返回的 JSON 數據
      setResult({
        image: data.imageUrl, // 使用後端返回的圖片 URL
        text: data.text,     // 使用後端返回的文字
      });
    } catch (error) {
      console.error("上傳錯誤", error);
      setError("圖片處理失敗，請重試"); // 設置錯誤訊息
    } finally {
      setUploading([false, false]); // 重置上傳狀態
    }
  };

  // 處理單個圖片上傳
  const handleIndividualUpload = (id: number, file: File | null) => {
    setFiles((prev) => ({ ...prev, [id]: file })); // 更新檔案狀態
    setError(null); // 重置錯誤訊息
  };

  return (
    <div className="App p-6 border-2 border-gray-300 rounded-lg m-4 shadow-md">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Stable Diffusion Model Demo</h1>
      <div className="flex gap-8">
        {/* 左側：圖片上傳區域 */}
        <div className="w-1/2 space-y-6">
          <ImageUploader id={0} title="圖片一" onUpload={handleIndividualUpload} />
          <ImageUploader id={1} title="圖片二" onUpload={handleIndividualUpload} />
          <button
            onClick={handleUploadAll}
            disabled={uploading.some((status) => status)} // 如果正在上傳，禁用按鈕
            className="mt-4 px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 w-48 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {uploading.some((status) => status) ? "處理中..." : "上傳並處理"}
          </button>
          {/* 顯示錯誤訊息 */}
          {error && <div className="text-red-500 mt-2">{error}</div>}
        </div>

        {/* 右側：處理結果顯示區域 */}
        <div className="w-1/2">
          <ResultDisplay result={result} />
        </div>
      </div>
    </div>
  );
}

export default App;