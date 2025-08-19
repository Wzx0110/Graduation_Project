import { useState, ChangeEvent, MouseEvent } from "react";

// 定義 ImageUploader 的 Props 介面
interface ImageUploaderProps {
  id: number; // 圖片 ID
  onUpload: (id: number, file: File | null) => void; // 上傳回調函數
  title: string; // 標題
}

function ImageUploader({ id, onUpload, title }: ImageUploaderProps) {
  // 狀態管理
  const [image, setImage] = useState<File | null>(null); // 上傳的圖片檔案
  const [preview, setPreview] = useState<string | null>(null); // 圖片預覽 URL
  const [error, setError] = useState<string | null>(null); // 錯誤訊息

  // 處理檔案選擇事件
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // 驗證檔案類型
      if (!file.type.startsWith("image/")) {
        setError("請選擇有效的圖片檔案");
        return;
      }

      setImage(file);
      setPreview(URL.createObjectURL(file)); // 生成圖片預覽 URL
      onUpload(id, file); // 通知父組件
      setError(null); // 重置錯誤訊息
    }
  };

  // 處理移除圖片事件
  const handleRemoveImage = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setImage(null);
    setPreview(null);
    onUpload(id, null); // 通知父組件
    setError(null); // 重置錯誤訊息
  };

  return (
    <div className="flex flex-col items-center">
      {/* 圖片上傳區域 */}
      <div
        className="w-full h-64 border-2 border-dashed border-gray-400 flex items-center justify-center cursor-pointer rounded-lg relative bg-gray-50"
        onClick={() => document.getElementById(`fileInput-${id}`)?.click()}
      >
        <span className="absolute top-2 left-2 text-gray-700 font-medium z-10 bg-white px-2 py-1 rounded border border-gray-300">
          {title}
        </span>
        {image ? (
          <div className="relative w-full h-full">
            <img
              src={preview || ""}
              alt="Preview"
              className="w-full h-full object-cover rounded-lg"
            />
            <button
              onClick={handleRemoveImage}
              className="absolute top-0 right-0 mt-2 mr-2 w-6 h-6 rounded-full bg-gray-600 text-white flex items-center justify-center cursor-pointer"
            >
              <span className="text-sm">×</span>
            </button>
          </div>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-gray-500">點擊上傳照片</span>
          </div>
        )}
      </div>

      {/* 檔案選擇輸入框 */}
      <input
        id={`fileInput-${id}`}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* 顯示錯誤訊息 */}
      {error && <div className="text-red-500 mt-2">{error}</div>}
    </div>
  );
}

export default ImageUploader;