import { useState } from "react";
import ImageUploader from "./ImageUploader";

function ImageUploadForm() {
  // 狀態管理
  const [images, setImages] = useState<{ [key: number]: File | null }>({}); // 上傳的圖片
  const [error, setError] = useState<string | null>(null); // 錯誤訊息

  // 處理圖片上傳
  const handleChildUpload = (id: number, file: File | null) => {
    setImages((prev) => ({ ...prev, [id]: file }));
    setError(null); // 重置錯誤訊息
  };

  // 處理表單提交
  const handleSubmit = async () => {
    if (!images[1] || !images[2]) {
      setError("請上傳兩張圖片");
      return;
    }

    const formData = new FormData();
    formData.append("image1", images[1] as File);
    formData.append("image2", images[2] as File);

    try {
      const response = await fetch("https://your-backend-endpoint/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("上傳失敗");
      }

      console.log("上傳成功", response);
      setError(null); // 重置錯誤訊息
    } catch (error) {
      console.error("上傳錯誤", error);
      setError("上傳失敗，請重試");
    }
  };

  return (
    <div>
      <ImageUploader id={1} onUpload={handleChildUpload} title="上傳圖片 1" />
      <ImageUploader id={2} onUpload={handleChildUpload} title="上傳圖片 2" />
      <button onClick={handleSubmit}>送出圖片</button>
      {/* 顯示錯誤訊息 */}
      {error && <div className="text-red-500">{error}</div>}
    </div>
  );
}

export default ImageUploadForm;