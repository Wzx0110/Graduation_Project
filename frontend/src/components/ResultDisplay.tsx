import React from 'react';

// 定義 ResultDisplay 的 Props 介面
interface ResultDisplayProps {
    result: {
        image?: string; // 處理後的圖片 URL
        text: string;  // 處理後的文字結果
    };
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result }) => {
    return (
        <div className="w-full space-y-4">
            {/* 圖片顯示區域 */}
            <div className="w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50">
                {result.image ? (
                    <img
                        src={result.image}
                        alt="處理結果"
                        className="max-w-full max-h-full object-contain"
                    />
                ) : (
                    <div className="text-gray-500">等待處理結果...</div>
                )}
            </div>

            {/* 文字顯示區域 */}
            <div className="w-full min-h-[100px] border border-gray-300 rounded-lg p-4 bg-white">
                {result.text || '等待處理結果...'}
            </div>
        </div>
    );
};

export default ResultDisplay;