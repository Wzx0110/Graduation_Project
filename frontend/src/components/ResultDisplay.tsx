import React from 'react';

interface ResultDisplayProps {
    result: {
        outputUrl?: string;
        text: string;
        status: 'idle' | 'processing' | 'success' | 'error';
    };
    isProcessing: boolean;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, isProcessing }) => {
    return (
        // 保持 w-full 使其填滿容器寬度
        <div className="w-full h-full flex items-center justify-center">
            {/* 影片顯示區域 */}
            <div className="w-full h-full flex items-center justify-center">
                {result.outputUrl ? (
                    <video
                        src={result.outputUrl}
                        controls // 加入播放控制條
                        loop // 影片循環播放
                        autoPlay // 自動播放
                        playsInline // 在移動裝置上行內播放
                        className="max-w-full max-h-full object-contain" // 保持影片在容器內且維持比例
                    >
                        您的瀏覽器不支援 Video 標籤。
                    </video>
                ) : (
                    <div className="text-gray-400 italic px-4 text-center">
                        {isProcessing ? '影片生成中...' : (result.status === 'error' ? '' : '等待生成影片...')}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ResultDisplay;