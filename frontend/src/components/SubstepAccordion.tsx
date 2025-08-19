// src/components/SubstepAccordion.tsx

import React, { useState } from 'react';
import { ChevronUpIcon, ChevronDownIcon, DownloadIcon } from './Icons';
import type { SubstepData } from '../types'; // 從中央類型定義文件導入

export const SubstepAccordion: React.FC<{ title: string; data: SubstepData }> = ({ title, data }) => {
    const [isOpen, setIsOpen] = useState(false);

    // 下載函數
    const handleDownload = (e: React.MouseEvent) => {
        e.stopPropagation(); // 防止觸發 accordion 的開合

        // 優先下載圖片
        const imagesToDownload = data.previews || (data.preview_match ? [data.preview_match] : []);
        if (imagesToDownload.length > 0) {
            imagesToDownload.forEach((base64Image: string, index: number) => {
                const link = document.createElement('a');
                link.href = base64Image;
                const safeTitle = title.replace(/[\s/\\?%*:|"<>]/g, '_');
                const fileName = `${safeTitle}_${index + 1}.png`;
                link.download = fileName;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            });
            return;
        }

        // 如果沒有圖片，則下載文本數據，優先使用 full_data
        const textToDownload = data.full_data || data.text;

        if (textToDownload) {
            // 檢查是否是 JSON
            if (textToDownload.trim().startsWith('{') || textToDownload.trim().startsWith('[')) {
                try {
                    const jsonObj = JSON.parse(textToDownload);
                    const jsonString = JSON.stringify(jsonObj, null, 2);
                    const blob = new Blob([jsonString], { type: 'application/json;charset=utf-8' });
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    const fileName = `${title.replace(/[\s/\\?%*:|"<>]/g, '_')}.json`;
                    link.download = fileName;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(url);
                    return;
                } catch (error) {
                    console.warn("數據不是有效的 JSON，將作為純文本下載。", error);
                }
            }

            // 如果不是 JSON 或解析失敗，則作為純文本 .txt 下載
            const blob = new Blob([textToDownload], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            const fileName = `${title.replace(/[\s/\\?%*:|"<>]/g, '_')}.txt`;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }
    };

    // 渲染內容的輔助函數
    const renderContent = () => {
        // 智能處理 previews 陣列
        if (data.previews && Array.isArray(data.previews)) {
            if (data.previews.length === 1) {
                // 如果只有一張圖，讓它佔滿寬度
                return <img src={data.previews[0]} alt={`${title} preview`} className="mt-2 rounded-md border border-gray-600 w-full" />;
            }
            if (data.previews.length === 2) {
                // 如果有兩張圖，左右對比
                return (
                    <div className="grid grid-cols-2 gap-2 mt-2">
                        <img src={data.previews[0]} alt={`${title} preview 1`} className="rounded-md border border-gray-600" />
                        <img src={data.previews[1]} alt={`${title} preview 2`} className="rounded-md border border-gray-600" />
                    </div>
                );
            }
        }
        // 處理單張大圖 (用於匹配線等)
        if (data.preview_match) {
            return <img src={data.preview_match} alt={`${title} match preview`} className="mt-2 rounded-md border border-gray-600 w-full" />;
        }
        return null;
    };

    const isDownloadable = data.previews || data.preview_match || data.text || data.full_data;

    // --- JSX 渲染 ---
    return (
        <div className="rounded-lg" style={{ backgroundColor: 'var(--secondary)', border: '1px solid var(--border)' }}>
            <button
                className="w-full flex items-center justify-between p-3 text-left rounded-lg transition-colors hover:bg-gray-700/50"
                onClick={() => setIsOpen(!isOpen)}
            >
                <span className="font-medium">{title}</span>
                {isOpen ? <ChevronUpIcon className="w-5 h-5 text-gray-400" /> : <ChevronDownIcon className="w-5 h-5 text-gray-400" />}
            </button>

            {isOpen && (
                <div className="p-4 border-t" style={{ borderColor: 'var(--border)' }}>
                    {data.text && <pre className="text-xs font-mono p-3 rounded-md whitespace-pre-wrap" style={{ backgroundColor: 'var(--background)', color: 'var(--muted-foreground)' }}>{data.text}</pre>}

                    <div className="flex justify-center">
                        {renderContent()}
                    </div>

                    <div className="mt-4 flex justify-end">
                        {isDownloadable && (
                            <button
                                onClick={handleDownload}
                                className="p-1 text-gray-400 hover:text-white"
                                title={`下載 ${title} 結果`}
                            >
                                <DownloadIcon className="w-5 h-5" />
                            </button>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};