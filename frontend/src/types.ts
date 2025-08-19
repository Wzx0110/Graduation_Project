// src/types.ts

// 這裡存放我們所有共用的類型定義

export type Status = 'pending' | 'running' | 'completed' | 'failed';
export type GenerationStrategy = "basic" | "controlnet" | "lora" | "controlnet_lora";

export interface SubstepData {
  status: Status;
  name: string; // 細項的名稱
  text?: string;
  full_data?: string;
  previews?: string[];
  preview_match?: string;
}