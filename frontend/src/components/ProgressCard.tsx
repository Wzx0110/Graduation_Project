// src/components/ProgressCard.tsx

import React from 'react';
import { SubstepAccordion } from './SubstepAccordion';
import type { Status, SubstepData } from '../types';

const statusDotStyles: Record<Status, string> = {
    pending: "bg-gray-600",
    running: "bg-yellow-500",
    completed: "bg-green-500",
    failed: "bg-red-500",
};

interface ProgressCardProps {
    title: string;
    status: Status;
    substeps: SubstepData[];
}

export const ProgressCard: React.FC<ProgressCardProps> = ({ title, status, substeps }) => {
    return (
        <div>
            <span
                className={`absolute flex items-center justify-center w-4 h-4 rounded-full -left-[9px] ring-8 ring-gray-800 
                   ${statusDotStyles[status]}`}
            >
                {status === 'running' && (
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
                )}
            </span>

            <h3 className="text-xl font-bold text-white mb-3">{title}</h3>

            {substeps.length > 0 && (
                <div className="space-y-2">
                    {substeps.map((substepData) => (
                        (substepData && substepData.name)
                            ? <SubstepAccordion key={substepData.name} title={substepData.name} data={substepData} />
                            : null
                    ))}
                </div>
            )}
        </div>
    );
};