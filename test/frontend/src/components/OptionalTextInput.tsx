import React from 'react';

interface OptionalTextInputProps {
    id: string; // 獨特的 ID，用於標識輸入框
    label: string; // 標籤文字
    value: string; // 當前輸入的值
    onChange: (value: string) => void; // 當值改變時的回調函數 
    placeholder?: string; // 佔位符文字
    className?: string; // 額外的 CSS 類名
}

const OptionalTextInput: React.FC<OptionalTextInputProps> = ({
    id,
    label,
    value,
    onChange,
    placeholder = '',
    className = '',
}) => {
    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        onChange(event.target.value);
    };

    return (
        <div className={`space-y-2 ${className}`}>
            <label
                htmlFor={id}
                className="block text-sm font-medium text-gray-700"
            >
                {label}
            </label>
            <input
                type="text"
                id={id}
                name={id}
                value={value}
                onChange={handleChange}
                placeholder={placeholder}
                className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            />
        </div>
    );
};

export default OptionalTextInput;