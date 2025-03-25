import { JSX } from "react";

interface BarProps {
    value: number;
    maxValue: number;
    bgColor: string;
    color: string;
    deadText: string;
}

export default function (props: BarProps): JSX.Element {
    return <>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
                width: '100px',
                height: '10px',
                backgroundColor: props.bgColor,
                borderRadius: '5px',
                overflow: 'hidden'
            }}>
                <div
                    style={{
                        width: `${(props.value / props.maxValue * 100).toFixed(2)}%`,
                        height: '100%',
                        backgroundColor: props.color,
                        transition: 'width 0.3s ease'
                    }}
                />
            </div>
            <span>{props.value > 0 ? props.value.toFixed(2) : props.deadText}</span>
        </div>
    </>
}