import { JSX } from "react";

interface ArrowProps {
    direction?: number;
    color?: string;
}

export default function Arrow(props: ArrowProps): JSX.Element {

    return (
        <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            height: '40px',
            color: props.color?? 'white'
        }}>
            <svg
                width="40"
                height="40"
                viewBox="0 0 24 24"
                style={{
                    transform: `rotate(${(props.direction ?? 0) * 180 / Math.PI}deg)`,
                    transition: 'transform 0.1s cubic-bezier(0.4, 0, 0.2, 1)'
                }}
            >
                <path
                    d="M12 2L20 12H15V22H9V12H4L12 2Z"
                    fill="currentColor"
                    stroke="currentColor"
                    strokeWidth="0.5"
                />
            </svg>
            <span>
                {((props.direction ?? -1) * 180 / Math.PI).toFixed(1)}°
            </span>
        </div>
    )
}