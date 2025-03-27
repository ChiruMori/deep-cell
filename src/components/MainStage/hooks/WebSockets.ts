import { useEffect, useState } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket'

export const useWebSockets = () => {

    const [tickPending, setTickPending] = useState(false);
    const [lastResTime, setLastResTime] = useState(Date.now());
    const [tickTimeout, setTickTimeout] = useState<number | null>(null);

    const {
        sendJsonMessage: sendTickMsg,
        lastJsonMessage: lastTickMsg,
        readyState: tickReady
    } = useWebSocket('ws://localhost:8000/training/tick', {
        shouldReconnect: () => true,
        reconnectInterval: 3000,
    });
    const {
        sendJsonMessage: sendFbMessage,
        readyState: fbReady
    } = useWebSocket('ws://localhost:8000/training/apoptosis', {
        shouldReconnect: () => true,
        reconnectInterval: 3000,
    });


    useEffect(() => {
        // 更新请求状态
        setTickPending(false);
        setLastResTime(Date.now());

        // 如果有超时检测，清除它
        if (tickTimeout !== null) {
            window.clearTimeout(tickTimeout);
            setTickTimeout(null);
        }
    }, [lastTickMsg]);

    return {
        sendTickMsg: <T = unknown>(jsonMessage: T) => {
            if (tickPending) {
                throw new Error('Pending tick request');
            }
            setTickPending(true);
            sendTickMsg(jsonMessage);
            setLastResTime(Date.now());
            // 设置超时检测
            setTickTimeout(window.setTimeout(() => {
                if (Date.now() - lastResTime > 3000) {
                    setTickPending(false);
                    console.debug('Tick timeout');
                }
            }, 3000));
        },
        lastTickMsg: lastTickMsg as any[] | null,
        sendFbMessage,
        allReady: tickReady === ReadyState.OPEN && fbReady === ReadyState.OPEN,
        tickPending: tickPending,
    }
}