import {
    extend,
    useTick
} from '@pixi/react'
import {
    Container,
    EventSystem,
    FederatedPointerEvent,
    Graphics,
    Text,
} from 'pixi.js'
import { SetStateAction, useCallback, useEffect, useState, useRef } from 'react'
import { Cells, typeAsNumber } from '../../models/cell'

extend({
    Container,
    Graphics,
    Text,
    EventSystem
})

const LOG_DATA_SAMPLE = false;
const LOG_SOCKET_LIFECYCLE = false;

// 初始未分化细胞个数
const CANCER_CNT = 200

// 添加实时反馈的接口
interface RealtimeFeedback {
    id: string;
    state: {
        life: number;
        hp: number;
        surround: number[];
        c_type: number;
    };
    action: {
        angle: number;
        strength: number;
        kw: number;
    };
    immediate_reward: number;
    is_terminal: boolean;
}

const MainStage = ({ setCnt, setCells, cells, setSelectedCell, setCurrentRound, setPause, userPaused }: {
    cells: ICell[],
    setCells: (value: SetStateAction<ICell[]>) => void,
    setCnt: (value: SetStateAction<CellTypeCounter>) => void,
    setCurrentRound: (value: SetStateAction<number>) => void,
    setSelectedCell: (value: SetStateAction<ICell | undefined>) => void,
    setPause: (value: SetStateAction<boolean>) => void,
    userPaused: boolean,
}) => {
    const [active, setActive] = useState('')
    // 使用ref存储WebSocket实例
    const tickWsRef = useRef<WebSocket | null>(null);
    const feedbackWsRef = useRef<WebSocket | null>(null);
    const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
    const wsReconnectTimerRef = useRef<number | null>(null);
    const initializedRef = useRef(false);

    // 添加请求状态跟踪
    const pendingTickRequestRef = useRef(false);
    const lastTickResponseTimeRef = useRef(Date.now());
    const tickTimeoutRef = useRef<number | null>(null);
    const cellsToUpdateRef = useRef<ICell[]>([]);

    const drawCallback = useCallback((graphics: Graphics) => {
        // 绘制网格
        graphics.clear()
        graphics.setFillStyle({ color: 'black' })
        const w = 800
        const h = 600
        const gridSize = 50
        graphics.rect(0, 0, w, h)
        graphics.fill()
        graphics.setStrokeStyle({ color: '#333333', width: 1 })
        for (let i = 0; i < w; i += gridSize) {
            graphics.moveTo(i, 0)
            graphics.lineTo(i, h)
        }
        for (let i = 0; i < h; i += gridSize) {
            graphics.moveTo(0, i)
            graphics.lineTo(w, i)
        }
        graphics.stroke()
        // 绘制细胞
        cells.forEach((cell) => {
            if (cell.id === active) {
                // 选中的细胞，绘制视野区域
                graphics.setStrokeStyle({ color: 'white', width: 1 })
                graphics.circle(cell.x, cell.y, Cells.VISION_DISTANCE)
                graphics.fillStyle = Cells.typeProperties(cell.type).color.substring(0, 7) + '33';
                graphics.fill()
                // 为选中的细胞绘制六个方向的射线（圆心到 VISION_DISTANCE）
                for (let i = 0; i < 6; i++) {
                    const angle = (Math.PI / 3) * i;
                    const x = cell.x + Cells.VISION_DISTANCE * Math.cos(angle);
                    const y = cell.y + Cells.VISION_DISTANCE * Math.sin(angle);
                    graphics.moveTo(cell.x, cell.y);
                    graphics.lineTo(x, y);
                }
                graphics.stroke()
            }
            // 细胞本身（一个大圆点）
            graphics.setFillStyle({ color: Cells.typeProperties(cell.type).color })
            graphics.circle(cell.x, cell.y, cell.r)
            graphics.fill()
        })
    }, [cells])

    useEffect(() => {
        if (!initializedRef.current) {
            initCells();
            initializedRef.current = true;
        }
    }, []);

    // 创建和管理WebSocket连接
    const createWebSocketConnections = useCallback(() => {
        if (wsStatus === 'connecting') return;

        // 关闭现有连接
        if (tickWsRef.current && tickWsRef.current.readyState !== WebSocket.CLOSED) {
            tickWsRef.current.close();
        }

        if (feedbackWsRef.current && feedbackWsRef.current.readyState !== WebSocket.CLOSED) {
            feedbackWsRef.current.close();
        }

        setWsStatus('connecting');
        setPause(true);

        // 创建新的WebSocket连接
        LOG_SOCKET_LIFECYCLE && console.log("创建新的WebSocket连接...");

        // 创建tick WebSocket
        const tickWs = new WebSocket('ws://localhost:8000/training/tick');
        tickWsRef.current = tickWs;

        // 创建feedback WebSocket
        const feedbackWs = new WebSocket('ws://localhost:8000/training/apoptosis');
        feedbackWsRef.current = feedbackWs;

        // 设置事件处理器
        tickWs.onopen = () => {
            LOG_SOCKET_LIFECYCLE && console.log("Tick WebSocket连接成功");
            setWsStatus('connected');
            pendingTickRequestRef.current = false;
            setPause(feedbackWsRef.current?.readyState !== WebSocket.OPEN);
        };

        tickWs.onmessage = (event) => {
            try {
                const responses = JSON.parse(event.data);
                LOG_DATA_SAMPLE && console.log('收到tick响应，Sample: ', responses[0])

                // 对保存的细胞引用应用更新
                if (cellsToUpdateRef.current.length > 0) {
                    for (const cell of cellsToUpdateRef.current) {
                        const res = responses.find((r: any) => r.id === cell.id);
                        if (res) {
                            const ml = {
                                direction: res.angle,
                                strength: res.strength,
                                kw: res.kw
                            }
                            cell.ml = ml;
                        }
                    }

                    // 清空临时细胞引用
                    cellsToUpdateRef.current = [];
                }

                // 更新请求状态
                pendingTickRequestRef.current = false;
                lastTickResponseTimeRef.current = Date.now();

                // 如果有超时检测，清除它
                if (tickTimeoutRef.current !== null) {
                    window.clearTimeout(tickTimeoutRef.current);
                    tickTimeoutRef.current = null;
                }
            } catch (error) {
                console.error("处理WebSocket响应时出错:", error);
                pendingTickRequestRef.current = false;
            }
        };

        tickWs.onerror = (error) => {
            console.error("Tick WebSocket错误:", error);
            setWsStatus('disconnected');
            scheduleReconnect();
        };

        tickWs.onclose = () => {
            LOG_SOCKET_LIFECYCLE && console.log("Tick WebSocket连接关闭");
            setWsStatus('disconnected');
            scheduleReconnect();
        };

        feedbackWs.onopen = () => {
            LOG_SOCKET_LIFECYCLE && console.log("Feedback WebSocket连接成功");
            setPause(tickWsRef.current?.readyState !== WebSocket.OPEN);
        };

        feedbackWs.onerror = (error) => {
            console.error("Feedback WebSocket错误:", error);
        };

        feedbackWs.onclose = () => {
            LOG_SOCKET_LIFECYCLE && console.log("Feedback WebSocket连接关闭");
        };
    }, [wsStatus]);

    // 计划重新连接
    const scheduleReconnect = useCallback(() => {
        if (wsReconnectTimerRef.current) {
            window.clearTimeout(wsReconnectTimerRef.current);
        }

        wsReconnectTimerRef.current = window.setTimeout(() => {
            LOG_SOCKET_LIFECYCLE && console.log("尝试重新连接WebSocket...");
            createWebSocketConnections();
            wsReconnectTimerRef.current = null;
        }, 2000); // 2秒后重试
    }, [createWebSocketConnections]);

    // 组件挂载和卸载
    useEffect(() => {
        // 初始化细胞
        if (!initializedRef.current) {
            initCells();
            initializedRef.current = true;
        }

        // 创建WebSocket连接
        createWebSocketConnections();

        // 清理函数
        return () => {
            if (tickWsRef.current) {
                tickWsRef.current.close();
            }

            if (feedbackWsRef.current) {
                feedbackWsRef.current.close();
            }

            if (wsReconnectTimerRef.current) {
                window.clearTimeout(wsReconnectTimerRef.current);
            }
        };
    }, []);

    const generateCells = useCallback(() => {
        const newCells = [] as ICell[]
        for (let i = 0; i < CANCER_CNT; i++) {
            newCells.push(Cells.create({
                x: Math.random() * 800,
                y: Math.random() * 600,
                r: Cells.typeProperties('cancer').radius
            }, 'cancer'));
        }
        return newCells;
    }, [])

    // 修改tickCallback使用同步等待模式
    const tickCallback = useCallback(() => {
        if (userPaused) {
            return;
        }
        // 如果前一个请求还在等待响应，跳过这一帧
        if (pendingTickRequestRef.current) {
            // 检查是否超时（超过3秒没有响应）
            const now = Date.now();
            if (now - lastTickResponseTimeRef.current > 3000) {
                console.warn("WebSocket请求超时，重置状态");
                pendingTickRequestRef.current = false;

                // 可选：重新连接
                if (wsStatus === 'connected') {
                    LOG_SOCKET_LIFECYCLE && console.log("尝试重新建立连接...");
                    createWebSocketConnections();
                }
            } else {
                // 还在等待，不处理这一帧
                return;
            }
        }

        setCells(currentCells => {
            // 检查是否需要重置
            if (currentCells.length === 0 && initializedRef.current) {
                // 新一轮开始
                setCurrentRound(r => r + 1);

                // 检查WebSocket连接
                if (wsStatus !== 'connected') {
                    LOG_SOCKET_LIFECYCLE && console.log("WebSocket未连接，尝试重新连接...");
                    createWebSocketConnections();
                    // 返回空数组，等下一帧处理
                    return currentCells;
                }

                return generateCells();
            }

            // 如果没有细胞或WebSocket未连接，不进行处理
            if (!currentCells || currentCells.length === 0) {
                return currentCells;
            }

            if (wsStatus !== 'connected') {
                // WebSocket未连接，不处理，但保持显示
                return currentCells;
            }

            const bornCells = [] as ICell[];
            const newCnt = {
                stem: 0,
                cancer: 0,
                erythrocyte: 0,
                alveolar: 0,
            } as CellTypeCounter;

            // 收集实时反馈
            const realtimeFeedbacks: RealtimeFeedback[] = [];

            const newCells = currentCells.filter((cell) => {
                if (!cell.ml) {
                    return true;
                }
                if (cell.hp < 0 || cell.life < 0) {
                    throw new Error('细胞状态异常，状态已过期');
                }
                // 记录动作前的状态
                const prevHp = cell.hp;
                const prevSonCnt = cell.sonCnt || 0;

                // 更新
                Cells.collide(cell, currentCells);
                Cells.move(cell);
                Cells.boundsFix(cell, { x: 0, y: 0, w: 800, h: 600 });

                // 细胞行为
                const born = Cells.actions.step(cell);
                if (born) {
                    bornCells.push(born);
                }

                const alive = cell.life > 0 && cell.hp > 0;

                // 计算即时奖励
                const hpChange = cell.hp - prevHp;
                const newSonCnt = cell.sonCnt || 0;
                // 繁殖奖励，需要抵消HP消耗
                const sonChange = (newSonCnt - prevSonCnt) * Cells.SPLIT_NEED * 0.1;

                // 训练核心：奖励函数
                // 计算即时奖励：HP变化 + 繁殖奖励 + 存活奖励
                const immediateReward =
                    // HP变化，细胞为其他细胞提供养分，会获得更高奖励
                    (hpChange * 0.1 + cell.feed * 0.2) * 0.4 +
                    // 成功繁殖（免疫细胞杀死负面细胞，杀死正常细胞时惩罚），40%权重
                    (sonChange > 0 ? 1 : 0) * 0.4 +
                    // 存活的基础奖励，死于养分不足（非自然死亡）时，给予负奖励
                    (alive ? 0.1 : (cell.hp <= 0 ? -1 : 0));
                const onlyAliveReward = (hpChange == 0 || hpChange == -1) && sonChange == 0 && alive;
                if (cell.mlForView) {
                    cell.mlForView.reward = immediateReward;
                }
                const exp = {
                    id: cell.id,
                    state: {
                        life: cell.life,
                        hp: cell.hp,
                        surround: cell.surround,
                        c_type: typeAsNumber(cell.type),
                        speed: [cell.xSpeed, cell.ySpeed],
                    },
                    action: {
                        angle: cell.ml?.direction || 0,
                        strength: cell.ml?.strength || 0,
                        kw: cell.ml?.kw || 0
                    },
                    immediate_reward: immediateReward,
                    is_terminal: !alive
                }
                // 收集实时反馈，仅有存活奖励时，降低收集频率
                if (!onlyAliveReward && Math.random() < 0.05) {
                    realtimeFeedbacks.push(exp);
                }

                // 细胞挂了，补充一次反馈（用于训练 cancer 分化方向）
                if (!alive) {
                    const finalReward =
                        // 分化的子代细胞存活时间越长，奖励越高
                        cell.lifeTime / Cells.typeProperties(cell.type).life * 0.5 +
                        // 繁殖越多，奖励越高
                        cell.sonCnt * 0.2 +
                        // 细胞剩余养分越高，奖励越高，如果因为养分不足死亡，则给予负奖励
                        (cell.hp > 0 ? 0.3 : -0.2)
                    const finalExp = {
                        id: cell.id,
                        state: {
                            life: cell.lifeTime,
                            hp: cell.hp,
                            surround: cell.surround,
                            c_type: typeAsNumber('cancer')
                        },
                        action: {
                            angle: cell.ml?.direction || 0,
                            strength: cell.ml?.strength || 0,
                            kw: cell.ml?.kw || 0
                        },
                        immediate_reward: finalReward,
                        is_terminal: true
                    }
                    realtimeFeedbacks.push(finalExp);
                }

                newCnt[cell.type] = (newCnt[cell.type] || 0) + (alive ? 1 : 0);
                return alive;
            });

            // 发送反馈数据 - 异步处理，不影响主循环
            if (feedbackWsRef.current &&
                feedbackWsRef.current.readyState === WebSocket.OPEN &&
                realtimeFeedbacks.length > 0) {

                try {
                    feedbackWsRef.current.send(JSON.stringify(realtimeFeedbacks));
                    LOG_DATA_SAMPLE && console.log('发送反馈数据，Sample: ', realtimeFeedbacks[0])
                } catch (error) {
                    console.error('发送反馈数据时出错:', error);
                }
            }

            // 发送数据到服务器 - 同步等待模式
            if (tickWsRef.current &&
                tickWsRef.current.readyState === WebSocket.OPEN &&
                newCells.length > 0 &&
                !tickWsRef.current.bufferedAmount) {

                try {
                    // 标记为等待响应状态
                    pendingTickRequestRef.current = true;
                    lastTickResponseTimeRef.current = Date.now();

                    // 保存当前细胞引用以便在响应中更新
                    cellsToUpdateRef.current = [...newCells];

                    const datas = newCells.map(cell => ({
                        c_type: typeAsNumber(cell.type),
                        life: cell.life,
                        hp: cell.hp,
                        surround: cell.surround || [0, 0, 0, 0, 0, 0],
                        speed: [cell.xSpeed, cell.ySpeed],
                        id: cell.id
                    }));

                    LOG_DATA_SAMPLE && console.log('发送tick请求，Sample: ', datas[0])

                    // 设置超时检测
                    if (tickTimeoutRef.current !== null) {
                        window.clearTimeout(tickTimeoutRef.current);
                    }

                    tickTimeoutRef.current = window.setTimeout(() => {
                        LOG_SOCKET_LIFECYCLE && console.warn("WebSocket请求超时，重置状态");
                        pendingTickRequestRef.current = false;
                        tickTimeoutRef.current = null;
                    }, 3000); // 3秒超时

                    // 发送数据
                    tickWsRef.current.send(JSON.stringify(datas));

                } catch (error) {
                    LOG_SOCKET_LIFECYCLE && console.error('发送WebSocket数据时出错:', error);
                    pendingTickRequestRef.current = false; // 出错时重置状态
                }
            } else if (tickWsRef.current?.bufferedAmount) {
                LOG_SOCKET_LIFECYCLE && console.log('WebSocket缓冲区已满，等待下一次发送');
                pendingTickRequestRef.current = true;
            }

            // 更新计数器和细胞数组
            setCnt(newCnt);

            // 合并出生的细胞
            return [...newCells, ...bornCells];
        });
    }, [wsStatus, createWebSocketConnections, userPaused]);

    const initCells = useCallback(() => {
        setCells(prevCells => {
            // 如果已经有细胞，就不再创建新细胞
            if (prevCells && prevCells.length > 0) {
                return prevCells;
            }
            return generateCells();
        });
        initializedRef.current = true;
    }, [setCells])

    useEffect(() => {
        initCells()
    }, [])

    useTick(tickCallback)

    return (
        <>
            <pixiContainer
                interactive={true}
                eventMode="static"
                width={800}
                height={600}
                hitArea={{
                    contains: (x: number, y: number) => {
                        return x >= 0 && x <= 800 && y >= 0 && y <= 600;
                    }
                }}
                onPointerDown={(e: FederatedPointerEvent) => {
                    const currentTarget = e.currentTarget as Container;
                    const pos = e.getLocalPosition(currentTarget);
                    const hitCell = cells.find(cell => Cells.checkHit(cell, pos.x, pos.y));
                    setSelectedCell(hitCell);
                    setActive(hitCell ? hitCell.id : '');
                }}
                onWheel={(_: WheelEvent) => {
                    if (cells.length === 0) return;
                    // 随机切换到一个细胞
                    const randomIndex = Math.floor(Math.random() * cells.length);
                    const randomCell = cells[randomIndex];
                    setSelectedCell(randomCell);
                    setActive(randomCell.id);
                }}
            >
                <pixiGraphics draw={drawCallback} />
                {wsStatus !== 'connected' && (
                    <pixiText text={`服务器连接${wsStatus === 'connecting' ? '中...' : '断开'}`}
                        x={800 / 2}
                        y={600 / 2}
                        anchor={0.5}
                        style={{
                            fontSize: 24,
                            fill: '#ffffff'
                        }}
                    />
                )}
                {pendingTickRequestRef.current && (
                    <pixiText text="Training..."
                        x={800 / 2}
                        y={550}
                        anchor={0.5}
                        style={{
                            fontSize: 14,
                            fill: '#ffff00'
                        }}
                    />
                )}
                {userPaused && (
                    <pixiText text="已暂停"
                        x={800 / 2}
                        y={600 / 2}
                        anchor={0.5}
                        style={{
                            fontSize: 24,
                            fill: '#ffffff'
                        }}
                    />
                )}
            </pixiContainer>
        </>
    )
}

export default MainStage