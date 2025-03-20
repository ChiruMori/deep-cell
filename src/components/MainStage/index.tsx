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
import { SetStateAction, useCallback, useEffect, useState } from 'react'
import { Cells } from '../../models/cell'

extend({
    Container,
    Graphics,
    Text,
    EventSystem
})

// 初始未分化细胞个数
const CANCER_CNT = 200

// 添加实时反馈的接口
interface RealtimeFeedback {
    id: string;
    state: {
        life: number;
        hp: number;
        surround: number[];
    };
    action: {
        angle: number;
        strength: number;
    };
    immediate_reward: number;
    is_terminal: boolean;  // 是否是终止状态（死亡）
}

const MainStage = ({ setCnt, setCells, cells, setSelectedCell, setCurrentRound }: {
    cells: ICell[],
    setCells: (value: SetStateAction<ICell[]>) => void,
    setCnt: (value: SetStateAction<CellTypeCounter>) => void,
    setCurrentRound: (value: SetStateAction<number>) => void,
    setSelectedCell: (value: SetStateAction<ICell | undefined>) => void
}) => {
    const [active, setActive] = useState('')
    // 用于读取的 WebSocket
    const [ws, setWs] = useState<WebSocket | null>(null);
    // 用于发送训练反馈的 WebSocket
    const [feedbackWs, setFeedbackWs] = useState<WebSocket | null>(null);
    const [isInitialized, setIsInitialized] = useState(false);

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
            graphics.setFillStyle({ color: Cells.typeProperties(cell.type).color })
            graphics.circle(cell.x, cell.y, cell.r)
            graphics.fill()
            // 选中的细胞，绘制border
            if (cell.id === active) {
                graphics.setStrokeStyle({ color: 'white', width: 2 })
                graphics.circle(cell.x, cell.y, cell.r)
                graphics.stroke()
            }
        })
    }, [cells])

    useEffect(() => {
        if (!isInitialized) return;

        const websocket = new WebSocket('ws://localhost:8000/training/tick');

        websocket.onopen = () => {
            console.log('WebSocket 连接已建立');
            setWs(websocket);
        };

        websocket.onerror = (error) => {
            console.error('WebSocket 错误:', error);
        };

        websocket.onclose = () => {
            console.log('WebSocket 连接已关闭');
            setWs(null);
        };

        return () => {
            websocket.close();
        };
    }, [isInitialized]);

    useEffect(() => {
        if (!ws) return;

        ws.onmessage = (event) => {
            const responses = JSON.parse(event.data);
            cells.forEach(cell => {
                const res = responses.find((r: ICell) => r.id === cell.id);
                if (res) {
                    Cells.changeAcc(cell, res.strength, res.angle);
                }
            });
        };
    }, [ws, cells]);

    // 添加训练反馈的 WebSocket 连接
    useEffect(() => {
        if (!isInitialized) return;

        const websocket = new WebSocket('ws://localhost:8000/training/apoptosis');

        websocket.onopen = () => {
            console.log('训练反馈 WebSocket 连接已建立');
            setFeedbackWs(websocket);
        };

        websocket.onerror = (error) => {
            console.error('训练反馈 WebSocket 错误:', error);
        };

        websocket.onclose = () => {
            console.log('训练反馈 WebSocket 连接已关闭');
            setFeedbackWs(null);
        };

        return () => {
            websocket.close();
        };
    }, [isInitialized]);

    const tickCallback = useCallback(() => {
        if (cells.length === 0) {
            setCurrentRound(currentRound => currentRound + 1)
            setCells(_ => generateCells())
            return
        }
        setCells(currentCells => {
            if (!currentCells || currentCells.length === 0) {
                return currentCells;
            }

            const bornCells = [] as ICell[]
            const newCnt = {
                stem: 0,
                cancer: 0,
                erythrocyte: 0,
                alveolar: 0,
            } as CellTypeCounter

            // 收集实时反馈
            const realtimeFeedbacks: RealtimeFeedback[] = []

            const newCells = currentCells.filter((cell) => {
                // 记录动作前的状态
                const prevHp = cell.hp
                const prevSonCnt = cell.sonCnt || 0

                // 更新
                Cells.collide(cell, currentCells)
                Cells.move(cell)
                Cells.boundsFix(cell, { x: 0, y: 0, w: 800, h: 600 })

                // 细胞行为
                const born = Cells.actions.step(cell)
                if (born) {
                    bornCells.push(born)
                }

                const alive = cell.life > 0 && cell.hp > 0

                // 计算即时奖励
                const hpChange = cell.hp - prevHp
                const newSonCnt = cell.sonCnt || 0
                const sonChange = newSonCnt - prevSonCnt

                // 训练核心：奖励函数
                // 计算即时奖励：HP变化 + 繁殖奖励 + 存活奖励
                const immediateReward =
                    // HP变化，归一化并赋予40%权重，细胞为其他细胞提供养分，也给予奖励
                    (hpChange + cell.feed) / 100 * 0.4 +
                    // 成功繁殖（免疫细胞杀死负面细胞，杀死正常细胞时惩罚），40%权重
                    (sonChange > 0 ? 1 : 0) * 0.4 +
                    // 存活给予20%的基础奖励，死于养分不足（非自然死亡）时，给予负奖励
                    (alive ? 0.2 : (cell.hp <= 0 ? -1 : 0.05));

                // 收集实时反馈
                realtimeFeedbacks.push({
                    id: cell.id,
                    state: {
                        life: cell.life,
                        hp: cell.hp,
                        surround: cell.surround
                    },
                    action: {
                        angle: cell.xAcc ? Math.atan2(cell.yAcc, cell.xAcc) : 0,
                        strength: Math.sqrt(cell.xAcc * cell.xAcc + cell.yAcc * cell.yAcc)
                    },
                    immediate_reward: immediateReward,
                    is_terminal: !alive
                });

                newCnt[cell.type] = (newCnt[cell.type] || 0) + (alive ? 1 : 0)
                return alive
            })

            // 发送实时反馈
            if (feedbackWs && feedbackWs.readyState === WebSocket.OPEN && realtimeFeedbacks.length > 0) {
                try {
                    feedbackWs.send(JSON.stringify({
                        type: 'realtime',
                        data: realtimeFeedbacks
                    }));
                } catch (error) {
                    console.error('发送实时反馈数据时出错:', error);
                }
            }

            // 只有当所有细胞都准备好了才发送数据
            if (ws && ws.readyState === WebSocket.OPEN
                && newCells.length > 0
                && newCells[0].surround[0] !== -1
                && !ws.bufferedAmount  // 确保之前的数据已经发送完
            ) {
                try {
                    const dts = JSON.stringify(newCells)
                    ws.send(dts);
                } catch (error) {
                    console.error('发送 WebSocket 数据时出错:', error);
                }
            }

            // 合并细胞数组
            newCells.push(...bornCells)
            setCnt(newCnt)
            return newCells;
        })
    }, [ws, feedbackWs, setCells, setCnt])

    const generateCells = function (): ICell[] {
        const newCells = [] as ICell[]
        for (let i = 0; i < CANCER_CNT; i++) {
            newCells.push(Cells.create({
                x: Math.random() * 800,
                y: Math.random() * 600,
                r: Cells.typeProperties('cancer').radius
            }, 'cancer'))
        }
        return newCells
    }

    const initCells = useCallback(() => {
        setCells(_ => generateCells());
        setIsInitialized(true);
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
            </pixiContainer >
        </>
    )
}

export default MainStage