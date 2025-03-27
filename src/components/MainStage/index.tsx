import { extend, useTick } from '@pixi/react'
import {
    Container,
    EventSystem,
    FederatedPointerEvent,
    Graphics,
    Text,
} from 'pixi.js'
import { SetStateAction, useCallback, useEffect, useState, useRef } from 'react'
import { Cells, typeAsNumber } from '../../common/cell'
import CellGraph from './CellGraph'
import { useCellLifecycle } from './hooks/CellLifecycle'
import { useWebSockets } from './hooks/WebSockets'
import C from '../../common/constants'

extend({
    Container,
    Graphics,
    Text,
    EventSystem
})

const LOG_DATA_SAMPLE = false;

const MainStage = ({ setCnt, setCells, cells, setSelectedCell, setCurrentRound, userPaused }: {
    cells: ICell[],
    setCells: (value: SetStateAction<ICell[]>) => void,
    setCnt: (value: SetStateAction<CellTypeCounter>) => void,
    setCurrentRound: (value: SetStateAction<number>) => void,
    setSelectedCell: (value: SetStateAction<ICell | undefined>) => void,
    userPaused: boolean,
}) => {
    const [active, setActive] = useState('')
    // 添加请求状态跟踪
    const ws = useWebSockets();

    const cellsToUpdateRef = useRef<ICell[]>([]);

    // 细胞生命周期
    const { updateCells } = useCellLifecycle();

    useEffect(() => {
        if (!ws.lastTickMsg) {
            return
        }
        try {
            LOG_DATA_SAMPLE && console.log('收到tick响应，Sample: ', ws.lastTickMsg[0])

            // 对保存的细胞引用应用更新
            if (cellsToUpdateRef.current.length > 0) {
                for (const cell of cellsToUpdateRef.current) {
                    const res = ws.lastTickMsg.find((r: any) => r.id === cell.id);
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
        } catch (error) {
            console.error("处理WebSocket响应时出错:", error);
        }
    }, [ws.lastTickMsg]);

    const tickCallback = useCallback(() => {
        if (userPaused || !ws.allReady || ws.tickPending) {
            return;
        }

        setCells(prevCells => {
            const tickRes = updateCells(prevCells);
            setCnt(tickRes.cnt);
            setCurrentRound(rnd => tickRes.newRound ? rnd + 1 : rnd);
            // 反馈数据
            if (tickRes.feedbacksForTraining) {
                ws.sendFbMessage(tickRes.feedbacksForTraining);
                LOG_DATA_SAMPLE && console.log('发送反馈数据，Sample: ', tickRes.feedbacksForTraining[0])
            }
            if (tickRes.newCells.length <= 0) {
                return tickRes.newCells;
            }
            // 保存当前细胞引用以便在响应中更新
            cellsToUpdateRef.current = [...tickRes.newCells];

            // 实时数据（发送新一帧的细胞数据）
            const datas = tickRes.newCells.map(cell => ({
                c_type: typeAsNumber(cell.type),
                life: cell.life,
                hp: cell.hp,
                surround: cell.surround || [0, 0, 0, 0, 0, 0],
                speed: [cell.xSpeed, cell.ySpeed],
                id: cell.id
            }));

            // 发送数据
            LOG_DATA_SAMPLE && console.log('发送tick请求，Sample: ', datas[0])
            ws.sendTickMsg(datas);

            return tickRes.newCells;
        });
    }, [userPaused, ws.allReady, cells, ws.tickPending]);

    useTick(tickCallback)

    return (
        <>
            <pixiContainer
                interactive={true}
                eventMode="static"
                width={C.CANVAS_WIDTH}
                height={C.CANVAS_WIDTH}
                hitArea={{
                    contains: (x: number, y: number) => {
                        return x >= 0 && x <= C.CANVAS_WIDTH && y >= 0 && y <= C.CANVAS_HEIGHT;
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
                <CellGraph
                    cells={cells}
                    active={active}
                    userPaused={userPaused}
                    tickPending={ws.tickPending}
                    wsWaiting={!ws.allReady}
                />
            </pixiContainer>
        </>
    )
}

export default MainStage