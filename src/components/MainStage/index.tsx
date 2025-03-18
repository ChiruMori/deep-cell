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

const MainStage = ({ setCnt, setCells, cells, setSelectedCell }: {
    cells: ICell[],
    setCells: (value: SetStateAction<ICell[]>) => void,
    setCnt: (value: SetStateAction<CellTypeCounter>) => void
    setSelectedCell: (value: SetStateAction<ICell | undefined>) => void
}) => {
    const [active, setActive] = useState('')

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

    const tickCallback = useCallback(() => {
        // 通过函数式更新获取最新cells
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

            const newCells = currentCells.filter((cell) => {
                // 随机改变cell的加速度（由深度学习输出）
                const force = Math.random()
                const angle = Math.random() * Math.PI * 2
                Cells.changeAcc(cell, force, angle)
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
                newCnt[cell.type] = (newCnt[cell.type] || 0) + (alive ? 1 : 0)
                return alive
            })
            // 合并细胞数组
            newCells.push(...bornCells)
            // 每次都更新状态，不再检查长度变化
            setCnt(newCnt)
            return newCells;
        })

    }, [setCells, setCnt])

    const initCells = useCallback(() => {
        setCells(prev => {
            const newCells = [...prev];
            // 保留现有细胞并追加新细胞
            for (let i = 0; i < 10; i++) {
                newCells.push(Cells.create({
                    x: Math.random() * 800,
                    y: Math.random() * 600,
                    r: Cells.typeProperties('stem').radius
                }, 'stem'));
            }
            for (let i = 0; i < 10; i++) {
                newCells.push(Cells.create({
                    x: Math.random() * 800,
                    y: Math.random() * 600,
                    r: Cells.typeProperties('cancer').radius
                }, 'cancer'));
            }
            return newCells;
        });
    }, [setCells])

    useEffect(() => {
        initCells()
    }, []) // 空依赖数组确保只执行一次

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