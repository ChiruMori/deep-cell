import { extend } from '@pixi/react'
import {
    Container,
    EventSystem,
    Graphics,
    Text,
} from 'pixi.js'
import { JSX, useCallback } from "react";
import { Cells } from '../../common/cell';
import C from '../../common/constants'

extend({
    Container,
    Graphics,
    Text,
    EventSystem
})

interface GraphProps {
    cells: ICell[];
    userPaused: boolean;
    tickPending: boolean;
    wsWaiting: boolean;
    active: string | undefined;
}

export default function (props: GraphProps): JSX.Element {


    const drawCallback = useCallback((graphics: Graphics) => {
        // 绘制网格
        graphics.clear()
        graphics.setFillStyle({ color: 'black' })
        const w = C.CANVAS_WIDTH
        const h = C.CANVAS_HEIGHT
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
        props.cells.forEach((cell) => {
            if (cell.id === props.active) {
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
    }, [props.cells])

    return <>
        <pixiGraphics draw={drawCallback} />
        {props.wsWaiting && (
            <pixiText text='服务器连接中...'
                x={800 / 2}
                y={600 / 2}
                anchor={0.5}
                style={{
                    fontSize: 24,
                    fill: '#ffffff'
                }}
            />
        )}
        {props.tickPending && (
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
        {props.userPaused && (
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
    </>
}