import { JSX } from "react";
import { Cells } from "../../models/cell";
interface Props {
    selectedCell?: ICell;
}

export default function RightSide({ selectedCell }: Props): JSX.Element {
    return (
        <div style={{
            width: '200px',
            padding: '20px',
            backgroundColor: 'rgba(0,0,0,0.7)',
            color: 'white',
            borderRadius: '8px',
            marginLeft: '20px'
        }}>
            <div className="space-y-2">
                <h3>细胞详情</h3>
                <h4>{selectedCell ? (selectedCell.type + ' - ' + selectedCell.id) : '未选择'}</h4>
                {selectedCell && (
                    <div
                        style={{
                            lineHeight: '1.8',
                            color: Cells.typeProperties(selectedCell.type).color
                        }}
                    >
                        <div>坐标：</div>
                        <div>({selectedCell.x.toFixed(1)}, {selectedCell.y.toFixed(1)})</div>
                        <div>半径：</div>
                        <div>{selectedCell.r}px</div>
                        <div>剩余生命：</div>
                        <div>{selectedCell.life > 0 ? selectedCell.life : '衰亡'}</div>
                        <div>养分值：</div>
                        <div>{selectedCell.hp > 0 ? selectedCell.hp : '凋亡'}</div>
                        <div>当前速度：</div>
                        <div>{Math.hypot(selectedCell.xSpeed, selectedCell.ySpeed).toFixed(1)}px/frame</div>
                    </div>
                )}
            </div>
        </div>
    )
}