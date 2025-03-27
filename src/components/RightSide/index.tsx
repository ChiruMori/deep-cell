import { JSX } from "react";
import { Cells } from "../../common/cell";
import Bar from "../Bar";
import Arrow from "../Arrow";
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
                        <div>坐标：({selectedCell.x.toFixed(1)}, {selectedCell.y.toFixed(1)})</div>
                        <div>半径：{selectedCell.r}px</div>
                        <div>当前速度：{Math.hypot(selectedCell.xSpeed, selectedCell.ySpeed).toFixed(1)}px/frame</div>
                        <div>剩余生命：</div>
                        <Bar value={selectedCell.life}
                            maxValue={Cells.typeProperties(selectedCell.type).life}
                            color='#EF4444'
                            bgColor="#374151"
                            deadText="衰亡"
                        />
                        <div>养分值：</div>
                        <Bar value={selectedCell.hp}
                            maxValue={Cells.typeProperties(selectedCell.type).hp}
                            color='#00AAFF'
                            bgColor="#374151"
                            deadText="凋亡"
                        />
                        <div>周围细胞：</div>
                        <div>{selectedCell.surround.map(n => Cells.typeAsText(Cells.numberAsType(n))).join(',')}</div>
                        <hr />
                        <div>方向：</div>
                        <Arrow direction={selectedCell.mlForView?.direction} color={Cells.typeProperties(selectedCell.type).color} />
                        <div>强度：</div>
                        <Bar value={selectedCell.mlForView?.strength ?? 0}
                            maxValue={1}
                            color='#AAFF00'
                            bgColor="#374151"
                            deadText="0"
                        />
                        <div>行为参数：</div>
                        <Bar value={selectedCell.mlForView?.kw ?? 0}
                            maxValue={1}
                            color='#00E7FF'
                            bgColor="#374151"
                            deadText="0"
                        />
                        <hr />
                        <div>当前奖励：</div>
                        <Bar value={selectedCell.mlForView?.reward ?? 0}
                            maxValue={1}
                            color='#84CC16'
                            bgColor="#374151"
                            deadText="无效"
                        />
                    </div>
                )}
            </div>
        </div>
    )
}