// 圆形接口
declare interface ICircle {
    x: number;
    y: number;
    r: number;
}

// 可移动对象接口
declare interface IMovable {
    xSpeed: number;
    ySpeed: number;
    xAcc: number;
    yAcc: number;
}

// 细胞接口
declare interface ICell extends ICircle, IMovable {
    type: CellType;
    life: number;
    hp: number;
    surround: number[];
    id: string;
    sonCnt: number;
    lifeTime: number;
    feed: number;
}

// 细胞类型
declare type CellType = 'stem' | 'cancer' | 'erythrocyte' | 'alveolar'
declare type CellTypeCounter = {
    [key in CellType]?: number;
}
declare interface CellTypeProperties {
    color: string;
    maxAcc: number;
    maxSpeed: number;
    radius: number;
    life: number;
    hp: number;
}

// 矩形
declare interface IRect {
    x: number;
    y: number;
    w: number;
    h: number;
}