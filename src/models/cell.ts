/// <reference path="cell.d.ts" />

// const LOG_CELL_ACTION = false;
const SPEED_DECAY = 0.95;
const HEAL_CNT = 10;
const cellTypesConfig: Record<CellType, CellTypeProperties> = {
    'stem': {
        maxAcc: 0.1,
        maxSpeed: 0.5,
        color: '#FFFF0088',
        radius: 3,
        life: 180,
        hp: 100
    },
    'cancer': {
        maxAcc: 0.5,
        maxSpeed: 1,
        color: '#88008888',
        radius: 5,
        life: 50,
        hp: 100
    },
    'erythrocyte': {
        maxAcc: 2,
        maxSpeed: 2,
        color: '#FF000088',
        radius: 8,
        life: 500,
        hp: 300
    },
    'alveolar': {
        maxAcc: 0.05,
        maxSpeed: 0.1,
        color: '#00FF0088',
        radius: 5,
        life: 200,
        hp: 500
    },
};
export const typeAsNumber = (type: CellType | null): number => {
    switch (type) {
        case 'stem':
            return 1;
        case 'cancer':
            return 2;
        case 'erythrocyte':
            return 3;
        case 'alveolar':
            return 4;
        default:
            console.error('Error type', type)
            return 0;
    }
}
const typeProperties = (type: CellType): CellTypeProperties => cellTypesConfig[type];
const changeAcc = (cell: ICell, rate: number, angle: number): void => {
    cell.xAcc = rate * typeProperties(cell.type).maxAcc * Math.cos(angle)
    cell.yAcc = rate * typeProperties(cell.type).maxAcc * Math.sin(angle)
}
export const Cells = {
    checkHit: (cell: ICell, x: number, y: number): boolean => {
        const dx = cell.x - x;
        const dy = cell.y - y;
        return Math.sqrt(dx * dx + dy * dy) <= cell.r;
    },
    create: (circle: ICircle, type: CellType): ICell => ({
        ...circle,
        xSpeed: 0,
        ySpeed: 0,
        xAcc: 0,
        yAcc: 0,
        type,
        life: typeProperties(type).life * Math.random() * 1.1,
        hp: typeProperties(type).hp * Math.random() * 1.1,
        surround: [-1, -1, -1, -1, -1, -1],
        bornSurround: [-1, -1, -1, -1, -1, -1],
        id: Math.random().toString(36).substring(2, 15),
        sonCnt: 0,
        lifeTime: 0,
        feed: 0,
        ml: null,
        mlForView: null,
    }),

    move: (cell: ICell): void => {
        // 速度自带衰减
        cell.xSpeed = cell.xSpeed * SPEED_DECAY + cell.xAcc
        cell.ySpeed = cell.ySpeed * SPEED_DECAY + cell.yAcc
        // 超速限制
        if (cell.xSpeed > typeProperties(cell.type).maxSpeed) {
            cell.xSpeed = typeProperties(cell.type).maxSpeed
        }
        if (cell.xSpeed < -typeProperties(cell.type).maxSpeed) {
            cell.xSpeed = -typeProperties(cell.type).maxSpeed
        }
        if (cell.ySpeed > typeProperties(cell.type).maxSpeed) {
            cell.ySpeed = typeProperties(cell.type).maxSpeed
        }
        if (cell.ySpeed < -typeProperties(cell.type).maxSpeed) {
            cell.ySpeed = -typeProperties(cell.type).maxSpeed
        }
        cell.x = cell.x + cell.xSpeed
        cell.y = cell.y + cell.ySpeed
    },

    boundsFix: (cell: ICell, bound: IRect): void => {
        if (cell.x - cell.r < bound.x) {
            cell.x = bound.x + cell.r;
            cell.xSpeed = 0;
        } else if (cell.x + cell.r > bound.x + bound.w) {
            cell.x = bound.x + bound.w - cell.r;
            cell.xSpeed = 0;
        }

        if (cell.y - cell.r < bound.y) {
            cell.y = bound.y + cell.r;
            cell.ySpeed = 0;
        } else if (cell.y + cell.r > bound.y + bound.h) {
            cell.y = bound.y + bound.h - cell.r;
            cell.ySpeed = 0;
        }
    },

    changeAcc,

    collide: (cell: ICell, cells: ICell[]): void => {
        const nearbyCells = [] as ICell[]
        cells.forEach((other) => {
            if (other === cell) {
                return
            }
            // x、y 距离快筛，超过双方直径的，直接跳过
            const dd = (cell.r + other.r) * 2
            if (Math.abs(cell.x - other.x) > dd || Math.abs(cell.y - other.y) > dd) {
                return
            }
            // 红细胞、肺泡会将自身 hp 分给除与自己不同类型的细胞
            cell.feed = 0
            if ((cell.type === 'erythrocyte' || cell.type === 'alveolar')
                && (cell.hp > typeProperties(other.type).hp)
                && (cell.type !== other.type)) {
                let feed = typeProperties(other.type).hp - other.hp
                feed = Math.min(feed, HEAL_CNT)
                other.hp += feed
                cell.feed = feed
            }
            nearbyCells.push(other)
            const distance = Math.sqrt(Math.pow(cell.x - other.x, 2) + Math.pow(cell.y - other.y, 2))
            if (distance < dd / 2) {
                const angle = Math.atan2(other.y - cell.y, other.x - cell.x)
                const force = (distance - dd / 2) / cell.r / cell.r / 2
                // 追加在速度上
                cell.xSpeed += force * Math.cos(angle)
                cell.ySpeed += force * Math.sin(angle)
            }
        })
        // 每个细胞有 6 个视线，以细胞中心向 6 个方向发射射线，每个射线夹角60°
        // 射线接触到的细胞为该视线的目标细胞（不要求严格第一个细胞）
        cell.surround = []
        for (let i = 0; i < 6; i++) {
            const angle = i * Math.PI / 3
            const x = cell.x + Math.cos(angle) * cell.r
            const y = cell.y + Math.sin(angle) * cell.r
            let minDistance = Infinity
            let targetCell = null as ICell | null
            nearbyCells.forEach((other) => {
                if (other === cell) {
                    return
                }
                // 计算射线与目标细胞的交点
                const dx = other.x - x
                const dy = other.y - y
                const d = Math.sqrt(dx * dx + dy * dy)
                const r1 = cell.r
                const r2 = other.r
                if (d > r1 + r2) {
                    return
                }
                const a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
                const h = Math.sqrt(r1 * r1 - a * a)
                const x2 = x + a * dx / d
                const y2 = y + a * dy / d
                const x3 = x2 + h * dy / d
                const y3 = y2 - h * dx / d
                // 计算交点到射线起点的距离
                const distance = Math.sqrt(Math.pow(x3 - x, 2) + Math.pow(y3 - y, 2))
                if (distance < minDistance) {
                    minDistance = distance
                    targetCell = other
                }
            })
            cell.surround.push(targetCell ? typeAsNumber(targetCell.type) : 0)
        }
    },

    actions: {
        step: (cell: ICell): ICell | null => {
            if (!cell.ml) {
                return null;
            }
            cell.lifeTime += 1
            cell.life = cell.life - 1
            cell.hp -= 1
            cell.mlForView = cell.ml
            cell.ml = null
            changeAcc(cell, cell.mlForView.strength, cell.mlForView.direction)
            switch (cell.type) {
                case 'stem':
                    // 养分充足且 CD 时间到，进行分裂
                    if (cell.life % 1000 === 0) {
                        return Cells.create({
                            x: cell.x,
                            y: cell.y,
                            r: typeProperties('cancer').radius,
                        }, 'cancer');
                    }
                    cell.sonCnt += 1
                    return null;
                case 'cancer':
                    // CD 时间到，进行分化
                    if (cell.life <= 0) {
                        const types = ['stem', 'erythrocyte', 'alveolar']
                        const indexByKw = Math.floor(cell.mlForView.kw * types.length)
                        // 如果 kw 为 0，则不进行分化，直接死亡
                        if (cell.mlForView.kw !== 0) {
                            cell.type = types[indexByKw] as CellType;
                            cell.life = typeProperties(cell.type).life;
                            cell.hp = typeProperties(cell.type).hp;
                            cell.r = typeProperties(cell.type).radius;
                            cell.bornSurround = [...cell.surround]
                        }
                    }
                    return null;
                case 'erythrocyte':
                    return null;
                case 'alveolar':
                    // 视线内没有细胞，恢复 HP
                    const alone = cell.surround.filter(s => s <= 0).length === 6
                    if (alone) {
                        cell.hp += HEAL_CNT;
                        if (cell.hp > typeProperties(cell.type).hp) {
                            cell.hp = typeProperties(cell.type).hp;
                        }
                    }
                    return null;
                default:
                    return null;
            }
        },
    },

    typeProperties,
};