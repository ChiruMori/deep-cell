/// <reference path="cell.d.ts" />

const SPEED_DECAY = 0.95;
const HEAL_CNT = 100;
const cellTypesConfig: Record<CellType, CellTypeProperties> = {
    'stem': {
        maxAcc: 0.1,
        maxSpeed: 0.5,
        color: '#FFFF0088',
        radius: 3,
        life: 1800,
        hp: 1000
    },
    'cancer': {
        maxAcc: 0.5,
        maxSpeed: 1,
        color: '#88008888',
        radius: 5,
        life: 120,
        hp: 1000
    },
    'erythrocyte': {
        maxAcc: 2,
        maxSpeed: 2,
        color: '#FF000088',
        radius: 8,
        life: 5000,
        hp: 10000
    },
    'alveolar': {
        maxAcc: 0.05,
        maxSpeed: 0.1,
        color: '#00FF0088',
        radius: 5,
        life: 2000,
        hp: 10000
    },
};
const typeAsNumber = (type: CellType | null): number => {
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
            return 0;
    }
}
const typeProperties = (type: CellType): CellTypeProperties => cellTypesConfig[type];
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
        id: Math.random().toString(36).substring(2, 15),
        sonCnt: 0,
        lifeTime: 0,
        feed: 0,
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

    changeAcc: (cell: ICell, rate: number, angle: number): void => {
        cell.xAcc = rate * typeProperties(cell.type).maxAcc * Math.cos(angle)
        cell.yAcc = rate * typeProperties(cell.type).maxAcc * Math.sin(angle)
    },

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
                const feed = typeProperties(other.type).hp - other.hp
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
            cell.lifeTime += 1
            cell.life = cell.life - 1
            cell.hp -= 1
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
                        // TODO: 由机器学习输出：分化类型
                        cell.type = ['stem', 'erythrocyte', 'alveolar'][Math.floor(Math.random() * 3)] as CellType;
                        cell.life = typeProperties(cell.type).life;
                        cell.hp = typeProperties(cell.type).hp;
                        cell.r = typeProperties(cell.type).radius;
                    }
                    return null;
                case 'erythrocyte':
                    return null;
                case 'alveolar':
                    // 视线内没有细胞，恢复 HP
                    if (cell.surround.length === 0 && cell.hp < typeProperties(cell.type).hp - HEAL_CNT) {
                        cell.hp += HEAL_CNT;
                    }
                    return null;
                default:
                    return null;
            }
        },
    },

    typeProperties,
};