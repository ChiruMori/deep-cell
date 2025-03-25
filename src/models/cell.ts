/// <reference path="cell.d.ts" />

// const LOG_CELL_ACTION = false;
const SPEED_DECAY = 0.95;
const HEAL_CNT = 10;
// 细胞分裂的养分消耗
const SPLIT_NEED = 500;
// 每个干细胞，指定帧内只能分裂一次
const SPLIT_COOLDOWN = 100;
// 细胞视线距离
const VISION_DISTANCE = 30;
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
        life: 100,
        hp: 500
    },
    'erythrocyte': {
        maxAcc: 2,
        maxSpeed: 2,
        color: '#FF000088',
        radius: 8,
        life: 2000,
        hp: 3000
    },
    'alveolar': {
        maxAcc: 0.05,
        maxSpeed: 0.1,
        color: '#00FF0088',
        radius: 5,
        life: 2500,
        hp: 5000
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
        behaviorCd: 0,
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
        if (!cell.ml) {
            return;
        }
        const nearbyCells = [] as ICell[]
        cell.feed = 0
        cells.forEach((other) => {
            if (other === cell) {
                return
            }
            // x、y 距离快筛，距离超过视线距离的，直接跳过
            if (Math.abs(cell.x - other.x) > VISION_DISTANCE + other.r || Math.abs(cell.y - other.y) > VISION_DISTANCE + other.r) {
                return
            }
            nearbyCells.push(other)
            const distance = Math.sqrt(Math.pow(cell.x - other.x, 2) + Math.pow(cell.y - other.y, 2))
            // 发生碰撞的情况
            const dd = cell.r + other.r
            if (distance < dd) {
                const angle = Math.atan2(other.y - cell.y, other.x - cell.x)
                const force = ((distance - dd) >> 3) / cell.r / cell.r / 2
                // 追加在速度上
                cell.xSpeed += force * Math.cos(angle)
                cell.ySpeed += force * Math.sin(angle)
                // 红细胞、肺泡会将自身 hp 分给除与自己不同类型的细胞
                if ((cell.type === 'erythrocyte' || cell.type === 'alveolar')
                    && (cell.hp > other.hp)
                    && (cell.type !== other.type)) {
                    let feed = typeProperties(other.type).hp - other.hp
                    feed = Math.max(feed, 0)
                    feed += Math.min(feed, HEAL_CNT)
                    other.hp += feed
                    cell.hp -= feed
                    cell.feed = feed
                }
            }
        })
        // 6 个扇形区域
        cell.surround = [0, 0, 0, 0, 0, 0]
        const minDistances = [Infinity, Infinity, Infinity, Infinity, Infinity, Infinity]

        nearbyCells.forEach((other) => {
            if (other === cell) {
                return
            }

            // 计算当前细胞到目标细胞的角度和距离
            const dx = other.x - cell.x
            const dy = other.y - cell.y
            const distance = Math.sqrt(dx * dx + dy * dy)

            // 如果距离超过视线距离，直接跳过
            if (distance > VISION_DISTANCE + other.r) {
                return
            }

            // 计算当前细胞到目标细胞的角度（范围为 -PI 到 PI）
            let cellToOtherAngle = Math.atan2(dy, dx)
            
            // 将角度转换为 0 到 2PI 的范围
            if (cellToOtherAngle < 0) {
                cellToOtherAngle += 2 * Math.PI
            }
            
            // 计算当前细胞所属的扇形索引（将 0-2PI 分成6个扇形）
            const sectorIndex = Math.floor((cellToOtherAngle / (2 * Math.PI)) * 6)
            
            // 如果当前距离小于之前记录的最小距离，则更新最小距离和扇形索引
            if (distance < minDistances[sectorIndex]) {
                minDistances[sectorIndex] = distance
                cell.surround[sectorIndex] = typeAsNumber(other.type)
            }
        })
    },

    actions: {
        step: (cell: ICell): ICell | null => {
            if (!cell.ml) {
                return null;
            }
            cell.lifeTime++
            cell.life = cell.life - 1
            cell.hp -= 1
            cell.mlForView = cell.ml
            cell.ml = null
            if (cell.behaviorCd > 0) {
                cell.behaviorCd -= 1
            }
            changeAcc(cell, cell.mlForView.strength, cell.mlForView.direction)
            switch (cell.type) {
                case 'stem':
                    // 养分充足且行为参数大于0.9，进行分裂
                    if (cell.hp > SPLIT_NEED && cell.mlForView.kw > 0.9 && cell.behaviorCd <= 0) {
                        cell.hp -= SPLIT_NEED;
                        cell.behaviorCd = SPLIT_COOLDOWN
                        cell.sonCnt += 1
                        return Cells.create({
                            x: cell.x,
                            y: cell.y,
                            r: typeProperties('cancer').radius,
                        }, 'cancer');
                    }
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
                            // HP 需要继承 cancer 的，否则分化成高 hp 细胞将过于有利
                            // cell.hp = typeProperties(cell.type).hp;
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
    VISION_DISTANCE,
    SPLIT_NEED,
};