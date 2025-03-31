/// <reference path="./../../../common/cell.d.ts" />
import { useCallback } from 'react';
import { Cells, typeAsNumber } from '../../../common/cell';
import { RealtimeFeedback, TickResult } from '../types';
import C from '../../../common/constants';

// 初始未分化细胞个数
const CANCER_CNT = 200
const CELL_COUNTER_TPL = {
    stem: 0,
    cancer: 0,
    erythrocyte: 0,
    alveolar: 0,
}

export const useCellLifecycle = () => {
    const generateCells = useCallback((count: number) => {
        return Array.from({ length: count },
            () => Cells.create({
                x: Math.random() * C.CANVAS_WIDTH,
                y: Math.random() * C.CANVAS_HEIGHT,
                r: Cells.typeProperties('cancer').radius
            }, 'cancer'));
    }, []);

    const updateCells = useCallback((currentCells: ICell[]): TickResult => {
        // 检查是否需要重置
        if (!currentCells || currentCells.length === 0) {
            return {
                newCells: generateCells(CANCER_CNT),
                newRound: true,
                feedbacksForTraining: [],
                cnt: { ...CELL_COUNTER_TPL }
            };
        }

        const bornCells = [] as ICell[];
        const newCnt = { ...CELL_COUNTER_TPL };

        // 收集实时反馈
        const realtimeFeedbacks: RealtimeFeedback[] = [];

        let newCells = currentCells.filter((cell) => {
            if (!cell.ml) {
                return true;
            }
            if (cell.hp < 0 || cell.life < 0) {
                throw new Error('细胞状态异常，状态已过期');
            }
            // 记录动作前的状态
            const prevSonCnt = cell.sonCnt || 0;

            // 碰撞检测与标记
            Cells.collide(cell, currentCells);
            // 位置更新
            Cells.move(cell);
            Cells.boundsFix(cell, { x: 0, y: 0, w: C.CANVAS_WIDTH, h: C.CANVAS_HEIGHT });

            const alive = cell.life > 0 && cell.hp > 0;

            // 计算即时奖励
            const newSonCnt = cell.sonCnt || 0;
            // 子代数量变化
            cell.behaviorHelper.sonChange = newSonCnt - prevSonCnt;

            newCnt[cell.type] = (newCnt[cell.type] || 0) + (alive ? 1 : 0);
            return alive;
        });

        // HP、生命结算后筛选
        newCells = newCells.filter((cell) => {
            // 没有接受响应的细胞不处理
            if (!cell.ml) {
                return true;
            }
            // 细胞特有行动
            const born = Cells.actions.step(cell, newCnt);
            if (born) {
                bornCells.push(born);
            }
            // 养料结算
            cell.hp += cell.behaviorHelper.hpChange;
            if (cell.hp > Cells.typeProperties(cell.type).hp) {
                cell.hp = Cells.typeProperties(cell.type).hp;
            }
            // 生命结算
            cell.life -= 1;
            // 是否存活
            const alive = cell.life > 0 && cell.hp > 0;
            // 奖励函数
            // HP变化，细胞为其他细胞提供养分，会获得更高奖励，hp 越低时，奖励越高
            const nowHpRatio = cell.hp / Cells.typeProperties(cell.type).hp;
            const hpReward = Math.max(0, Math.min((
                // HP 变化，按当前 hp 比例结算，hp比例越低，奖励越高
                cell.behaviorHelper.hpChange * (1 - nowHpRatio) * 0.1
                // Feed 奖励
                + cell.behaviorHelper.feed * 0.2) 
                * 0.4, 0.4));
            // 成功繁殖（免疫细胞杀死负面细胞，杀死正常细胞时惩罚），40%权重
            const sonReward = Math.max(0, Math.min((cell.behaviorHelper.sonChange > 0 ? 1 : 0) * 0.4, 0.4));
            // S 型曲线奖励，鼓励保持中等HP水平
            const aliveReward = alive ? Math.min(1 / (1 + Math.exp(-(cell.hp / Cells.typeProperties(cell.type).hp - 0.5) * 10)) * 0.2, 0.2) : -0.2;
            // 运动奖励，鼓励细胞尝试运动而不是原地不动
            const moveReward = Math.max(0, Math.min(cell.xSpeed * cell.xSpeed + cell.ySpeed * cell.ySpeed / Cells.typeProperties(cell.type).maxSpeed / Cells.typeProperties(cell.type).maxSpeed) * 0.05, 0.05);
            // 计算即时奖励
            const immediateReward = hpReward + sonReward + aliveReward + cell.behaviorHelper.reward + moveReward;
            // 计算后，清除标记
            cell.behaviorHelper.hpChange = 0;
            cell.behaviorHelper.sonChange = 0;
            cell.behaviorHelper.reward = 0;
            cell.behaviorHelper.feed = 0;

            const onlyAliveReward = (cell.behaviorHelper.hpChange == 0 || cell.behaviorHelper.hpChange == -1) && cell.behaviorHelper.sonChange == 0 && alive;
            if (cell.mlForView) {
                cell.mlForView.reward = immediateReward;
            }
            const exp = {
                id: cell.id,
                state: {
                    life: cell.life / Cells.typeProperties(cell.type).life,
                    hp: cell.hp / Cells.typeProperties(cell.type).hp,
                    surround: cell.surround,
                    c_type: typeAsNumber(cell.type),
                    speed: [cell.xSpeed, cell.ySpeed],
                },
                action: {
                    angle: cell.ml?.direction || 0,
                    strength: cell.ml?.strength || 0,
                    kw: cell.ml?.kw || 0
                },
                immediate_reward: immediateReward,
                is_terminal: !alive
            }
            // 收集实时反馈，仅有存活奖励时，降低收集频率（禁用）
            if (!onlyAliveReward && Math.random() < 1.1) {
                realtimeFeedbacks.push(exp);
            }

            // 细胞挂了，补充一次反馈（用于训练 cancer 分化方向）
            if (!alive) {
                const finalReward =
                    // 分化的子代细胞存活时间越长，奖励越高
                    cell.lifeTime / Cells.typeProperties(cell.type).life * 0.5 +
                    // 繁殖越多，奖励越高
                    cell.sonCnt * 0.2 +
                    // 细胞剩余养分越高，奖励越高，如果因为养分不足死亡，则给予负奖励
                    (cell.hp > 0 ? 0.3 : -0.2);
                cell.behaviorHelper.reward = 0;
                const finalExp = {
                    id: cell.id,
                    state: {
                        life: cell.lifeTime / Cells.typeProperties(cell.type).life,
                        hp: cell.hp / Cells.typeProperties(cell.type).hp,
                        surround: cell.surround,
                        c_type: typeAsNumber('cancer'),
                        speed: [cell.xSpeed, cell.ySpeed],
                    },
                    action: {
                        angle: cell.ml?.direction || 0,
                        strength: cell.ml?.strength || 0,
                        kw: cell.ml?.kw || 0
                    },
                    immediate_reward: finalReward,
                    is_terminal: true
                }
                realtimeFeedbacks.push(finalExp);
            }
            return alive;
        });

        // 合并出生的细胞
        return {
            newCells: [...newCells, ...bornCells],
            newRound: false,
            feedbacksForTraining: realtimeFeedbacks,
            cnt: newCnt
        };
    }, []);

    return { updateCells };
};