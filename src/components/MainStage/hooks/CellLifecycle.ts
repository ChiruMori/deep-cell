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

        const newCells = currentCells.filter((cell) => {
            if (!cell.ml) {
                return true;
            }
            if (cell.hp < 0 || cell.life < 0) {
                throw new Error('细胞状态异常，状态已过期');
            }
            // 记录动作前的状态
            const prevHp = cell.hp;
            const prevSonCnt = cell.sonCnt || 0;

            // 更新
            Cells.collide(cell, currentCells);
            Cells.move(cell);
            Cells.boundsFix(cell, { x: 0, y: 0, w: 800, h: 600 });

            // 细胞行为
            const born = Cells.actions.step(cell);
            if (born) {
                bornCells.push(born);
            }

            const alive = cell.life > 0 && cell.hp > 0;

            // 计算即时奖励
            const hpChange = cell.hp - prevHp;
            const newSonCnt = cell.sonCnt || 0;
            // 繁殖奖励，需要抵消HP消耗
            const sonChange = (newSonCnt - prevSonCnt) * Cells.SPLIT_NEED * 0.1;

            // 奖励函数
            // HP变化，细胞为其他细胞提供养分，会获得更高奖励
            const hpReward = Math.max(0, Math.min((hpChange * 0.1 + cell.feed * 0.2) * 0.4, 0.4));
            // 成功繁殖（免疫细胞杀死负面细胞，杀死正常细胞时惩罚），40%权重
            const sonReward = Math.max(0, Math.min((sonChange > 0 ? 1 : 0) * 0.4, 0.4));
            // 存活的基础奖励，hp 比例越高奖励越高，最高 0.2
            const aliveReward = alive ? Math.min(cell.hp / Cells.typeProperties(cell.type).hp * 0.2, 0.2) : 0;
            // 计算即时奖励：HP变化 + 繁殖奖励 + 存活奖励 + 额外奖励
            const immediateReward = hpReward + sonReward + aliveReward + cell.behaviorHelper.reward;
            cell.behaviorHelper.reward = 0;
            if (hpChange > 0 && cell.type !== 'alveolar') {
                console.log(hpChange, immediateReward, cell)
            }

            const onlyAliveReward = (hpChange == 0 || hpChange == -1) && sonChange == 0 && alive;
            if (cell.mlForView) {
                cell.mlForView.reward = immediateReward;
            }
            const exp = {
                id: cell.id,
                state: {
                    life: cell.life,
                    hp: cell.hp,
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
            // 收集实时反馈，仅有存活奖励时，降低收集频率
            if (!onlyAliveReward && Math.random() < 0.1) {
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
                        life: cell.lifeTime,
                        hp: cell.hp,
                        surround: cell.surround,
                        c_type: typeAsNumber('cancer')
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

            newCnt[cell.type] = (newCnt[cell.type] || 0) + (alive ? 1 : 0);
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