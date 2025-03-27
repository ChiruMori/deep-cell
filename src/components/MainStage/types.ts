
// 实时反馈的接口
export interface RealtimeFeedback {
    id: string;
    state: {
        life: number;
        hp: number;
        surround: number[];
        c_type: number;
    };
    action: {
        angle: number;
        strength: number;
        kw: number;
    };
    immediate_reward: number;
    is_terminal: boolean;
}

export interface TickResult {
    newCells: ICell[];
    feedbacksForTraining: RealtimeFeedback[];
    newRound: boolean;
    cnt: CellTypeCounter;
}