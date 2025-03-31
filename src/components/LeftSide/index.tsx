import { JSX, SetStateAction, useEffect, useState } from "react";

interface Props {
  cells: ICell[]
  cnt: CellTypeCounter,
  currentRound: number,
  setUserPaused: (pause: SetStateAction<boolean>) => void,
  roundTick: number,
  userPaused: boolean
}

export default function LeftSide({ cells, cnt, currentRound, setUserPaused, roundTick, userPaused }: Props): JSX.Element {

  const [maxRoundTick, setMaxRoundTick] = useState(0)
  const [maxCellCnt, setMaxCellCnt] = useState(0)

  useEffect(() => {
    if (roundTick > maxRoundTick) {
      setMaxRoundTick(roundTick)
    }
  }, [roundTick, maxRoundTick])

  useEffect(() => {
    if (cells.length > maxCellCnt) {
      setMaxCellCnt(cells.length)
    }
  }, [cells, maxCellCnt])

  return (
    <div style={{
      width: '200px',
      padding: '20px',
      backgroundColor: 'rgba(0,0,0,0.7)',
      color: 'white',
      borderRadius: '8px',
      marginRight: '20px'
    }}>
      <h3 style={{ marginBottom: '15px' }}>细胞统计</h3>
      <div style={{ lineHeight: '1.8' }}>
        <div>总数: {cells.length}</div>
        <div>干细胞: {cnt.stem}</div>
        <div>未分化: {cnt.cancer}</div>
        <div>红细胞: {cnt.erythrocyte}</div>
        <div>肺泡细胞: {cnt.alveolar}</div>
        <hr />
        <div>当前轮次: {currentRound} - {roundTick}Tick</div>
        <div>最长存活时间: {maxRoundTick}Tick</div>
        <div>最多细胞数: {maxCellCnt}</div>
        <hr />
        <button onClick={() => setUserPaused(!userPaused)}>
          {userPaused ? '继续' : '暂停'}
        </button>
      </div>
    </div>
  )
}