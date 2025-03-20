import { JSX, useEffect, useState } from "react";

interface Props {
  cells: ICell[]
  cnt: CellTypeCounter,
  currentRound: number,
}

export default function LeftSide({ cells, cnt, currentRound }: Props): JSX.Element {
  
  // 秒数计时（setInterval，当currentRound变化时，重新计时）
  const [roundSecond, setRoundSecond] = useState(0)
  const [maxRoundSecond, setMaxRoundSecond] = useState(0)
  const [maxCellCnt, setMaxCellCnt] = useState(0)

  useEffect(() => {
    setRoundSecond(0)
    const interval = setInterval(() => {
      setRoundSecond(roundSecond + 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [currentRound])
  
  useEffect(() => {
    if (roundSecond > maxRoundSecond) {
      setMaxRoundSecond(roundSecond)
    }
  }, [roundSecond])

  useEffect(() => {
    if (cells.length > maxCellCnt) {
      setMaxCellCnt(cells.length)
    }
  }, [cells])

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
        <div>当前轮次: {currentRound} - {roundSecond}s</div>
        <div>最长存活时间: {maxRoundSecond}s</div>
        <div>最多细胞数: {maxCellCnt}</div>
      </div>
    </div>
  )
}