import { JSX, SetStateAction, useEffect, useRef, useState } from "react";

interface Props {
  cells: ICell[]
  cnt: CellTypeCounter,
  currentRound: number,
  pause: boolean,
  setUserPaused: (pause: SetStateAction<boolean>) => void,
  userPaused: boolean
}

export default function LeftSide({ cells, cnt, currentRound, pause, setUserPaused, userPaused }: Props): JSX.Element {

  // 秒数计时（setInterval，当currentRound变化时，重新计时）
  const [roundSecond, setRoundSecond] = useState(0)
  const [maxRoundSecond, setMaxRoundSecond] = useState(0)
  const [maxCellCnt, setMaxCellCnt] = useState(0)

  // 使用 ref 存储可变的暂停状态
  const pauseRef = useRef(false);
  useEffect(() => {
    pauseRef.current = pause || userPaused;
  }, [pause, userPaused]);

  useEffect(() => {
    let interval: number;
    
    // 只在轮次变化时重置计时
    const handler = () => {
      if (!pauseRef.current) {
        setRoundSecond(prev => prev + 1);
      }
    };

    // 初始立即执行一次
    setRoundSecond(0);
    interval = setInterval(handler, 1000);
    
    return () => {
      clearInterval(interval);
    };
  }, [currentRound]); // 只监听 currentRound 变化

  useEffect(() => {
    if (roundSecond > maxRoundSecond) {
      setMaxRoundSecond(roundSecond)
    }
  }, [roundSecond, maxRoundSecond])

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
        <div>当前轮次: {currentRound} - {roundSecond}s</div>
        <div>最长存活时间: {maxRoundSecond}s</div>
        <div>最多细胞数: {maxCellCnt}</div>
        <hr />
        <button onClick={() => setUserPaused(!userPaused)}>
          {userPaused ? '继续' : '暂停'}
        </button>
      </div>
    </div>
  )
}