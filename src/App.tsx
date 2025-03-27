import { Application } from '@pixi/react'
import MainStage from './components/MainStage'
import LeftSide from './components/LeftSide'
import RightSide from './components/RightSide'
import { useState } from 'react';

function App() {
  const [cells, setCells] = useState<ICell[]>([]);
  const [selectedCell, setSelectedCell] = useState<ICell | undefined>(undefined)
  const [currentRound, setCurrentRound] = useState(1)
  const [userPaused, setUserPaused] = useState(false)
  const [cnt, setCnt] = useState<CellTypeCounter>({
    stem: 0,
    cancer: 0,
    erythrocyte: 0,
    alveolar: 0,
  });

  return (
    <>
      {/* 使画布水平居中 */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        position: 'absolute',
        height: '100%',
        width: '100%',
        backgroundColor: '#001',
        color: 'white'
      }}>
        <LeftSide cells={cells} cnt={cnt} currentRound={currentRound} userPaused={userPaused} setUserPaused={setUserPaused} />
        <Application width={800} height={600}>
          <MainStage setCnt={setCnt} setCells={setCells} cells={cells} setSelectedCell={setSelectedCell} setCurrentRound={setCurrentRound} userPaused={userPaused} />
        </Application>
        <RightSide selectedCell={selectedCell} />
      </div>
    </>
  )
}

export default App
