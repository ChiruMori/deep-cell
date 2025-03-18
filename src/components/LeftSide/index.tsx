import { JSX } from "react";

interface Props {
  cells: ICell[]
  cnt: CellTypeCounter
}

export default function LeftSide({ cells, cnt }: Props): JSX.Element {
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
      </div>
    </div>
  )
}