import React, { useState, useEffect, HTMLAttributes, MutableRefObject } from 'react';
import '@grapecity/spread-sheets-io';
import { IEventTypeObj, SpreadSheets, Worksheet } from '@grapecity/spread-sheets-react';
import GC from '@grapecity/spread-sheets';
import { SetterOrUpdater, useRecoilValue } from 'recoil';
import { FileType, deepClone, options } from '../../utils/spread';
import { selectedFileState } from '../../store/spread';
import { saveAs } from 'file-saver';
import styled from 'styled-components';

export interface Props extends HTMLAttributes<HTMLDivElement> {
  forwardedRef: MutableRefObject<{
    save: () => void;
    applyFormula: (func: string, parameters: string[], cell: string) => void;
    resetFormula: (cell: string) => void;
    refresh: () => void;
  }>;
  handlePreview: (range: string) => void;
  setApplyStatus: SetterOrUpdater<FileLoadedStatusType>;
}

function Preview({ forwardedRef, handlePreview, setApplyStatus, ...rest }: Props) {
  const [spread, setSpread] = useState<GC.Spread.Sheets.Workbook | null>(null);
  const [lastValue, setLastValue] = useState<any>(null);
  const selectedFile = useRecoilValue(selectedFileState);

  const fileType = FileType.Excel;

  let hostStyle = {
    width: '97%',
    height: '100%',
    border: '1px solid darkgray',
  };

  const initSpread = (spread: GC.Spread.Sheets.Workbook) => {
    setSpread(spread);
  };

  const open = () => {
    var file = selectedFile;
    if (!file) {
      return;
    }

    spread?.import(
      file,
      function () {},
      function () {},
      deepClone(options),
    );
    setSpread(spread);
  };

  const save = () => {
    var fileName = 'export.' + fileType;

    spread?.export(
      function (blob: Blob) {
        saveAs(blob, fileName);
      },
      function () {},
      deepClone(options),
    );
  };

  const applyFormula = (func: string, params: string[], cell: string) => {
    try {
      const sheet = spread?.getActiveSheet();
      const resultCell = excelCellToIndex(cell);
      if (!resultCell) return;

      const parameters = params.map(String);

      if (func === 'VLOOKUP' || func === 'HLOOKUP' || func === 'MATCH') {
        [parameters[0], parameters[1]] = [parameters[1], parameters[0]];
      }
      console.log(parameters, `=${func}(${parameters.join(', ')})`);
      setLastValue(sheet?.getValue(resultCell[0], resultCell[1]) || '');
      sheet?.setFormula(resultCell[0], resultCell[1], `=${func}(${parameters.join(', ')})`);
      setApplyStatus('submit');
    } catch (err) {
      alert('다시 시도해주십시오.');
    }
  };

  const resetFormula = (cell: string) => {
    const sheet = spread?.getActiveSheet();
    const resultCell = excelCellToIndex(cell);
    if (!resultCell) return;

    sheet?.clear(
      resultCell[0],
      resultCell[1],
      1,
      1,
      GC.Spread.Sheets.SheetArea.viewport,
      GC.Spread.Sheets.StorageType.data,
    );
    sheet?.setValue(resultCell[0], resultCell[1], lastValue);
  };

  const refresh = () => {
    spread?.refresh();

    let sheet = spread?.getActiveSheet();
    if (!sheet) return;

    sheet.options.selectionBackColor = 'rgba(204,255,51, 0.3)';
    sheet.options.selectionBorderColor = 'Accent 3 -40';
    // sheet.selectionPolicy(2);
  };

  function excelCellToIndex(cell: string) {
    const match = cell.match(/([A-Z]+)(\d+)/);
    if (!cell) return;

    if (!match) {
      throw new Error('Invalid Excel cell format');
    }

    const column = match[1];
    const row = parseInt(match[2], 10);

    const columnIdx =
      column
        .split('')
        .reduce((acc, char) => acc * 26 + (char.charCodeAt(0) - 'A'.charCodeAt(0) + 1), 0) - 1;

    const rowIdx = row - 1;
    return [rowIdx, columnIdx];
  }

  const onSelectionChanged = (
    _: IEventTypeObj,
    args: GC.Spread.Sheets.ISelectionChangedEventArgs,
  ) => {
    let selection = args.newSelections.pop();
    if (!selection) return;
    const selected = indicesToExcelRange(selection);
    handlePreview(selected);

    // let sheet = spread?.getActiveSheet();
    // if (!sheet) return;
    // sheet.addSelection(selection.row, selection.col, selection.rowCount, selection.colCount);
    // sheet.addSelection(4, 3, 2, 2);
  };

  function indicesToExcelCell(row: number, col: number): string {
    const columnChar = String.fromCharCode('A'.charCodeAt(0) + col);
    const rowNumber = row + 1;

    return `${columnChar}${rowNumber}`;
  }

  function indicesToExcelRange(selection: GC.Spread.Sheets.Range): string {
    const { row, rowCount, col, colCount } = selection;
    const startCell = indicesToExcelCell(row, col);
    let endCell;

    if (colCount === 1 && rowCount === 1) {
      return startCell;
    } else {
      const endRow = row + rowCount - 1;
      const endCol = col + colCount - 1;
      endCell = indicesToExcelCell(endRow, endCol);
    }

    return `${startCell}:${endCell}`;
  }

  useEffect(() => {
    if (!selectedFile) return;
    open();
  }, [selectedFile]);

  useEffect(() => {
    if (forwardedRef && forwardedRef.current) {
      forwardedRef.current = {
        save,
        applyFormula,
        resetFormula,
        refresh,
      };
    }
  }, [forwardedRef, save, applyFormula, resetFormula, refresh]);

  return (
    <PreviewContainer {...rest}>
      <SpreadSheets
        workbookInitialized={(spread) => initSpread(spread)}
        hostStyle={hostStyle}
        selectionChanged={onSelectionChanged}
      >
        <Worksheet></Worksheet>
      </SpreadSheets>
    </PreviewContainer>
  );
}

export default Preview;

const PreviewContainer = styled.div`
  margin: 1.8rem 0;
  width: 100%;
  flex: 1;

  #vp,
  table,
  canvas {
    width: 100% !important;
  }
`;
