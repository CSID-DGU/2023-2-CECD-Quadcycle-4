import { atom } from 'recoil';
import GC from '@grapecity/spread-sheets';

export const spreadState = atom<GC.Spread.Sheets.Workbook | null>({
  key: 'spread',
  default: null,
});

export const selectedFileState = atom<File | null>({
  key: 'selectedFile',
  default: null,
});

export const FileLoadedState = atom<FileLoadedStatusType>({
  key: 'fileLoaded',
  default: 'ready',
});
