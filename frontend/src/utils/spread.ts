import GC from '@grapecity/spread-sheets';

export const FileType = {
  SJS: 'sjs',
  Excel: 'xlsx',
  SSJson: 'ssjson',
  Csv: 'csv',
};

export const options: GC.Spread.Sheets.ImportOptions = {
  fileType: GC.Spread.Sheets.FileType.excel,
};

export const deepClone = (obj: Object) => {
  return JSON.parse(JSON.stringify(obj));
};
