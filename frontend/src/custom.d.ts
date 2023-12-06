type Status = 'info' | 'warning' | 'error' | 'success';
type ColorType =
  | 'red'
  | 'blue'
  | 'geekblue'
  | 'green'
  | 'lime'
  | 'purple'
  | 'orange'
  | 'gold'
  | 'magenta'
  | 'cyan'
  | 'grey'
  | 'primary';

interface RequestType {
  request: string;
  type: ParameterType;
}

type FileLoadedStatusType =
  | 'ready'
  | 'uploaded'
  | 'loading'
  | 'loaded'
  | 'preview'
  | 'edit'
  | 'submit'
  | 'restored';

type ParameterType = 'range' | 'cell' | 'number' | 'text' | 'list' | 'criteria' | 'boolean';

interface GuideInfoType {
  title: string;
  description: string;
  color?: ColorType;
}

declare module '*.svg' {
  import React = require('react');

  export const ReactComponent: React.FC<React.SVGProps<SVGSVGElement>>;
  const src: string;
  export default src;
}

declare module 'file-saver';
