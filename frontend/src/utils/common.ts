export const textColors: Record<string, string> = {
  red: '#db1824',
  blue: '#0d68fc',
  geekblue: '#1c20ba',
  green: '#009717',
  lime: '#61ae18',
  purple: '#5a0ca5',
  orange: '#de651d',
  gold: '#da801a',
  magenta: '#ce1374',
  cyan: '#008c91',
  grey: '#cccccc',
  primary: '#2E3192',
  white: '#fff',
};

export const colors: Record<string, string> = {
  red: '#ffefee',
  blue: '#dff2ff',
  geekblue: '#edf3ff',
  green: '#f2ffeb',
  lime: '#fbffe3',
  purple: '#faedff',
  orange: '#fff7e3',
  gold: '#fffbe3',
  magenta: '#ffedf5',
  cyan: '#dcfffa',
  grey: '#fcfdff',
  primary: '#2E3192',
  white: '#fff',
};

export const resultMessage: Record<FileLoadedStatusType, string> = {
  ready: '수정을 원하는 엑셀 파일과 명령어를 입력하세요.',
  uploaded: '',
  restored: '',
  loading: '언어 모델 처리 및 분석 중입니다. 잠시만 기다려주세요.',
  loaded: '분석 완료되었습니다. 다음 단계로 이동합니다.',
  preview: '작업 사항을 적용 중입니다. 잠시만 기다려주세요.',
  edit: '작업 사항을 적용 중입니다. 잠시만 기다려주세요.',
  submit: '요청한 작업이 완료되었습니다. 다운로드 버튼을 통해 확인가능합니다.',
};

export const guideMessage: Record<string, GuideInfoType[]> = {
  ready: [
    {
      title: '엑셀 파일 업로드하기',
      description:
        '수정을 원하는 엑셀 파일을 업로드합니다. 파일 확장자는 .xls 혹은 .xlsx만 가능합니다.',
      color: 'magenta',
    },
  ],
  uploaded: [
    {
      title: '엑셀 프리뷰 확인',
      description:
        '업로드한 파일의 프리뷰를 확인할 수 있습니다. 프리뷰 문서 내 수정 작업이 가능합니다. 추후 다운로드 시 변경 결과가 반영됩니다.',
      color: 'blue',
    },
    {
      title: '요청할 엑셀 작업 입력하기',
      description:
        '엑셀 파일에서 진행하고 싶은 작업을 입력합니다. 예로, "급여 열의 합계를 구하고 싶어요."와 같이 원하는 작업을 작성합니다.',
      color: 'green',
    },
  ],
  restored: [
    {
      title: '엑셀 프리뷰 확인',
      description:
        '업로드한 파일의 프리뷰를 확인할 수 있습니다. 프리뷰 문서 내 수정 작업이 가능합니다. 추후 다운로드 시 변경 결과가 반영됩니다.',
      color: 'blue',
    },
    {
      title: '요청할 엑셀 작업 입력하기',
      description:
        '엑셀 파일에서 진행하고 싶은 작업을 입력합니다. 예로, "급여 열의 합계를 구하고 싶어요."와 같이 원하는 작업을 작성합니다.',
      color: 'green',
    },
  ],
  loading: [
    {
      title: '자연어 처리 모델 실행 중입니다.',
      description: '자연어 처리 결과가 곧 생성됩니다. 잠시만 기다려주세요.',
      color: 'geekblue',
    },
  ],
  loaded: [
    {
      title: '자연어 처리 모델 처리가 완료되었습니다.',
      description: '자연어 처리 결과가 곧 생성됩니다. 잠시만 기다려주세요.',
      color: 'gold',
    },
  ],
  preview: [
    {
      title: '엑셀 프리뷰 확인',
      description:
        '업로드한 파일의 프리뷰를 확인할 수 있습니다. 프리뷰 문서 내 수정 작업이 가능합니다. 추후 다운로드 시 변경 결과가 반영됩니다.',
      color: 'lime',
    },
  ],
  range: [
    {
      title: 'Range 타입 입력',
      description:
        '프리뷰에 보여지는 엑셀 표에서 해당하는 영역을 지정 혹은 드래그를 이용해 선택합니다. ',
      color: 'purple',
    },
  ],
  cell: [
    {
      title: 'Cell 타입 입력',
      description: '프리뷰에 보여지는 엑셀 표에서 하나의 셀을 선택합니다.',
      color: 'blue',
    },
  ],
  number: [
    {
      title: 'Number 타입 입력',
      description: '질문에 적합한 값을 숫자로 입력합니다. 실수 형식으로만 입력 가능합니다.',
      color: 'magenta',
    },
  ],
  text: [
    {
      title: 'Number 타입 입력',
      description: '질문에 적합한 값을 텍스트로 입력합니다.',
      color: 'green',
    },
  ],
  criteria: [
    {
      title: 'Criteria 타입 입력',
      description: '질문에 적합한 조건을 지정하는 수, 식 또는 텍스트를 입력해주세요.',
      color: 'cyan',
    },
  ],
  list: [
    {
      title: 'List 타입 입력',
      description:
        '원하는 값을 ‘,’를 이용하여 연속해서 입력합니다. 예를 들면, [”안녕”, “하세요”]로 입력할 수 있습니다.',
      color: 'orange',
    },
  ],
  boolean: [
    {
      title: 'Boolean 타입 입력',
      description:
        '정확하게 일치하는 값을 찾으려면 FALSE, 비슷한 값을 찾으려면 TRUE(생략가능)을 선택해주세요. 생략하려면 TRUE를 선택하세요.',
      color: 'geekblue',
    },
  ],
  submit: [
    {
      title: '엑셀 작업 완료',
      description:
        '엑셀 편집 작업이 완료되었습니다. 프리뷰를 통해 결과 확인 후 요청한 작업이 맞다면 다운로드 버튼을 통해 다운가능합니다. 추가 작업이 있다면 Continue를, 원하지 않는 결과라면 Rollback을 선택하세요.',
      color: 'orange',
    },
  ],
};

export const GITHUB_URL = 'https://github.com/Quad-Cycle/Excelerator';
