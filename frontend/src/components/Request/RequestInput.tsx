import React, { HTMLAttributes, useEffect } from 'react';
import styled from 'styled-components';
import Input from '../Input';
import Toggle from './ToggleInput';
import { debounce } from 'lodash';

export interface Props extends HTMLAttributes<HTMLDivElement> {
  type: ParameterType;
  selectedRange?: string;
  updateParameters: (value: any) => void;
  value?: any;
}

function RequestInput({ type, selectedRange, updateParameters, value }: Props) {
  const debouncedUpdateParameters = debounce(updateParameters, 300);

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    debouncedUpdateParameters(value);
  };

  const onChangeText = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = `"${e.target.value}"`;
    console.log(value);
    debouncedUpdateParameters(value);
  };

  useEffect(() => {
    if (type === 'boolean') updateParameters(true);
  }, [type]);

  switch (type) {
    case 'cell':
    case 'range':
      return (
        <InputWrapper>
          <Input type={'text'} defaultValue={selectedRange} readOnly={true} onChange={onChange} />
        </InputWrapper>
      );
    case 'number':
      return (
        <InputWrapper>
          <Input
            placeholder={'숫자를 입력해주세요.'}
            type={'number'}
            onChange={onChange}
            defaultValue={value}
          />
        </InputWrapper>
      );
    case 'list':
      return (
        <InputWrapper>
          <Input
            type={'text'}
            placeholder={'리스트 형식으로 텍스트를 입력해주세요.'}
            onChange={onChange}
            defaultValue={value}
          />
        </InputWrapper>
      );
    case 'criteria':
      return (
        <InputWrapper>
          <Input
            type={'text'}
            placeholder={'조건을 지정하는 수, 식 또는 텍스트를 입력해주세요.'}
            onChange={onChange}
            defaultValue={value}
          />
        </InputWrapper>
      );

    case 'boolean':
      return (
        <InputWrapper>
          <Toggle updateParameters={updateParameters} defaultChecked={value} />
        </InputWrapper>
      );
    case 'text':
    default:
      return (
        <InputWrapper>
          <Input
            type={'text'}
            placeholder={'텍스트를 입력해주세요.'}
            onChange={onChangeText}
            defaultValue={value}
          />
        </InputWrapper>
      );
  }
}

export default RequestInput;

const InputWrapper = styled.div``;
