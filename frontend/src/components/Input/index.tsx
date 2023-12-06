import React, { InputHTMLAttributes } from 'react';
import styled from 'styled-components';

export interface Props extends InputHTMLAttributes<HTMLInputElement> {}

function Input({ placeholder = '범위를 드래그하거나 입력하세요.', ...rest }: Props) {
  return <InputElement placeholder={placeholder} {...rest} />;
}

export default Input;

const InputElement = styled.input`
  border-radius: 0.625rem;
  border: 1px solid var(--grey1, #d7d7d7);
  background: #fff;
  color: #005ffc;
  font-weight: var(--bold);
  font-size: var(--text);
  min-width: 15.625rem;
  padding: 0.44rem 0.56rem;

  &:focus,
  &:hover {
    outline: 2px solid var(--primary2);
  }
`;
