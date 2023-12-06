import React, { InputHTMLAttributes } from 'react';
import styled from 'styled-components';
import Icon from '../Icon';
import { textColors, colors } from '../../utils/common';

type ButtonColors = 'white' | 'primary' | 'grey';

export interface Props extends Omit<InputHTMLAttributes<HTMLInputElement>, 'color'> {
  text: string;
  color?: ButtonColors;
  icon: string;
}

function InputButton({ text, color = 'white', icon, onChange, ...rest }: Props) {
  const buttonColor = color;
  const lineColor = color === 'white' ? 'primary' : 'white';

  return (
    <Wrapper buttoncolor={buttonColor} linecolor={lineColor}>
      <label htmlFor='input-file'>
        {icon && <Icon name={icon} color={lineColor} />}
        <span>{text}</span>
      </label>
      <input type='file' accept='.xlsx' id='input-file' onChange={onChange} {...rest} />
    </Wrapper>
  );
}

export default InputButton;

const Wrapper = styled.div<{ buttoncolor: ButtonColors; linecolor: ButtonColors }>`
  width: fit-content;
  min-width: 300px;

  label {
    color: ${(props) => textColors[props.linecolor]};
    border: 1px solid ${(props) => textColors[props.linecolor]};
    box-shadow: none;
    background: ${(props) => colors[props.buttoncolor]};
    overflow: visible;
    cursor: pointer;
    padding: 0.5rem 1.5rem 0.5rem 1.25rem;
    border-radius: 0.625rem;
    display: flex;
    gap: 0.5rem;
    align-items: center;
    justify-content: center;
    font-size: var(--medium);
    line-height: 0.875rem;
    font-weight: var(--bold);

    &:hover {
      background: var(--primary3, #f6faff);
    }
  }

  input {
    display: none;
  }
`;
