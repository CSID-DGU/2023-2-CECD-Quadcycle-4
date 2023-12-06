import React, { ButtonHTMLAttributes } from 'react';
import styled from 'styled-components';
import Icon from '../Icon';
import { textColors, colors } from '../../utils/common';

type ButtonColors = 'white' | 'primary';

export interface Props extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'color'> {
  text: string;
  color?: ButtonColors;
  icon: string;
}

function Button({ text, color = 'white', icon, ...rest }: Props) {
  const buttonColor = color;
  const lineColor = color === 'white' ? 'primary' : 'white';
  return (
    <ButtonElement buttoncolor={buttonColor} linecolor={lineColor} {...rest}>
      {icon && <Icon name={icon} color={lineColor} />}
      <span>{text}</span>
    </ButtonElement>
  );
}

export default Button;

const ButtonElement = styled.button<{ buttoncolor: ButtonColors; linecolor: ButtonColors }>`
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

  &:disabled {
    background: var(--grey1);
    cursor: not-allowed;
  }
`;
