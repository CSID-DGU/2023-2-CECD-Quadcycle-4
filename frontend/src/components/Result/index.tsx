import React, { HTMLAttributes } from 'react';
import styled from 'styled-components';
import Icon from '../Icon';

export interface Props extends HTMLAttributes<HTMLDivElement> {
  status?: 'loading' | Status;
}

function Result({ status = 'loading', children, ...rest }: Props) {
  return (
    <ResultContainer {...rest}>
      <Icon name={status === 'loading' ? 'spinner' : status} color={ColorByStatus[status]} />
      <span>{children}</span>
    </ResultContainer>
  );
}

export default Result;

const ColorByStatus: Record<string, string> = {
  loading: 'primary',
  success: '#65D7E7',
  info: '#5789E3',
  warning: '#FBBF09',
  error: '#DD626A',
};

const ResultContainer = styled.div`
  border-radius: 1.25rem;
  background: var(--White, #fff);
  box-shadow: var(--dropdown);
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: center;
  padding-top: 6rem;
  padding-bottom: 6rem;
  flex-direction: column;
  gap: 1rem;

  span {
    text-align: center;
    font-weight: var(--text);
    font-size: var(--large);
  }
`;
