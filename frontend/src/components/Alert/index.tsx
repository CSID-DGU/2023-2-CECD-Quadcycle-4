import React from 'react';
import { HTMLAttributes } from 'react';
import styled from 'styled-components';
import Icon from '../Icon';

export interface Props extends HTMLAttributes<HTMLDivElement> {
  title: string;
  status?: Status;
  content: string;
  theme?: ColorType;
}

function Alert({ title, status = 'info', theme = 'blue', content, ...rest }: Props) {
  return (
    <AlertItem color={theme} {...rest}>
      <AlertIcon>
        <Icon name={status} color={theme} />
      </AlertIcon>
      <AlertTitle color={theme}>{title}</AlertTitle>
      <AlertContent>{content}</AlertContent>
    </AlertItem>
  );
}

export default Alert;

const AlertItem = styled.div`
  border-radius: 1.75rem;
  background: ${(props) => props.color && `var(--${props.color}-bg)`};
  text-align: left;
  padding: 3.3rem 1.5rem 2.5rem;
  position: relative;
  margin-bottom: 1.05rem;
`;

const AlertTitle = styled.span`
  font-size: var(--large);
  font-weight: var(--bold);
  color: ${(props) => props.color && `var(--${props.color}-text)`};
`;

const AlertContent = styled.p`
  font-size: var(--medium);
  font-weight: var(--text);
  margin-top: 0.7rem;
  line-height: 1rem;
  word-break: keep-all;
`;

const AlertIcon = styled.div`
  position: absolute;
  top: 0;
  right: 0;
  margin-right: 1rem;
  margin-top: 1.2rem;
`;
