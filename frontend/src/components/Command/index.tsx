import React, { HTMLAttributes } from 'react';
import styled from 'styled-components';

export interface Props extends HTMLAttributes<HTMLDivElement> {
  children: string;
}

function Command({ children, ...rest }: Props) {
  return <CommandWrapper {...rest}>{children}</CommandWrapper>;
}

export default Command;

const CommandWrapper = styled.div`
  font-size: var(--title);
  font-weight: var(--bold);
  margin-bottom: 0.5rem;
  margin-top: 1rem;
`;
