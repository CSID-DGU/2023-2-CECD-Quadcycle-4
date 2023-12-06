import React, { HTMLAttributes } from 'react';
import styled from 'styled-components';
import { textColors, colors } from '../../utils/common';

export interface Props extends HTMLAttributes<HTMLDivElement> {
  theme?: ColorType;
}

function Badge({ theme = 'blue', children, ...rest }: Props) {
  return (
    <BadgeElement theme={theme} {...rest}>
      {children}
    </BadgeElement>
  );
}

export default Badge;

const BadgeElement = styled.div<{ theme: ColorType }>`
  border-radius: 0.25rem;
  padding: 0.25rem 0.75rem;
  color: ${(props) => textColors[props.theme]};
  background: ${(props) => colors[props.theme]};
  font-size: var(--medium);
`;
