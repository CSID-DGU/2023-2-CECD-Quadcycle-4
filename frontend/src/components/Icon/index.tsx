import React, { CSSProperties, HTMLAttributes } from 'react';
import styled from 'styled-components';
import { textColors } from '../../utils/common';
import { images } from '../../utils/images';

export interface Props extends Omit<HTMLAttributes<HTMLSpanElement>, 'color'> {
  name: string;
  color?: keyof ColorType | CSSProperties['color'];
  size?: number;
}

function Icon({ name, color = 'white', size = 24, style, ...rest }: Props) {
  let SelectedIcon = name in images ? images[name as string] : null;

  return (
    <IconWrapper
      style={{
        width: size,
        height: size,
        animation: name === 'spinner' ? 'rotation 1s linear infinite' : '',
        ...style,
      }}
      {...rest}
    >
      {SelectedIcon ? (
        <SelectedIcon
          style={{
            width: '100%',
            height: '100%',
            color: color
              ? (color as string) in textColors
                ? textColors[color as string]
                : (color as string)
              : undefined,
          }}
        />
      ) : null}
    </IconWrapper>
  );
}

export default Icon;

const IconWrapper = styled.span`
  display: inline-block;
  line-height: 0;

  svg {
    width: 100%;
    height: 100%;

    [fill] {
      &:not([fill='none']) {
        fill: currentColor;
      }
    }

    [stroke] {
      &:not([stroke='none']) {
        stroke: currentColor;
      }
    }
  }
`;
