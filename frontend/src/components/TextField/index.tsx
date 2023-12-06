import React, {
  TextareaHTMLAttributes,
  forwardRef,
  useRef,
  ReactElement,
  cloneElement,
  Children,
} from 'react';
import styled from 'styled-components';

export interface Props extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  rightAddon?: ReactElement;
}

// eslint-disable-next-line react/display-name
const TextField = forwardRef<HTMLTextAreaElement, Props>(
  ({ placeholder = '플레이스 홀더', rightAddon, ...rest }, ref) => {
    const right = rightAddon != null ? Children.only(rightAddon) : null;
    const textareaRef: React.MutableRefObject<HTMLTextAreaElement | null> = useRef(null);

    const handleResizeHeight = () => {
      if (!textareaRef.current) return;

      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    };

    return (
      <TextFieldWrapper>
        <textarea
          ref={(inputRef) => {
            textareaRef.current = inputRef;
            if (ref) {
              if (typeof ref === 'function') {
                ref(inputRef);
              } else {
                ref.current = inputRef;
              }
            }
          }}
          placeholder={placeholder}
          onChange={handleResizeHeight}
          rows={1}
          {...rest}
        />
        {right != null
          ? cloneElement(right, {
              ...right.props,
              className: 'textarea-icon',
            })
          : null}
      </TextFieldWrapper>
    );
  },
);
export default TextField;

type TextFieldWrapperType = Pick<Props, 'rightAddon'>;

const TextFieldWrapper = styled.div<TextFieldWrapperType>`
  position: relative;
  /* margin-bottom: 1.5rem; */

  .textarea-icon {
    position: absolute;
    top: 50%;
    transform: translate(-150%, -50%);
    cursor: pointer;

    svg {
      [fill] {
        &:not([fill='none']) {
          fill: var(--grey1);
        }
      }

      [stroke] {
        &:not([stroke='none']) {
          stroke: var(--grey1);
        }
      }
    }

    :hover {
      svg {
        [fill] {
          &:not([fill='none']) {
            fill: var(--primary1);
          }
        }

        [stroke] {
          &:not([stroke='none']) {
            stroke: var(--primary1);
          }
        }
      }
    }
  }

  textarea:hover:not(:disabled) + .textarea-icon,
  textarea:focus:not(:disabled) + .textarea-icon,
  textarea:active:not(:disabled) + .textarea-icon {
    svg {
      [fill] {
        &:not([fill='none']) {
          fill: var(--primary1);
        }
      }

      [stroke] {
        &:not([stroke='none']) {
          stroke: var(--primary1);
        }
      }
    }
  }

  textarea {
    width: 100%;
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    border-radius: 0.625rem;
    background: var(--white, #fff);
    box-shadow: var(--dropdown);
    border: none;
    padding: 1.6rem 1rem;
    line-height: 1.2rem;

    &:focus,
    &:hover {
      outline: 2px solid var(--primary2js);
    }

    overflow-y: hidden;
    overflow: hidden;

    ::placeholder {
      color: var(--grey1);
    }
  }
`;
