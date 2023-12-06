import React, { HTMLAttributes } from 'react';
import { styled } from 'styled-components';
import { ReactComponent as ArrowUp } from '../../assets/icons/arrow-up.svg';
import { ReactComponent as ArrowDown } from '../../assets/icons/arrow-down.svg';
import Badge from '../Badge';
import Button from '../Button';
import RequestInput from './RequestInput';

export interface Props extends HTMLAttributes<HTMLDivElement> {
  item: RequestType;
  index: number;
  lastIndex: number;
  handlePrevRequest?: () => void;
  handleNextRequest?: () => void;
  selectedRange?: string;
  onSubmit?: () => void;
  setParameters: React.Dispatch<React.SetStateAction<string[]>>;
  isFilled?: boolean;
  paramValue?: any;
}

function Request({
  index,
  item,
  lastIndex,
  selectedRange,
  handleNextRequest,
  handlePrevRequest,
  setParameters,
  onSubmit,
  isFilled,
  paramValue,
  ...rest
}: Props) {
  const updateParameters = (value: any) => {
    setParameters((prevParameters) => {
      const changed = [...prevParameters];
      changed[index] = value;
      return changed;
    });
  };

  const labels = {
    range: '사용자 지정 범위',
    cell: '사용자 지정 셀',
    number: '사용자 지정 숫자',
    text: '사용자 지정 텍스트',
    boolean: '사용자 지정 불리언 값',
    criteria: '사용자 지정 조건',
    list: '사용자 지정 리스트',
  };

  return (
    <RequestContainer {...rest}>
      <RequestBlock>
        <QuestionBlock>
          <span>
            {`${index + 1}. `}
            {item.request}
          </span>
          <Badge theme={themeByType[item.type]}>
            {item.type.charAt(0).toUpperCase() + item.type.slice(1)}
          </Badge>
        </QuestionBlock>
        <InputField>
          <span>{labels[item.type]}: </span>
          <RequestInput
            type={item.type}
            selectedRange={selectedRange}
            updateParameters={updateParameters}
            value={paramValue}
          />
        </InputField>
      </RequestBlock>
      {index === lastIndex && (
        <Button
          text={'Submit'}
          icon={'send'}
          color={'primary'}
          style={{ marginRight: '1.5rem', alignSelf: 'flex-end' }}
          onClick={onSubmit}
          disabled={!isFilled}
        />
      )}
      <ArrowButtons>
        <ArrowButton disabled={index === 0} onClick={handlePrevRequest}>
          <ArrowUp />
        </ArrowButton>
        <ArrowButton disabled={index === lastIndex} onClick={handleNextRequest}>
          <ArrowDown />
        </ArrowButton>
      </ArrowButtons>
    </RequestContainer>
  );
}

export default Request;

const themeByType: Record<string, ColorType> = {
  range: 'purple',
  cell: 'blue',
  number: 'magenta',
  text: 'green',
  criteria: 'cyan',
  list: 'orange',
  boolean: 'geekblue',
};

const RequestContainer = styled.div`
  border-radius: 1.25rem;
  background: var(--White, #fff);
  box-shadow: var(--dropdown);
  padding: 2rem;
  display: flex;
  font-size: var(--large);
  font-weight: var(--text);
  align-items: center;
  justify-content: center;
`;

const RequestBlock = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
`;

const QuestionBlock = styled.div`
  display: flex;
  gap: 0.5rem;
  align-items: center;

  div {
    width: fit-content;
  }
`;

const InputField = styled.div`
  margin-left: 0.8rem;
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const ArrowButtons = styled.div`
  flex: 0 0 auto;
  display: flex;
  gap: 0.4rem;
  flex-direction: column;
`;

const ArrowButton = styled.button<{ disabled: boolean }>`
  cursor: pointer;
  border: none;
  box-shadow: none;
  border-radius: 0;
  padding: 0;
  overflow: visible;
  background: inherit;

  svg {
    width: 1.5rem;
    height: 1.5rem;

    [fill] {
      &:not([fill='none']) {
        fill: ${(props) => (props.disabled ? `var(--grey1)` : `var(--primary1)`)};
      }
    }

    [stroke] {
      &:not([stroke='none']) {
        fill: ${(props) => (props.disabled ? `var(--grey1)` : `var(--primary1)`)};
      }
    }
  }
`;
