import React, { ChangeEvent, useState, InputHTMLAttributes } from 'react';
import styled from 'styled-components';

export interface Props extends InputHTMLAttributes<HTMLInputElement> {
  updateParameters: (value: any) => void;
}

function Toggle({ disabled = false, defaultChecked = true, updateParameters, ...rest }: Props) {
  const [check, setCheck] = useState<boolean>(defaultChecked);

  const onChangeToggle = (e: ChangeEvent<HTMLInputElement>) => {
    setCheck(e.target.checked);
    updateParameters(e.target.checked);
  };
  return (
    <ToggleSwitch check={check}>
      <input
        type='checkbox'
        defaultChecked={defaultChecked}
        onChange={onChangeToggle}
        disabled={disabled}
        {...rest}
      />
      <span>{check ? 'TRUE' : 'FALSE'}</span>
    </ToggleSwitch>
  );
}

export default Toggle;

interface ToggleSwitchType {
  check: boolean;
}

const ToggleSwitch = styled.label<ToggleSwitchType>`
  --toggle-size: 80px;
  --toggle-circle-size: 30px;

  position: relative;
  display: inline-block;
  width: var(--toggle-size);
  height: calc(var(--toggle-size) / 2.5);

  input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  span {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--white);
    -webkit-transition: 0.4s;
    transition: 0.4s;
    border-radius: 34px;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.25);
    color: ${({ check }) => (check ? 'white' : 'var(--primary)')};
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    justify-content: ${({ check }) => (check ? 'flex-start' : 'flex-end')};
    padding-left: 10px;
    padding-right: 10px;
  }

  span:before {
    position: absolute;
    content: '';
    height: var(--toggle-circle-size);
    width: var(--toggle-circle-size);
    left: 1px;
    bottom: 1px;
    background-color: white;
    -webkit-transition: 0.4s;
    transition: 0.4s;
    border-radius: 50%;
    background-color: var(--primary1);
  }

  input:checked + span {
    background-color: var(--primary1);
  }

  input:checked + span:before {
    -webkit-transform: translateX(calc(var(--toggle-size) - var(--toggle-circle-size)));
    -ms-transform: translateX(calc(var(--toggle-size) - var(--toggle-circle-size)));
    transform: translateX(calc(var(--toggle-size) - var(--toggle-circle-size)));
    background-color: var(--white);
  }

  input:disabled + span {
    background-color: var(--lightgrey);
    cursor: not-allowed;
  }

  input:disabled + span:before {
    background-color: var(--grey);
  }
`;
