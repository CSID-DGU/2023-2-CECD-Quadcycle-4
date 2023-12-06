import React from 'react';
import styled from 'styled-components';
import Icon from '../Icon';
import { GITHUB_URL } from '../../utils/common';

function Nav() {
  const menus = [
    { name: 'home', clickTrigger: () => window.location.reload() },
    { name: 'chat', clickTrigger: () => window.open(`${GITHUB_URL}/issues`, '_blank') },
    {
      name: 'github',
      clickTrigger: () => window.open(GITHUB_URL, '_blank'),
    },
    { name: 'rollback', clickTrigger: () => window.location.reload() },
  ];

  return (
    <Wrapper>
      <NavBar>
        <ul>
          {menus.map((menu) => (
            <li key={menu.name} onClick={menu.clickTrigger}>
              <Icon name={menu.name} size={28} />
            </li>
          ))}
        </ul>
      </NavBar>
    </Wrapper>
  );
}

export default Nav;

const Wrapper = styled.section`
  height: 100%;
  padding: 2.4rem 3rem;
  flex: 0 0 auto;
`;

const NavBar = styled.nav`
  width: 6rem;
  height: 100%;
  border-radius: 2.5rem;
  background: var(--primary1, #2e3192);
  padding: 7rem 2rem 2.5rem;

  ul {
    list-style: none;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 1rem;
  }

  li {
    cursor: pointer;
    width: 3rem;
    height: 3rem;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  li:hover {
    border-radius: 0.625rem;
    background: rgba(255, 255, 255, 0.4);
  }

  li:last-child {
    margin-top: auto;
  }
`;
