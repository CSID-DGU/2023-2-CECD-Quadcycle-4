import React, { HTMLAttributes } from 'react';
import styled from 'styled-components';
import Alert from '../Alert';
import { guideMessage } from '../../utils/common';
import { useRecoilValue } from 'recoil';
import { FileLoadedState } from '../../store/spread';

export interface Props extends HTMLAttributes<HTMLDivElement> {
  paramType?: ParameterType;
}

function Guides({ paramType, ...rest }: Props) {
  const fileLoadedStatus = useRecoilValue(FileLoadedState);

  return (
    <GuidesPanel {...rest}>
      <PanelTitle>Guides</PanelTitle>
      <AlertContainer>
        {fileLoadedStatus === 'preview' ? (
          <>
            {guideMessage[fileLoadedStatus]?.map((item: GuideInfoType) => (
              <Alert
                key={item.title}
                title={item.title}
                content={item.description}
                theme={item.color}
              ></Alert>
            ))}
            {paramType &&
              guideMessage[paramType]?.map((item: GuideInfoType) => (
                <Alert
                  key={item.title}
                  title={item.title}
                  content={item.description}
                  theme={item.color}
                ></Alert>
              ))}
          </>
        ) : (
          guideMessage[fileLoadedStatus]?.map((item: GuideInfoType) => (
            <Alert
              key={item.title}
              title={item.title}
              content={item.description}
              theme={item.color}
            ></Alert>
          ))
        )}
      </AlertContainer>
    </GuidesPanel>
  );
}

export default Guides;

const GuidesPanel = styled.aside`
  width: 20rem;
  height: 100%;
  float: right;
  border-radius: 0rem 3.125rem 3.125rem 0rem;
  background: var(--white);
  padding: 2rem;
  padding-top: 6rem;
  flex: 0 0 auto;
`;

const PanelTitle = styled.h3`
  font-weight: var(--extraBold);
  font-size: var(--title);
  text-align: left;
  margin-bottom: 1.3rem;
`;

const AlertContainer = styled.div``;
