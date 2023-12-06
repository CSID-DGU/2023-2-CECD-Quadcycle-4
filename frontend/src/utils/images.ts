import { ReactComponent as Info } from '../assets/icons/info.svg';
import { ReactComponent as Error } from '../assets/icons/error.svg';
import { ReactComponent as Warning } from '../assets/icons/warning.svg';
import { ReactComponent as Success } from '../assets/icons/success.svg';
import { ReactComponent as ArrowDown } from '../assets/icons/arrow-down.svg';
import { ReactComponent as ArrowUp } from '../assets/icons/arrow-up.svg';
import { ReactComponent as Chat } from '../assets/icons/chat-question.svg';
import { ReactComponent as Download } from '../assets/icons/download.svg';
import { ReactComponent as Github } from '../assets/icons/github.svg';
import { ReactComponent as Home } from '../assets/icons/home.svg';
import { ReactComponent as Notice } from '../assets/icons/notice.svg';
import { ReactComponent as Rollback } from '../assets/icons/rollback.svg';
import { ReactComponent as Send } from '../assets/icons/send.svg';
import { ReactComponent as Spinner } from '../assets/icons/spinner.svg';
import { ReactComponent as Upload } from '../assets/icons/upload.svg';
import { ReactComponent as Continue } from '../assets/icons/continue.svg';
import { ReactComponent as Clip } from '../assets/icons/clip.svg';

export const images: Record<string, React.FC<React.SVGProps<SVGSVGElement>>> = {
  info: Info,
  error: Error,
  warning: Warning,
  success: Success,
  arrowdown: ArrowDown,
  arrowup: ArrowUp,
  download: Download,
  chat: Chat,
  github: Github,
  notice: Notice,
  home: Home,
  send: Send,
  rollback: Rollback,
  spinner: Spinner,
  upload: Upload,
  continue: Continue,
  clip: Clip,
};
