import { Toaster } from 'react-hot-toast';

import ChatApp from './components/ChatApp';

function App() {
  return(
  <>
  <ChatApp />
  <Toaster position="top-right" reverseOrder={false} />

  </> );
}

export default App;
