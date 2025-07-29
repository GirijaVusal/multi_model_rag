import React, { useEffect, useState } from 'react';
import { getAllSessions, loadSessionMessages, createNewSession, addToSession } from '../db';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import toast from 'react-hot-toast';

const ChatApp = () => {
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState('');
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [tempSessionId, setTempSessionId] = useState('');


  const [showModal, setShowModal] = useState(false);
const [pdfUrl, setPdfUrl] = useState("");
const [isCreatingKB, setIsCreatingKB] = useState(false);


const handleKB = () => {
  setShowModal(true);
};

  // Load existing sessions and messages
  useEffect(() => {
    (async () => {
      const all = await getAllSessions();
      setSessions(all.sort((a, b) => b.createdAt - a.createdAt));

      if (all.length > 0) {
        setCurrentSessionId(all[0].sessionId);
        const sessionMessages = await loadSessionMessages(all[0].sessionId);
        setMessages(sessionMessages);
      } else {
        const newId = uuidv4();
        setTempSessionId(newId);
        setCurrentSessionId(newId);
        setMessages([]);
      }
    })();
  }, []);

  const handleNewChat = async () => {
    const newSessionId = uuidv4();
    setTempSessionId(newSessionId);
    setCurrentSessionId(newSessionId);
    setMessages([]);
  };
 

 const handleSend = async () => {
  const userText = userInput.trim();
  if (!userText) return;

  setUserInput('');

  // Optimistically add user message
  const updatedMessages = [...messages, { human: userText }];
  setMessages(updatedMessages);

  // Prepare last 2 user messages for context
  const history = updatedMessages.slice(-3).map(m => ({
    role: 'user',
    content: m.human || ''
  }));

  // const payload = {
  //   model: 'llama-3.3-70b-versatile',
  //   messages: [...history, { role: 'user', content: userText }]
  // };


  const payload = {
    message: userText,
    collection_name: "transformer"
  };


  try {
    const response = await axios.post(
      'http://127.0.0.1:8000/api/chat/ask/',
      payload,
      {
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );

    // const aiReply = response.data.choices[0].message.content;
    const aiReply = response.data.bot || "No response from the bot.";


    // Append AI reply to last message
    const fullMessages = [...updatedMessages.slice(0, -1), { human: userText, ai: aiReply }];
    setMessages(fullMessages);

    // Save to DB only if AI message exists
    const idToUse = tempSessionId || currentSessionId;
    if (tempSessionId) {
      await createNewSession(tempSessionId);
      setCurrentSessionId(tempSessionId);
      setTempSessionId('');
    }

    await addToSession(idToUse, userText, aiReply);

    const updatedSessions = await getAllSessions();
    setSessions(updatedSessions.sort((a, b) => b.createdAt - a.createdAt));

  } catch (err) {
    console.error('Error communicating with API:', err);
  }
};


  const handleSessionSelect = async (sessionId) => {
    setCurrentSessionId(sessionId);
    setTempSessionId('');
    const sessionMessages = await loadSessionMessages(sessionId);
    setMessages(sessionMessages);
  };



  const handleCreateKB = async () => {
  if (!pdfUrl.trim()) return;

  setIsCreatingKB(true);

  try {

    const payload = {
      file_url: pdfUrl,
      images_background_context: "",
      collection_name: "transformer",
    };

    const response = await axios.post('http://127.0.0.1:8000/api/store_embedding/', payload);


    console.log("Knowledge base created:", response.message);
    // Optionally show a toast or update state

    const message = response.data.message || "Knowledge base created successfully.";

    toast.success(message);  

    setShowModal(false);
    setPdfUrl("");
  } catch (err) {
    console.error("Error creating knowledge base:", err);
    // Optionally show error message
  } finally {
    setIsCreatingKB(false);
  }
};


  return (
    <div className="flex h-screen bg-slate-900 text-slate-100">
      {/* Sidebar */}
      <aside className="w-64 border-r border-slate-700 p-4 overflow-y-auto">
       <button 
          onClick={handleKB}
          className="w-full mb-4 py-2 px-4 bg-sky-600 hover:bg-sky-500 rounded-lg text-white font-semibold transition"
        >
        Create Knowledge Base
      </button>

        <h2 className="font-bold mb-4 text-slate-200">Previous Sessions</h2>
        <button 
          onClick={handleNewChat}
          className="w-full mb-4 py-2 px-4 bg-sky-600 hover:bg-sky-500 rounded-lg text-white font-semibold transition"
        >
          + New Chat
        </button>
        {sessions.map((s) => (
          <div
            key={s.sessionId}
            onClick={() => handleSessionSelect(s.sessionId)}
            className={`p-2 mb-2 rounded-lg cursor-pointer transition truncate
              ${s.sessionId === currentSessionId
                ? 'bg-sky-700 text-white'
                : 'hover:bg-slate-700'
              }`}
          >
            {(s.messages[0]?.ai?.split(' ').slice(0, 3).join(' ') || 'New Session') + '...'}
          </div>
        ))}
      </aside>

      {/* Chat Area */}
      <main className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-5">
          <div className="max-w-3xl w-full mx-auto space-y-5">
            {messages.map((msg, idx) => (
              <div key={idx} className="flex flex-col gap-y-1">
                <div className="self-end max-w-[85%]">
                  <div className="bg-sky-600 text-white rounded-2xl px-4 py-2.5 shadow-md">
                    {msg.human}
                  </div>
                </div>
                <div className="self-start max-w-[85%]">
                  <div className="bg-slate-700 text-slate-100 rounded-2xl px-4 py-2.5 shadow-md">
                    {msg.ai}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Fixed Input */}
        <div className="sticky bottom-0 bg-slate-900/80 backdrop-blur-sm px-4 py-3">
          <div className="max-w-3xl w-full mx-auto relative flex items-center">
            <input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Message..."
              className="w-full bg-slate-800 border border-slate-600 rounded-2xl pl-4 pr-12 py-3 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 transition"
            />
            <button
              onClick={handleSend}
            >
            </button>
          </div>
        </div>
      </main>

      {/* Modal Popup */}
      {showModal && (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-slate-800 text-white p-6 rounded-lg shadow-lg w-full max-w-md relative">
          <h3 className="text-xl font-bold mb-4">Create Knowledge Base</h3>
          <input
            type="text"
            value={pdfUrl}
            onChange={(e) => setPdfUrl(e.target.value)}
            placeholder="Enter PDF URL"
            className="w-full mb-4 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
          />
          <div className="flex justify-end space-x-3">
            <button
              className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded"
              onClick={() => setShowModal(false)}
              disabled={isCreatingKB}
            >
              Cancel
            </button>
            <button
              className="px-4 py-2 bg-sky-600 hover:bg-sky-500 rounded"
              onClick={handleCreateKB}
              disabled={isCreatingKB}
            >
              {isCreatingKB ? 'Creating...' : 'Create'}
            </button>
          </div>
        </div>
      </div>
    )}


    </div>
  );
};

export default ChatApp;



