import { openDB } from 'idb';

const DB_NAME = 'chat-db';
const STORE_NAME = 'sessions';

export async function getDB() {
  return openDB(DB_NAME, 2, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'sessionId' });
        store.createIndex('createdAt', 'createdAt');
      }
    },
  });
}

export async function createNewSession(sessionId) {
  const db = await getDB();
  await db.put(STORE_NAME, {
    sessionId,
    createdAt: Date.now(),
    messages: [],
  });
}

export async function getSession(sessionId) {
  const db = await getDB();
  return await db.get(STORE_NAME, sessionId);
}

// export async function addToSession(sessionId, humanText, aiText) {
// //   if (!aiText || aiText.trim() === '') return; // skip if no AI response
//   if (!aiText || aiText.trim() === '') return; // Correct early return

//     const db = await getDB();
//     const session = await db.get(STORE_NAME, sessionId);
//     if (session) {
//         session.messages.push({ human: humanText, ai: aiText });
//         await db.put(STORE_NAME, session);
//     }
//     }

export async function addToSession(sessionId, humanText, aiText) {
  console.log('🧪 addToSession input:', { sessionId, humanText, aiText });

  if (!aiText || aiText.trim() === '') {
    console.log('🚫 Skipping empty AI response');
    return;
  }

  const db = await getDB();
  const session = await db.get(STORE_NAME, sessionId);

  if (session) {
    session.messages.push({ human: humanText, ai: aiText });
    await db.put(STORE_NAME, session);
  }
}





export async function getAllSessions() {
  const db = await getDB();
  return await db.getAll(STORE_NAME); // Returns [{ sessionId, createdAt, messages: [...] }, ...]
}

// export async function getAllSessions() {
//   const db = await getDB();
//   const allSessions = await db.getAll(STORE_NAME);

//   for (const session of allSessions) {
//     if (!session.messages || session.messages.length < 2) {
//       await db.delete(STORE_NAME, session.sessionId);
//     }
//   }

//   // Return only valid sessions (length >= 2)
//   return allSessions.filter(s => s.messages && s.messages.length >= 2);
// }


export async function loadSessionMessages(sessionId) {
  const db = await getDB();
  const session = await db.get(STORE_NAME, sessionId);
  return session?.messages || [];
}
