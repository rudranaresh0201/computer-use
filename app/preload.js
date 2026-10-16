// The only doors the page gets: send a request to the agent, and hear its
// notifications. No Node, no filesystem, no child processes in the page.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('agent', {
  request: (method, params) => ipcRenderer.invoke('agent:request', method, params),
  onEvent: (callback) => ipcRenderer.on('agent:event', (_event, msg) => callback(msg)),
});
