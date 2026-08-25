const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("cloakDesktop", {
  getServer: () => ipcRenderer.invoke("cloak-get-server"),
  setServer: (url) => ipcRenderer.send("cloak-set-server", url),
});
