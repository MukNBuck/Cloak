const { app, BrowserWindow, Menu, dialog, ipcMain, nativeTheme } = require("electron");
const fs = require("fs");
const path = require("path");

const SETTINGS_PATH = () => path.join(app.getPath("userData"), "settings.json");
const DEFAULT_SERVER = "http://127.0.0.1:8001";

let mainWindow = null;

function loadSettings() {
  try {
    return JSON.parse(fs.readFileSync(SETTINGS_PATH(), "utf8"));
  } catch {
    return {};
  }
}

function saveSettings(patch) {
  const current = loadSettings();
  const next = { ...current, ...patch };
  fs.mkdirSync(path.dirname(SETTINGS_PATH()), { recursive: true });
  fs.writeFileSync(SETTINGS_PATH(), JSON.stringify(next, null, 2));
  return next;
}

function serverUrl() {
  const fromEnv = (process.env.CLOAK_SERVER_URL || "").trim().replace(/\/$/, "");
  if (fromEnv) return fromEnv;
  const saved = (loadSettings().serverUrl || "").trim();
  return saved || DEFAULT_SERVER;
}

async function serverIsUp(url) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(`${url.replace(/\/$/, "")}/health`, {
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

function createWindow() {
  nativeTheme.themeSource = "dark";
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 860,
    minWidth: 1100,
    minHeight: 680,
    backgroundColor: "#0e0e0e",
    autoHideMenuBar: true,
    title: "Cloak",
    icon: path.join(__dirname, "icon.png"),
    titleBarStyle: "hidden",
    titleBarOverlay: {
      color: "#171717",
      symbolColor: "#c4c7c5",
      height: 36,
    },
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const menu = Menu.buildFromTemplate([
    {
      label: "Cloak",
      submenu: [
        {
          label: "Change server…",
          click: () => promptServer(mainWindow),
        },
        { type: "separator" },
        { role: "reload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "quit" },
      ],
    },
  ]);
  Menu.setApplicationMenu(menu);

  loadApp(mainWindow);
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  return mainWindow;
}

async function loadApp(win) {
  const url = serverUrl();
  const ok = await serverIsUp(url);
  if (ok) {
    win.loadURL(url);
    return;
  }
  win.loadFile(path.join(__dirname, "connect.html"), {
    query: { server: url },
  });
}

async function promptServer(win) {
  const current = serverUrl();
  const result = await dialog.showMessageBox(win, {
    type: "question",
    buttons: ["Use current", "Enter new URL"],
    defaultId: 1,
    cancelId: 0,
    title: "Cloak server",
    message: `Current server:\n${current}`,
    detail: "The desktop window loads the Gemini-style UI from the FastAPI backend.",
  });
  if (result.response !== 1) return;
  const entered = await promptUrl(win, current);
  if (!entered) return;
  saveSettings({ serverUrl: entered });
  loadApp(win);
}

function promptUrl(win, current) {
  return new Promise((resolve) => {
    const child = new BrowserWindow({
      parent: win,
      modal: true,
      width: 480,
      height: 220,
      backgroundColor: "#0e0e0e",
      autoHideMenuBar: true,
      webPreferences: {
        preload: path.join(__dirname, "preload.js"),
        contextIsolation: true,
      },
    });
    child.loadFile(path.join(__dirname, "connect.html"), {
      query: { server: current, modal: "1" },
    });
    const finish = (url) => {
      ipcMain.removeListener("cloak-set-server", onSet);
      if (!child.isDestroyed()) child.close();
      resolve(url);
    };
    const onSet = (_event, url) => finish(url);
    ipcMain.once("cloak-set-server", onSet);
    child.on("closed", () => finish(null));
  });
}

ipcMain.handle("cloak-get-server", () => serverUrl());
ipcMain.on("cloak-set-server", (_event, url) => {
  const clean = String(url || "").trim().replace(/\/$/, "");
  if (!clean) return;
  saveSettings({ serverUrl: clean });
  if (mainWindow) loadApp(mainWindow);
});

app.whenReady().then(() => {
  const ses = require("electron").session.defaultSession;
  const allowMic = (permission) =>
    permission === "media" ||
    permission === "audioCapture" ||
    permission === "microphone";
  ses.setPermissionRequestHandler((_webContents, permission, callback) => {
    callback(allowMic(permission) || permission === "clipboard-sanitized-write");
  });
  ses.setPermissionCheckHandler((_webContents, permission) => allowMic(permission));
  if (typeof ses.setDevicePermissionHandler === "function") {
    ses.setDevicePermissionHandler((details) => details.deviceType === "audio" || details.deviceType === "microphone");
  }
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
