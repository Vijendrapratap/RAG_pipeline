# App icon

Place a Windows icon here as `icon.ico` (multi-resolution: 16, 32, 48, 256 px).
electron-builder uses it for the window, taskbar, installer, and tray.

Until you add one, the app still runs — the tray falls back to an empty image
and the window uses the default Electron icon.

Quick way to make one: take the saffron "वि" brand mark (or a 256×256 PNG logo)
and convert it to `.ico` (e.g. https://icoconvert.com or ImageMagick:
`magick logo-256.png -define icon:auto-resize=256,48,32,16 icon.ico`).
