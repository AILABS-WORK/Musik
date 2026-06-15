import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Fixed port so the Tauri shell and tests can rely on it.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 5173, strictPort: true },
})
