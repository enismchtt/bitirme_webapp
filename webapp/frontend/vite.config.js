import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api requests to the FastAPI backend during dev so the frontend can
// just call relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
