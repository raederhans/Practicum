import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const basePath = process.env.VITE_BASE_PATH || '/'

if (!basePath.startsWith('/') || !basePath.endsWith('/')) {
  throw new Error('VITE_BASE_PATH must start and end with a slash.')
}

export default defineConfig({
  base: basePath,
  plugins: [vue()],
  build: {
    modulePreload: { polyfill: false },
    sourcemap: false,
    cssCodeSplit: true,
  },
})
