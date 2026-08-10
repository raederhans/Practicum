import { randomBytes } from 'node:crypto'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const basePath = process.env.VITE_BASE_PATH || '/'

if (!basePath.startsWith('/') || !basePath.endsWith('/')) {
  throw new Error('VITE_BASE_PATH must start and end with a slash.')
}

export function applyDevelopmentCsp(html, nonce) {
  const productionStylePolicy = "style-src 'self'"
  const productionConnectionPolicy = "connect-src 'none'"

  if (
    html.split(productionStylePolicy).length !== 2
    || html.split(productionConnectionPolicy).length !== 2
  ) {
    throw new Error('The development CSP transform could not find the production policy.')
  }

  return html
    .replace(productionStylePolicy, `${productionStylePolicy} 'nonce-${nonce}'`)
    .replace(productionConnectionPolicy, "connect-src 'self' ws:")
}

export default defineConfig(({ command, isPreview }) => {
  const isDevelopmentServer = command === 'serve' && !isPreview
  const developmentNonce = isDevelopmentServer
    ? randomBytes(18).toString('base64')
    : null

  return {
    base: basePath,
    html: isDevelopmentServer ? { cspNonce: developmentNonce } : undefined,
    plugins: [
      vue(),
      isDevelopmentServer && {
        name: 'nightlight-development-csp',
        transformIndexHtml(html) {
          return applyDevelopmentCsp(html, developmentNonce)
        },
      },
    ],
    build: {
      modulePreload: { polyfill: false },
      sourcemap: false,
      cssCodeSplit: true,
    },
  }
})
