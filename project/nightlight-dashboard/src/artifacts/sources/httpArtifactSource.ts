import { ArtifactError } from '../errors'
import type { ArtifactReadOptions, ArtifactSource } from '../ports'

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === 'AbortError'
    : typeof error === 'object' && error !== null && 'name' in error && error.name === 'AbortError'
}

export class HttpArtifactSource implements ArtifactSource {
  readonly baseUrl: string

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`
  }

  async read(path: string, options: ArtifactReadOptions = {}): Promise<unknown> {
    const normalizedPath = path.replace(/^\/+/, '')
    const requestPath = `${this.baseUrl}${normalizedPath}`
    let response: Response

    try {
      response = await fetch(requestPath, { signal: options.signal })
    } catch (cause) {
      if (isAbortError(cause) || options.signal?.aborted) {
        throw new ArtifactError('aborted', `Artifact request was aborted: ${normalizedPath}`, {
          path: normalizedPath,
          cause,
        })
      }
      throw new ArtifactError('network', `Artifact request failed: ${normalizedPath}`, {
        path: normalizedPath,
        cause,
      })
    }

    if (!response.ok) {
      const code = response.status === 404 ? 'not-found' : 'http'
      throw new ArtifactError(code, `Artifact returned HTTP ${response.status}: ${normalizedPath}`, {
        path: normalizedPath,
        status: response.status,
      })
    }

    let body: string
    try {
      body = await response.text()
    } catch (cause) {
      throw new ArtifactError('network', `Artifact response could not be read: ${normalizedPath}`, {
        path: normalizedPath,
        cause,
      })
    }

    try {
      return JSON.parse(body)
    } catch (cause) {
      throw new ArtifactError('invalid-json', `Artifact is not valid JSON: ${normalizedPath}`, {
        path: normalizedPath,
        cause,
      })
    }
  }
}
