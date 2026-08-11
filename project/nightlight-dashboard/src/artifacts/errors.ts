export type ArtifactErrorCode =
  | 'aborted'
  | 'network'
  | 'not-found'
  | 'http'
  | 'invalid-json'
  | 'invalid-schema'
  | 'unsupported-version'

export class ArtifactError extends Error {
  readonly code: ArtifactErrorCode
  readonly path: string | null
  readonly status: number | null

  constructor(
    code: ArtifactErrorCode,
    message: string,
    options: { path?: string; status?: number; cause?: unknown } = {},
  ) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause })
    this.name = 'ArtifactError'
    this.code = code
    this.path = options.path ?? null
    this.status = options.status ?? null
  }
}

export class ArtifactSchemaError extends ArtifactError {
  constructor(message: string, options: { path?: string; cause?: unknown } = {}) {
    super('invalid-schema', message, options)
    this.name = 'ArtifactSchemaError'
  }
}

export class UnsupportedArtifactVersionError extends ArtifactError {
  readonly version: unknown

  constructor(version: unknown, options: { path?: string } = {}) {
    super('unsupported-version', `Unsupported artifact schema version: ${String(version)}`, options)
    this.name = 'UnsupportedArtifactVersionError'
    this.version = version
  }
}
