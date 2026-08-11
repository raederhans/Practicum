export interface ArtifactReadOptions {
  signal?: AbortSignal
}

export interface ArtifactSource {
  read(path: string, options?: ArtifactReadOptions): Promise<unknown>
}
