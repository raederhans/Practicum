import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'

async function readProtocol() {
  try {
    return await readFile(new URL('../USER_STUDY_PROTOCOL.md', import.meta.url), 'utf8')
  } catch (error) {
    if (error?.code === 'ENOENT') return null
    throw error
  }
}

describe('real-user study protocol', () => {
  it('defines an executable small study without presenting planned work as participant evidence', async () => {
    const protocol = await readProtocol()

    expect(protocol).not.toBeNull()
    expect(protocol).toMatch(/## Participant and session criteria/i)
    expect(protocol).toMatch(/## Moderator setup/i)
    expect(protocol).toMatch(/## Tasks and prompts/i)
    expect(protocol).toMatch(/## Success and error measures/i)
    expect(protocol).toMatch(/## Stop and escalation rules/i)
    expect(protocol).toMatch(/## Blank results structure/i)
    expect(protocol).toMatch(/No sessions have been run/i)
  })

  it('tests the analysis-admission boundary directly', async () => {
    const protocol = await readProtocol() ?? ''

    expect(protocol).toMatch(/analysis admission/i)
    expect(protocol).toMatch(/community recovery/i)
    expect(protocol).toMatch(/not assessed/i)
    expect(protocol).toMatch(/observation-ready/i)
  })
})
