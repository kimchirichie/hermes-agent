import { afterEach, describe, expect, it, vi } from 'vitest'

import { type VoiceLiveHandlers, VoiceLiveSession } from './voice-live'

// The two appends are NOT the same channel to the user: the vendor speaks
// `session.commentary.append` (Hermes' answer) and never reads
// `session.thinking.append` (quiet progress). So a turn that must tell the user
// something — "that did not go through" — has to ride the commentary append;
// sending it as thinking leaves the speaker silent and lets the voice model
// invent an outcome (MS journey 1285, 2026-11-02).

const handlers: VoiceLiveHandlers = {
  onClosed: () => undefined,
  onDelegation: () => undefined,
  onError: () => undefined
}

function installApi() {
  const api = vi.fn(async (_request: unknown) => ({
    ok: true,
    session: { id: 's1' },
    transport: { sdp: 'answer', type: 'answer' }
  }))

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

  return api
}

/** Enough of RTCPeerConnection/getUserMedia for `start()` to open the data channel. */
function installWebRTC(): Array<Record<string, unknown>> {
  const sent: Array<Record<string, unknown>> = []

  const channel = {
    addEventListener: () => undefined,
    readyState: 'open' as const,
    send: (data: string) => sent.push(JSON.parse(data) as Record<string, unknown>)
  }

  const PeerConnection = class {
    addEventListener = () => undefined
    connectionState = 'new'
    createDataChannel = () => channel
    createOffer = async () => ({})
    iceGatheringState = 'complete'
    localDescription = { sdp: 'v=0\r\n' }
    setLocalDescription = async () => undefined
    setRemoteDescription = async () => undefined
  }

  Object.defineProperty(globalThis, 'RTCPeerConnection', { configurable: true, value: PeerConnection })
  Object.defineProperty(window.navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => ({ getAudioTracks: () => [] }) }
  })

  return sent
}

afterEach(() => {
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(globalThis, 'RTCPeerConnection')
  Reflect.deleteProperty(window.navigator, 'mediaDevices')
})

describe('GPT-Live append wire types', () => {
  it('sends a spoken failure as commentary and keeps think off the speaker', async () => {
    installApi()
    const sent = installWebRTC()

    const session = new VoiceLiveSession(handlers)
    await session.start([])

    session.speak('del-404', 'That did not go through.')
    session.think('del-404', 'Hermes is working: search_journeys. Not done yet.')

    expect(sent.map(frame => frame.type)).toEqual(['session.commentary.append', 'session.thinking.append'])
    expect(sent[0]).toMatchObject({
      content: 'That did not go through.',
      delegation_id: 'del-404',
      type: 'session.commentary.append'
    })
  })
})
