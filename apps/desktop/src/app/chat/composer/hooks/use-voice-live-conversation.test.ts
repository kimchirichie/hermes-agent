// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { chunkForCommentary, toLiveHistory, type VoiceLiveHandlers } from '@/lib/voice-live'
import { $notifications, clearNotifications } from '@/store/notifications'

import { delegationPrompt, useVoiceLiveConversation } from './use-voice-live-conversation'

// The live-voice toasts must not show machine strings — neither the wire close
// reasons (`connection_lost`, `closed`) nor the raw `DOMException` text a denied
// `getUserMedia` throws (#111987).
//
// The transport is the only seam in the live hook, so the fake session records
// the handlers it registers (tests drive the close/error paths directly) and
// lets `start` reject with exactly what the real `getUserMedia` would throw.
const transport = vi.hoisted(() => ({
  failure: null as unknown,
  handlers: [] as VoiceLiveHandlers[],
  sessions: [] as Array<{ speak: ReturnType<typeof vi.fn>; think: ReturnType<typeof vi.fn> }>
}))

vi.mock('@/lib/voice-live', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    VoiceLiveSession: class {
      close = vi.fn()
      instruct = vi.fn()
      setMuted = vi.fn()
      speak = vi.fn()
      think = vi.fn()

      constructor(handlers: VoiceLiveHandlers) {
        transport.handlers.push(handlers)
        transport.sessions.push(this)
      }

      async start(): Promise<void> {
        if (transport.failure) {
          throw transport.failure
        }
      }
    }
  }
})

const voice = en.notifications.voice

function mountLive() {
  return renderHook(() =>
    useVoiceLiveConversation({
      busy: false,
      consumePendingResponse: vi.fn(),
      enabled: true,
      onSubmit: vi.fn(),
      pendingResponse: () => null,
      seedHistory: () => []
    })
  )
}

/** Mount, start, and hand back the handlers the started session registered. */
async function openSession(): Promise<VoiceLiveHandlers> {
  const hook = mountLive()

  await act(async () => {
    await hook.result.current.start()
  })

  return transport.handlers.at(-1) as VoiceLiveHandlers
}

/** Start a session whose transport fails the way the real one can. */
async function failedStart(failure: unknown): Promise<void> {
  transport.failure = failure
  const hook = mountLive()

  await act(async () => {
    await hook.result.current.start()
  })
}

function resetToasts() {
  clearNotifications()
  transport.failure = null
  transport.handlers.length = 0
  transport.sessions.length = 0
}

describe('Voice-live toast copy', () => {
  beforeEach(resetToasts)
  afterEach(cleanup)

  it('names our own close reasons in copy and passes server-sent reasons through verbatim', async () => {
    let handlers = await openSession()

    act(() => {
      handlers.onClosed('connection_lost', 127)
    })

    expect($notifications.get()[0].title).toBe(voice.liveEnded)
    expect($notifications.get()[0].message).toBe(`${voice.liveEndedConnectionLost} (127s)`)

    cleanup()
    resetToasts()
    handlers = await openSession()

    act(() => {
      handlers.onClosed('quota_exhausted', 12)
    })

    // Server strings are unbounded: no mapping, no redaction claim.
    expect($notifications.get()[0].message).toBe('quota_exhausted (12s)')
  })

  it('maps a getUserMedia DOMException to the recorder mic copy and leaves non-mic failures alone', async () => {
    await failedStart(new DOMException('The request is not allowed by the user agent.', 'NotAllowedError'))

    expect($notifications.get()[0].title).toBe(voice.couldNotStartSession)
    expect($notifications.get()[0].message).toBe(voice.microphonePermissionDenied)

    cleanup()
    resetToasts()
    await failedStart(new Error('Missing local SDP offer'))

    expect($notifications.get()[0].message).toBe('Missing local SDP offer')
  })
})

describe('GPT-Live delegation → Hermes turn', () => {
  it('sends the latest user words as the turn and the exchange as model-only context', () => {
    // The delegation event carries no text: both are reconstructed from
    // transcript deltas, fragments of one speaker concatenated as received.
    const { context, prompt } = delegationPrompt([
      { endMs: 1000, speaker: 'assistant', startMs: 0, text: 'Hi, how ' },
      { endMs: 1500, speaker: 'assistant', startMs: 1000, text: 'can I help?' },
      { endMs: 2500, speaker: 'user', startMs: 1500, text: 'What is ' },
      { endMs: 3200, speaker: 'user', startMs: 2500, text: 'the weather in Paris?' }
    ])

    expect(prompt).toBe('What is the weather in Paris?')
    expect(context).toContain('Voice assistant: Hi, how can I help?')
    expect(context).toContain('User: What is the weather in Paris?')
  })

  it('splits a long reply into vendor-sized commentary appends on sentence boundaries', () => {
    const sentence = 'This is a sentence about the result. '
    const chunks = chunkForCommentary(sentence.repeat(80), 400)

    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks.every(chunk => chunk.length <= 400)).toBe(true)
    expect(chunks.every(chunk => chunk.endsWith('.'))).toBe(true)
    expect(chunks.join(' ')).toBe(sentence.repeat(80).trim())
  })

  it('seeds the live session with the most recent text turns within budget', () => {
    const turns = Array.from({ length: 40 }, (_, index) => ({
      role: (index % 2 === 0 ? 'user' : 'assistant') as 'assistant' | 'user',
      text: `turn ${index}`
    }))

    const history = toLiveHistory(turns, 6)

    expect(history).toHaveLength(6)
    expect(history.at(-1)?.content[0]?.text).toBe('turn 39')
    expect(history[0]?.role).toBe('user')
    expect(history.find(m => m.role === 'assistant')?.content[0]?.type).toBe('output_text')
  })
})

// A turn that ends with nothing to say (the delegation 404s, the turn is
// interrupted, every tool call fails) used to be reported with `think`, which
// the live model never reads aloud — so it filled the silence itself and told
// the user the request had succeeded (MS journey 1285, 2026-11-02). The failure
// has to be spoken, and only when the turn did not already speak.
describe('GPT-Live turn settling without a reply', () => {
  const SETTLE_MS = 16_000

  beforeEach(() => {
    resetToasts()
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    resetToasts()
  })

  async function openLiveSession(busy = false) {
    const hook = renderHook(
      ({ isBusy }: { isBusy: boolean }) =>
        useVoiceLiveConversation({
          busy: isBusy,
          consumePendingResponse: vi.fn(),
          enabled: true,
          onSubmit: vi.fn(),
          pendingResponse: () => null,
          seedHistory: () => []
        }),
      { initialProps: { isBusy: busy } }
    )

    await act(async () => {
      await hook.result.current.start()
    })

    return {
      handlers: transport.handlers.at(-1) as VoiceLiveHandlers,
      hook,
      session: transport.sessions.at(-1) as { speak: ReturnType<typeof vi.fn>; think: ReturnType<typeof vi.fn> }
    }
  }

  it('speaks the failure when a delegated turn settles with nothing spoken', async () => {
    const { handlers, session } = await openLiveSession()

    act(() => {
      handlers.onDelegation('del-404', [])
    })

    await act(async () => {
      vi.advanceTimersByTime(SETTLE_MS)
    })

    expect(session.speak).toHaveBeenCalledWith('del-404', 'That did not go through.')
    // The old wording only ever reached the model as silent context.
    expect(session.think).not.toHaveBeenCalledWith('del-404', expect.stringContaining('spoken result'))
  })

  it('does not report a turn that already spoke its reply', async () => {
    let response: null | { id: string; pending: boolean; text: string } = null

    const hook = renderHook(() =>
      useVoiceLiveConversation({
        busy: false,
        consumePendingResponse: () => {
          response = null
        },
        enabled: true,
        onSubmit: vi.fn(),
        pendingResponse: () => response,
        seedHistory: () => []
      })
    )

    await act(async () => {
      await hook.result.current.start()
    })

    const handlers = transport.handlers.at(-1) as VoiceLiveHandlers
    const session = transport.sessions.at(-1) as { speak: ReturnType<typeof vi.fn> }

    act(() => {
      handlers.onDelegation('del-ok', [])
    })

    response = { id: 'reply-1', pending: false, text: 'The update is in.' }

    await act(async () => {
      vi.advanceTimersByTime(SETTLE_MS)
    })

    expect(session.speak).toHaveBeenCalledWith('del-ok', 'The update is in.')
    expect(session.speak).not.toHaveBeenCalledWith('del-ok', 'That did not go through.')
  })

  it('speaks a superseded notice for the turn a newer delegation interrupted', async () => {
    const { handlers, hook, session } = await openLiveSession()

    act(() => {
      handlers.onDelegation('del-first', [])
    })

    await act(async () => {
      hook.rerender({ isBusy: true })
    })

    act(() => {
      handlers.onDelegation('del-second', [])
    })

    expect(session.speak).toHaveBeenCalledWith(
      'del-first',
      'That request was replaced by your newer one, so it did not go through.'
    )
  })
})
