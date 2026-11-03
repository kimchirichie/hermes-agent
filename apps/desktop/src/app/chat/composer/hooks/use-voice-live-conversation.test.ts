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

  async function openLiveSession(
    busy = false,
    overrides: {
      consumePendingResponse?: () => void
      onSteer?: (text: string) => Promise<'queued' | 'steered' | false> | 'queued' | 'steered' | false
      onStopWord?: () => void
      onSubmit?: (text: string, voiceContext: string) => Promise<void> | void
      pendingResponse?: () => null | { id: string; pending: boolean; text: string }
    } = {}
  ) {
    const hook = renderHook(
      ({ isBusy }: { isBusy: boolean }) =>
        useVoiceLiveConversation({
          busy: isBusy,
          consumePendingResponse: overrides.consumePendingResponse ?? vi.fn(),
          enabled: true,
          onSteer: overrides.onSteer,
          onStopWord: overrides.onStopWord,
          onSubmit: overrides.onSubmit ?? vi.fn(),
          pendingResponse: overrides.pendingResponse ?? (() => null),
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

  it('steers the running turn when the user speaks again mid-turn — it never interrupts it', async () => {
    const onSteer = vi.fn(async () => 'steered' as const)
    const onSubmit = vi.fn()
    const { handlers, hook, session } = await openLiveSession(false, { onSteer, onSubmit })

    act(() => {
      handlers.onDelegation('del-first', [{ endMs: 1_000, speaker: 'user', startMs: 0, text: 'start the import' }])
    })

    await act(async () => {
      hook.rerender({ isBusy: true })
    })

    act(() => {
      handlers.onDelegation('del-second', [
        { endMs: 2_000, speaker: 'user', startMs: 1_000, text: 'start the import' },
        { endMs: 2_500, speaker: 'assistant', startMs: 2_000, text: 'On it.' },
        { endMs: 3_000, speaker: 'user', startMs: 2_500, text: 'and skip the archive step' }
      ])
    })

    await act(async () => {
      await Promise.resolve()
    })

    expect(onSteer).toHaveBeenCalledTimes(1)
    expect(onSteer).toHaveBeenCalledWith('and skip the archive step')
    // The correction reached the running turn: no second Hermes turn, and the
    // old turn is not declared dead ("that request was replaced…" is gone).
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(session.speak).not.toHaveBeenCalled()
  })

  it('reports the steered turn under the delegation the live model is waiting on', async () => {
    let response: null | { id: string; pending: boolean; text: string } = null

    const onSteer = vi.fn(async () => 'steered' as const)

    const { handlers, hook, session } = await openLiveSession(false, {
      consumePendingResponse: () => {
        response = null
      },
      onSteer,
      pendingResponse: () => response
    })

    act(() => {
      handlers.onDelegation('del-first', [{ endMs: 1_000, speaker: 'user', startMs: 0, text: 'start the import' }])
    })

    await act(async () => {
      hook.rerender({ isBusy: true })
    })

    act(() => {
      handlers.onDelegation('del-second', [
        { endMs: 1_000, speaker: 'user', startMs: 0, text: 'start the import' },
        { endMs: 1_500, speaker: 'assistant', startMs: 1_000, text: 'On it.' },
        { endMs: 2_000, speaker: 'user', startMs: 1_500, text: 'and skip the archive step' }
      ])
    })

    await act(async () => {
      await Promise.resolve()
    })

    response = { id: 'reply-steered', pending: false, text: 'Imported, and the archive step was skipped.' }

    await act(async () => {
      // The steered turn settles, then its answer lands.
      hook.rerender({ isBusy: false })
    })

    await act(async () => {
      vi.advanceTimersByTime(1_000)
    })

    // The answer is for what the user asked LAST, so it rides the newest
    // delegation — the one the live model is awaiting.
    expect(session.speak).toHaveBeenCalledWith('del-second', 'Imported, and the archive step was skipped.')
    expect(session.speak).not.toHaveBeenCalledWith('del-first', expect.anything())
  })

  it('keeps reading from where it stopped after a steer — the running turn is not repeated', async () => {
    // A steer seals the live bubble and the answer continues in a new one, so
    // the collected turn text only GROWS (same first bubble id). Restarting the
    // spoken cursor on the steer would say everything already spoken again.
    let response: null | { id: string; pending: boolean; text: string } = null

    const onSteer = vi.fn(async () => 'steered' as const)

    const { handlers, hook, session } = await openLiveSession(false, {
      onSteer,
      pendingResponse: () => response
    })

    act(() => {
      handlers.onDelegation('del-first', [{ endMs: 1_000, speaker: 'user', startMs: 0, text: 'start the import' }])
    })

    await act(async () => {
      hook.rerender({ isBusy: true })
    })

    response = { id: 'bubble-1', pending: true, text: 'Reading the export now. Checking rows.' }

    await act(async () => {
      vi.advanceTimersByTime(1_000)
    })

    expect(session.speak).toHaveBeenCalledWith('del-first', 'Reading the export now.')

    act(() => {
      handlers.onDelegation('del-second', [
        { endMs: 1_000, speaker: 'user', startMs: 0, text: 'start the import' },
        { endMs: 1_500, speaker: 'assistant', startMs: 1_000, text: 'On it.' },
        { endMs: 2_000, speaker: 'user', startMs: 1_500, text: 'and skip the archive step' }
      ])
    })

    await act(async () => {
      await Promise.resolve()
    })

    response = {
      id: 'bubble-1',
      pending: true,
      text: 'Reading the export now. Checking rows.\n\nThe archive step is skipped. Next up.'
    }

    await act(async () => {
      vi.advanceTimersByTime(1_000)
    })

    const spokenAfterSteer = session.speak.mock.calls
      .filter(([delegation]) => delegation === 'del-second')
      .map(([, text]) => text)
      .join(' ')

    expect(spokenAfterSteer).toContain('archive step is skipped')
    expect(spokenAfterSteer).not.toContain('Reading the export now')
  })

  it('queues the words and says so when a steer cannot carry them', async () => {
    const onSteer = vi.fn(async () => 'queued' as const)
    const onSubmit = vi.fn()
    const { handlers, hook, session } = await openLiveSession(false, { onSteer, onSubmit })

    act(() => {
      handlers.onDelegation('del-first', [{ endMs: 1_000, speaker: 'user', startMs: 0, text: 'start the import' }])
    })

    await act(async () => {
      hook.rerender({ isBusy: true })
    })

    act(() => {
      handlers.onDelegation('del-second', [
        { endMs: 1_000, speaker: 'user', startMs: 0, text: 'start the import' },
        { endMs: 1_500, speaker: 'assistant', startMs: 1_000, text: 'On it.' },
        { endMs: 2_000, speaker: 'user', startMs: 1_500, text: 'and skip the archive step' }
      ])
    })

    await act(async () => {
      await Promise.resolve()
    })

    // The words were kept for the next turn, and the live model is told so on
    // the delegation it awaits: silence here is what makes it invent a success.
    expect(onSteer).toHaveBeenCalledWith('and skip the archive step')
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(session.speak).toHaveBeenCalledWith(
      'del-second',
      'That request is queued for right after the one already running.'
    )
  })

  it('still ends the conversation on a spoken stop word while a turn is running', async () => {
    const onSteer = vi.fn(async () => 'steered' as const)
    const onStopWord = vi.fn()
    const { handlers, hook } = await openLiveSession(false, { onSteer, onStopWord })

    act(() => {
      handlers.onDelegation('del-first', [{ endMs: 1_000, speaker: 'user', startMs: 0, text: 'start the import' }])
    })

    await act(async () => {
      hook.rerender({ isBusy: true })
    })

    act(() => {
      handlers.onDelegation('del-stop', [{ endMs: 2_000, speaker: 'user', startMs: 1_000, text: 'stop' }])
    })

    expect(onStopWord).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(hook.result.current.status).toBe('idle')
  })
})
