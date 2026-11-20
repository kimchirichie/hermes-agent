// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { chunkForCommentary, toLiveHistory, type VoiceLiveHandlers } from '@/lib/voice-live'
import { $notifications, clearNotifications } from '@/store/notifications'

import { delegationPrompt, MAX_SILENT_MS, PROGRESS_LABEL, useVoiceLiveConversation, WORKING_LINES } from './use-voice-live-conversation'

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
  sessions: [] as Array<{ reference: ReturnType<typeof vi.fn>; speak: ReturnType<typeof vi.fn>; think: ReturnType<typeof vi.fn> }>
}))

vi.mock('@/lib/voice-live', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    VoiceLiveSession: class {
      close = vi.fn()
      instruct = vi.fn()
      reference = vi.fn()
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

// Reply contract (policy module 48): the reply is buffered and ONE directive is
// spoken on settle. "failed" is only an error or cut-off; a turn with no
// directive is "not confirmed"; the detail card never reaches the voice.
describe('GPT-Live reply contract', () => {
  const SETTLE_MS = 16_000

  type Reply = null | { id: string; interrupted?: boolean; pending: boolean; text: string }

  beforeEach(() => {
    resetToasts()
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    resetToasts()
  })

  async function open(
    options: {
      busy?: boolean
      onSteer?: (text: string) => Promise<'queued' | 'steered' | false> | 'queued' | 'steered' | false
      onStopWord?: () => void
      onSubmit?: (text: string, voiceContext: string) => Promise<void> | void
      reply?: () => Reply
    } = {}
  ) {
    const reply = options.reply ?? (() => null)

    const hook = renderHook(
      ({ isBusy }: { isBusy: boolean }) =>
        useVoiceLiveConversation({
          busy: isBusy,
          consumePendingResponse: vi.fn(),
          enabled: true,
          onSteer: options.onSteer,
          onStopWord: options.onStopWord,
          onSubmit: options.onSubmit ?? vi.fn(),
          pendingResponse: reply,
          seedHistory: () => []
        }),
      { initialProps: { isBusy: options.busy ?? false } }
    )

    await act(async () => {
      await hook.result.current.start()
    })

    return {
      handlers: transport.handlers.at(-1) as VoiceLiveHandlers,
      hook,
      session: transport.sessions.at(-1) as {
        reference: ReturnType<typeof vi.fn>
        speak: ReturnType<typeof vi.fn>
        think: ReturnType<typeof vi.fn>
      }
    }
  }

  const advance = (ms: number) =>
    act(async () => {
      vi.advanceTimersByTime(ms)
    })

  it('speaks only the directive on a done reply and keeps ids in the chat', async () => {
    let reply: Reply = null
    const { handlers, session } = await open({ reply: () => reply })

    act(() => handlers.onDelegation('del-ok', []))
    reply = {
      id: 'r1',
      pending: false,
      text: 'DONE: The refund is staged for your approval.\n---\nJourney 1285, order #48213, 9f3ab2c'
    }
    await advance(400)

    const said = session.speak.mock.calls.filter(([id]) => id === 'del-ok').map(([, text]) => text)

    expect(said).toEqual(['The refund is staged for your approval.'])
    expect(said.join(' ')).not.toMatch(/1285|48213|9f3ab2c/)
  })

  it('speaks nothing of a reply while it is still streaming, then one message on settle', async () => {
    let reply: Reply = null
    const { handlers, hook, session } = await open({ reply: () => reply })

    act(() => handlers.onDelegation('del-buf', []))
    await act(async () => hook.rerender({ isBusy: true }))
    reply = { id: 'r1', pending: true, text: 'DONE: First sentence. Second sentence. Third' }
    await advance(1_000)

    expect(session.speak).not.toHaveBeenCalled()

    reply = { id: 'r1', pending: false, text: 'DONE: First sentence. Second sentence.' }
    await act(async () => hook.rerender({ isBusy: false }))
    await advance(400)

    expect(session.speak).toHaveBeenCalledTimes(1)
    expect(session.speak).toHaveBeenCalledWith('del-buf', 'First sentence. Second sentence.')
  })

  it('never speaks raw text that has no directive, and says not confirmed', async () => {
    let reply: Reply = null
    const { handlers, session } = await open({ reply: () => reply })

    act(() => handlers.onDelegation('del-raw', []))
    reply = { id: 'r1', pending: false, text: 'I updated the order and it all went through fine.' }
    await advance(400)

    expect(session.speak).toHaveBeenCalledTimes(1)
    expect(session.speak).toHaveBeenCalledWith('del-raw', 'I could not confirm that.')
  })

  it('uses the last directive, so mid-turn narration never counts', async () => {
    let reply: Reply = null
    const { handlers, session } = await open({ reply: () => reply })

    act(() => handlers.onDelegation('del-last', []))
    reply = { id: 'r1', pending: false, text: 'DONE: wrong early guess.\n\nFAILED: The send did not go through.\n---\ndetail' }
    await advance(400)

    expect(session.speak).toHaveBeenCalledWith('del-last', 'The send did not go through.')
  })

  it('says not confirmed, not failed, when a tool-only turn settles with no reply', async () => {
    const { handlers, session } = await open()

    act(() => handlers.onDelegation('del-tool', []))
    await advance(SETTLE_MS)

    expect(session.speak).toHaveBeenCalledWith('del-tool', 'I could not confirm that.')
    expect(session.speak).not.toHaveBeenCalledWith('del-tool', 'That did not go through.')
  })

  it('says failed when the reply was cut off', async () => {
    let reply: Reply = null
    const { handlers, session } = await open({ reply: () => reply })

    act(() => handlers.onDelegation('del-cut', []))
    reply = { id: 'r1', interrupted: true, pending: false, text: 'DONE: partly writ' }
    await advance(400)

    expect(session.speak).toHaveBeenCalledWith('del-cut', 'That did not go through.')
  })

  it('says failed when the turn cannot be submitted', async () => {
    const { handlers, session } = await open({ onSubmit: () => Promise.reject(new Error('gateway down')) })

    act(() => handlers.onDelegation('del-err', []))
    await act(async () => {
      await Promise.resolve()
    })

    expect(session.speak).toHaveBeenCalledWith('del-err', 'That did not go through.')
  })

  it('speaks the working lines once each, keyed to the delegation, and stops at settle', async () => {
    let reply: Reply = null
    const { handlers, hook, session } = await open({ reply: () => reply })

    act(() => handlers.onDelegation('del-slow', []))
    await act(async () => hook.rerender({ isBusy: true }))
    await advance(WORKING_LINES[0].afterMs + 100)

    expect(session.speak).toHaveBeenCalledWith('del-slow', WORKING_LINES[0].text)

    await act(async () => hook.rerender({ isBusy: true }))
    await advance(WORKING_LINES[1].afterMs - WORKING_LINES[0].afterMs)

    expect(session.speak.mock.calls.filter(([, text]) => WORKING_LINES.some(l => l.text === text))).toHaveLength(2)

    reply = { id: 'r1', pending: false, text: 'DONE: Finished.' }
    await act(async () => hook.rerender({ isBusy: false }))
    await advance(MAX_SILENT_MS)

    expect(session.speak.mock.calls.filter(([, text]) => WORKING_LINES.some(l => l.text === text))).toHaveLength(2)
    expect(session.speak).toHaveBeenCalledWith('del-slow', 'Finished.')
    expect(session.speak).not.toHaveBeenCalledWith('del-slow', 'I could not confirm that.')
  })

  const isWorkingLine = ([, text]: unknown[]) => WORKING_LINES.some(l => l.text === text)

  it('never speaks working for a delegation nothing picked up: no signal ends as not confirmed', async () => {
    const { handlers, session } = await open()

    act(() => handlers.onDelegation('del-dead', []))
    await advance(WORKING_LINES[1].afterMs + 100)

    expect(session.speak.mock.calls.filter(isWorkingLine)).toHaveLength(0)

    await advance(SETTLE_MS)

    expect(session.speak.mock.calls.filter(isWorkingLine)).toHaveLength(0)
    expect(session.speak).toHaveBeenCalledWith('del-dead', 'I could not confirm that.')
  })

  it('never speaks working when the delegation crashes before starting', async () => {
    const { handlers, session } = await open({ onSubmit: () => Promise.reject(new Error('crash')) })

    act(() => handlers.onDelegation('del-crash', []))
    await advance(MAX_SILENT_MS)

    expect(session.speak.mock.calls.filter(isWorkingLine)).toHaveLength(0)
    expect(session.speak).toHaveBeenCalledWith('del-crash', 'That did not go through.')
  })

  it('counts the working wait from the progress signal, not from the send', async () => {
    const { handlers, hook, session } = await open()

    act(() => handlers.onDelegation('del-late', []))
    await advance(10_000)
    await act(async () => hook.rerender({ isBusy: true }))
    await advance(WORKING_LINES[0].afterMs - 1_000)

    expect(session.speak.mock.calls.filter(isWorkingLine)).toHaveLength(0)

    await advance(2_000)

    expect(session.speak).toHaveBeenCalledWith('del-late', WORKING_LINES[0].text)
  })

  it('sends the labelled progress event once, on the first real signal, and never before it', async () => {
    const { handlers, hook, session } = await open()
    const isProgress = ([, text]: unknown[]) => typeof text === 'string' && text.startsWith('[progress]')

    act(() => handlers.onDelegation('del-prog', []))
    await advance(10_000)

    expect(session.think.mock.calls.filter(isProgress)).toHaveLength(0)

    await act(async () => hook.rerender({ isBusy: true }))
    await act(async () => hook.rerender({ isBusy: true }))

    expect(session.think.mock.calls.filter(isProgress)).toHaveLength(1)
    expect(session.think).toHaveBeenCalledWith('del-prog', PROGRESS_LABEL)
  })

  it('never sends the progress label for a delegation nothing picked up', async () => {
    const { handlers, session } = await open()

    act(() => handlers.onDelegation('del-none', []))
    await advance(SETTLE_MS)

    expect(session.think.mock.calls.filter(([, text]) => typeof text === 'string' && text.startsWith('[progress]'))).toHaveLength(0)
  })

  it('caps the silence: a turn that never settles ends as not confirmed', async () => {
    const { handlers, hook, session } = await open()

    act(() => handlers.onDelegation('del-hang', []))
    await act(async () => hook.rerender({ isBusy: true }))
    await advance(MAX_SILENT_MS + 100)

    expect(session.speak).toHaveBeenCalledWith('del-hang', 'I could not confirm that.')
  })

  it('holds the tool-progress think until the automatic acknowledgement has had its turn', async () => {
    const hook = renderHook(() =>
      useVoiceLiveConversation({
        activeToolLabel: () => 'terminal',
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onSubmit: vi.fn(),
        pendingResponse: () => null,
        seedHistory: () => []
      })
    )

    await act(async () => {
      await hook.result.current.start()
    })

    const handlers = transport.handlers.at(-1) as VoiceLiveHandlers
    const session = transport.sessions.at(-1) as { think: ReturnType<typeof vi.fn> }

    act(() => handlers.onDelegation('del-think', []))
    await advance(1_000)
    expect(session.think).not.toHaveBeenCalled()
    await advance(4_000)
    expect(session.think).toHaveBeenCalledWith('del-think', expect.stringContaining('terminal'))
  })

  it('steers a talk-over into the running turn and reports under the newest delegation', async () => {
    let reply: Reply = null
    const onSteer = vi.fn(async () => 'steered' as const)
    const { handlers, hook, session } = await open({ onSteer, reply: () => reply })

    act(() => handlers.onDelegation('del-first', [{ endMs: 1, speaker: 'user', startMs: 0, text: 'start the import' }]))
    await act(async () => hook.rerender({ isBusy: true }))
    act(() =>
      handlers.onDelegation('del-second', [
        { endMs: 1, speaker: 'user', startMs: 0, text: 'start the import' },
        { endMs: 2, speaker: 'assistant', startMs: 1, text: 'On it.' },
        { endMs: 3, speaker: 'user', startMs: 2, text: 'and skip the archive' }
      ])
    )
    await act(async () => {
      await Promise.resolve()
    })
    reply = { id: 'r1', pending: false, text: 'DONE: Imported, archive skipped.' }
    await act(async () => hook.rerender({ isBusy: false }))
    await advance(400)

    expect(onSteer).toHaveBeenCalledWith('and skip the archive')
    expect(session.speak).toHaveBeenCalledWith('del-second', 'Imported, archive skipped.')
    expect(session.speak).not.toHaveBeenCalledWith('del-first', expect.anything())
  })

  it('queues the words and says so when a steer cannot carry them', async () => {
    const onSteer = vi.fn(async () => 'queued' as const)
    const { handlers, hook, session } = await open({ onSteer })

    act(() => handlers.onDelegation('del-first', [{ endMs: 1, speaker: 'user', startMs: 0, text: 'start the import' }]))
    await act(async () => hook.rerender({ isBusy: true }))
    act(() => handlers.onDelegation('del-second', [{ endMs: 3, speaker: 'user', startMs: 2, text: 'skip archive' }]))
    await act(async () => {
      await Promise.resolve()
    })

    expect(session.speak).toHaveBeenCalledWith(
      'del-second',
      'That request is queued for right after the one already running.'
    )
  })

  it('keeps the detail card as a silent reference and speaks it only on a details request', async () => {
    let reply: Reply = null
    const onSubmit = vi.fn()
    const { handlers, hook, session } = await open({ onSubmit, reply: () => reply })

    act(() => handlers.onDelegation('del-a', [{ endMs: 1, speaker: 'user', startMs: 0, text: 'quote total' }]))
    reply = { id: 'r1', pending: false, text: 'DONE: The Taylor ISD quote is ready.\n---\nTotal $4,812.50 on proposal 585, card t_54aa843d' }
    await advance(400)

    expect(session.speak).toHaveBeenCalledTimes(1)
    expect(session.speak).toHaveBeenCalledWith('del-a', 'The Taylor ISD quote is ready.')
    expect(session.reference).toHaveBeenCalledWith('del-a', expect.stringContaining('$4,812.50'))

    reply = null
    act(() => handlers.onDelegation('del-b', [{ endMs: 2, speaker: 'user', startMs: 1, text: 'details' }]))
    expect(onSubmit).toHaveBeenCalledTimes(1)
    const detailSpeech = session.speak.mock.calls.filter(([id]) => id === 'del-b').map(([, text]) => text).join(' ')
    expect(detailSpeech).toContain('4,812.50')
    expect(detailSpeech).not.toMatch(/585|t_54aa843d/)
    await act(async () => hook.rerender({ isBusy: false }))
  })

  it('does not speak an older result\'s details after a turn that failed', async () => {
    let reply: Reply = null
    const { handlers, hook, session } = await open({ reply: () => reply })

    act(() => handlers.onDelegation('del-a', [{ endMs: 1, speaker: 'user', startMs: 0, text: 'quote total' }]))
    reply = { id: 'r1', pending: false, text: 'DONE: The quote is ready.\n---\nTotal $4,812.50' }
    await advance(400)

    reply = null
    act(() => handlers.onDelegation('del-b', [{ endMs: 3, speaker: 'user', startMs: 2, text: 'next one' }]))
    reply = { id: 'r2', interrupted: true, pending: false, text: 'DONE: partly writ' }
    await advance(400)

    reply = null
    act(() => handlers.onDelegation('del-c', [{ endMs: 5, speaker: 'user', startMs: 4, text: 'details' }]))
    const detailSpeech = session.speak.mock.calls.filter(([id]) => id === 'del-c').map(([, text]) => text).join(' ')
    expect(detailSpeech).not.toContain('4,812')
    await act(async () => hook.rerender({ isBusy: false }))
  })

  it('treats a murmur while work runs as no turn: no steer, no speech', async () => {
    const onSteer = vi.fn(async () => 'steered' as const)
    const { handlers, hook, session } = await open({ onSteer })

    act(() => handlers.onDelegation('del-first', [{ endMs: 1, speaker: 'user', startMs: 0, text: 'start the import' }]))
    await act(async () => hook.rerender({ isBusy: true }))
    session.speak.mockClear()

    for (const text of ['Mm', 'Got it', 'uh', 'journeys', 'yeah okay']) {
      act(() => handlers.onDelegation(`del-${text}`, [{ endMs: 3, speaker: 'user', startMs: 2, text }]))
    }

    await act(async () => {
      await Promise.resolve()
    })

    expect(onSteer).not.toHaveBeenCalled()
    expect(session.speak).not.toHaveBeenCalled()
    expect(session.think).toHaveBeenCalledWith('del-Mm', expect.stringContaining('Say nothing'))
  })

  it('never speaks a label, an id, or a progress claim next to a result', async () => {
    let reply: Reply = null
    const { handlers, session } = await open({ reply: () => reply })

    act(() => handlers.onDelegation('del-c', []))
    reply = {
      id: 'r1',
      pending: false,
      text: 'DONE: Hermes is working, not done yet. Filed: 1677 as card 584 for the nudge build.\n---\nt_54aa843d'
    }
    await advance(400)

    const said = session.speak.mock.calls.map(([, text]) => String(text)).join(' ')
    expect(said).not.toMatch(/DONE|not done yet|1677|584|t_54aa843d/i)
    expect(said).toContain('nudge build')
  })

  it('still ends the conversation on a spoken stop word while a turn is running', async () => {
    const onStopWord = vi.fn()
    const { handlers, hook } = await open({ onStopWord })

    act(() => handlers.onDelegation('del-first', [{ endMs: 1, speaker: 'user', startMs: 0, text: 'start the import' }]))
    await act(async () => hook.rerender({ isBusy: true }))
    act(() => handlers.onDelegation('del-stop', [{ endMs: 2, speaker: 'user', startMs: 1, text: 'stop' }]))

    expect(onStopWord).toHaveBeenCalledTimes(1)
    expect(hook.result.current.status).toBe('idle')
  })
})
