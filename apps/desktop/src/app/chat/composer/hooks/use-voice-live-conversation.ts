import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { sanitizeTextForSpeech } from '@/lib/speech-text'
import { type LiveHistoryMessage, type LiveTranscriptFragment, VoiceLiveSession } from '@/lib/voice-live'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'
import { notify, notifyError } from '@/store/notifications'

import type { SteerOutcome } from '../busy-submit-state'
import { useComposerScope } from '../scope'

import { micError } from './use-mic-recorder'
import type { ConversationStatus } from './use-voice-conversation'

/** How long an accepted delegation may sit before the gateway shows the turn running. */
const SUBMIT_SETTLE_GRACE_MS = 15_000
/** Quiet after the last user transcript fragment before the utterance is judged
 *  as a whole ("stop" ends the chat; "stop the container" is a request). */
const UTTERANCE_SETTLE_MS = 1_500

/** Spoken on the delegation the live model just created when the words could
 *  not reach the running turn. `think` is never read aloud, and silence lets
 *  the live model claim the request went through (the settle-path lesson). */
const QUEUED_NOTICE = 'That request is queued for right after the one already running.'
const UNDELIVERED_NOTICE = 'Sorry, I could not pass that through while the current request is still running.'

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

interface VoiceLiveConversationOptions {
  busy: boolean
  enabled: boolean
  onFatalError?: () => void
  /**
   * Steer the running Hermes turn with the words the user just spoke — the
   * composer's steer seam. Voice never interrupts a turn: a spoken correction
   * mid-turn redirects it (the composer's Steer action, not its Stop button),
   * and a steer that cannot carry the words queues them as the next turn. The
   * outcome decides which delegation the reply is reported under.
   */
  onSteer?: (text: string) => Promise<SteerOutcome> | SteerOutcome
  onStopWord?: () => void
  /** Submit a Hermes turn: `text` is the user's last words (the bubble and the
   *  persisted row), `voiceContext` the recent spoken exchange for the model. */
  onSubmit: (text: string, voiceContext: string) => Promise<void> | void
  pendingResponse: () => PendingVoiceResponse | null
  consumePendingResponse: () => void
  /** Text turns to seed the live model with when the session opens. */
  seedHistory: () => LiveHistoryMessage[]
  /** Names of tools currently running in the turn (quiet progress for the voice). */
  activeToolLabel?: () => null | string
  beforeMicOpen?: () => Promise<void> | void
}

/** Turn transcript fragments into the Hermes turn: `prompt` is what the user
 *  last said (the persisted user row), `context` the recent spoken exchange
 *  that rides the model input only (see tools/voice_live.py). */
export function delegationPrompt(context: LiveTranscriptFragment[]): { context: string; prompt: string } {
  const turns: Array<{ speaker: 'assistant' | 'user'; text: string }> = []

  for (const fragment of context) {
    const last = turns.at(-1)

    if (last && last.speaker === fragment.speaker) {
      last.text += fragment.text
    } else {
      turns.push({ speaker: fragment.speaker, text: fragment.text })
    }
  }

  const lastUser = [...turns].reverse().find(turn => turn.speaker === 'user')
  const prompt = (lastUser?.text ?? '').replace(/\s+/g, ' ').trim()

  const transcript = turns
    .map(turn => `${turn.speaker === 'user' ? 'User' : 'Voice assistant'}: ${turn.text.replace(/\s+/g, ' ').trim()}`)
    .filter(line => !line.endsWith(': '))
    .join('\n')

  return { context: transcript, prompt: prompt || transcript.slice(-400) }
}

/**
 * Body of the session-end toast. `connection_lost` and `closed` are our own
 * machine reasons (`lib/voice-live.ts`) and get i18n copy; so does a blank
 * reason, which has no wording of its own. Any other reason is server-sent and
 * unbounded, so it passes through verbatim (issue #111987 — no redaction claim
 * for vendor strings).
 */
export function liveEndedMessage(
  reason: string,
  usageSeconds: null | number,
  copy: { liveEndedClosed: string; liveEndedConnectionLost: string }
): string {
  let text = reason?.trim() ?? ''

  if (text === 'connection_lost') {
    text = copy.liveEndedConnectionLost
  } else if (text === 'closed' || !text) {
    text = copy.liveEndedClosed
  }

  return usageSeconds != null ? `${text} (${Math.round(usageSeconds)}s)` : text
}

/**
 * GPT-Live conversation engine — same public shape as `useVoiceConversation`
 * so the composer can mount either from `voice.voice_chat_mode`.
 *
 * Status mapping: `listening` = session up, voice idle; `speaking` = the
 * remote track is producing audio; `thinking` = a delegation is in flight in
 * Hermes. There is no `transcribing` phase: the voice model owns speech.
 */
export function useVoiceLiveConversation({
  busy,
  enabled,
  onFatalError,
  onSteer,
  onStopWord,
  onSubmit,
  pendingResponse,
  consumePendingResponse,
  seedHistory,
  activeToolLabel,
  beforeMicOpen
}: VoiceLiveConversationOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [muted, setMuted] = useState(false)
  const [level, setLevel] = useState(0)
  // Mirrors delegationRef for the reply-drive effect: a new delegation must
  // restart the feed loop, and a ref write alone does not re-render.
  const [activeDelegation, setActiveDelegation] = useState<null | string>(null)
  const sessionRef = useRef<null | VoiceLiveSession>(null)
  // The scope's session owner (a Bot's own connection + profile) picks the
  // GPT-Live backend and voice; a ref keeps the long-lived start closures
  // reading the current value.
  const { connectionId: ownerConnectionId, profile: ownerProfile } = useComposerScope()
  const ownerRef = useRef({ connectionId: ownerConnectionId, profile: ownerProfile })
  ownerRef.current = { connectionId: ownerConnectionId, profile: ownerProfile }
  // Bumped by every start/end so an in-flight start() that lost the race
  // (StrictMode double-effect, quick toggle) closes its session instead of
  // leaving a second billed one running.
  const startEpochRef = useRef(0)
  const startingRef = useRef(false)
  // Set at delegation submit; a turn is only "settled" once it has been seen
  // running (busy) or produced a reply — the gateway ack lags the submit.
  const turnObservedRef = useRef(false)
  const submittedAtRef = useRef(0)
  const enabledRef = useRef(enabled)
  const busyRef = useRef(busy)
  const speakingRef = useRef(false)
  const userUtteranceRef = useRef('')
  const utteranceTimerRef = useRef<null | number>(null)
  const delegationRef = useRef<null | string>(null)
  const spokenLengthRef = useRef(0)
  const spokenResponseIdRef = useRef<null | string>(null)
  const lastToolLabelRef = useRef<null | string>(null)
  const wasEnabledRef = useRef(enabled)
  // Last spoken correction wins: a second utterance arriving while the first
  // steer is still in flight supersedes it.
  const steerSeqRef = useRef(0)

  const latest = useRef({
    activeToolLabel,
    beforeMicOpen,
    onFatalError,
    onSteer,
    onStopWord,
    onSubmit,
    pendingResponse,
    consumePendingResponse,
    seedHistory
  })

  latest.current = {
    activeToolLabel,
    beforeMicOpen,
    onFatalError,
    onSteer,
    onStopWord,
    onSubmit,
    pendingResponse,
    consumePendingResponse,
    seedHistory
  }

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  const setDelegation = useCallback((id: null | string) => {
    delegationRef.current = id
    setActiveDelegation(id)
  }, [])

  const refreshStatus = useCallback(() => {
    if (!sessionRef.current) {
      setStatus('idle')

      return
    }

    if (speakingRef.current) {
      setStatus('speaking')
    } else if (delegationRef.current) {
      setStatus('thinking')
    } else {
      setStatus('listening')
    }
  }, [])

  const end = useCallback(async () => {
    startEpochRef.current += 1
    startingRef.current = false

    if (utteranceTimerRef.current) {
      window.clearTimeout(utteranceTimerRef.current)
      utteranceTimerRef.current = null
    }

    userUtteranceRef.current = ''
    const session = sessionRef.current
    sessionRef.current = null
    setDelegation(null)
    spokenResponseIdRef.current = null
    spokenLengthRef.current = 0
    speakingRef.current = false
    session?.close()
    setMuted(false)
    setLevel(0)
    setStatus('idle')
  }, [setDelegation])

  const start = useCallback(async () => {
    if (sessionRef.current || startingRef.current) {
      return
    }

    startingRef.current = true
    const epoch = ++startEpochRef.current

    try {
      await latest.current.beforeMicOpen?.()
    } catch {
      // A wake-pause failure must not block an explicit start.
    }

    if (!enabledRef.current || startEpochRef.current !== epoch) {
      startingRef.current = false

      return
    }

    /**
     * A spoken correction while a turn is running: steer it, never interrupt
     * it. `delegationId` is the delegation the live model just created and is
     * waiting on, so the running turn's reply is reported under it once the
     * steer lands.
     */
    const steerRunningTurn = async (live: VoiceLiveSession, delegationId: string, prompt: string) => {
      const seq = ++steerSeqRef.current
      let outcome: SteerOutcome = false

      try {
        outcome = (await latest.current.onSteer?.(prompt)) ?? false
      } catch {
        outcome = false
      }

      // The conversation ended, or a newer correction superseded this one.
      if (sessionRef.current !== live || steerSeqRef.current !== seq) {
        return
      }

      if (outcome === 'steered') {
        // The running turn carries the correction now, so its reply answers the
        // delegation the live model is waiting on — report it under that one.
        //
        // The spoken cursor is deliberately NOT reset: a steer seals the live
        // bubble and the answer continues in a new one, so every word already
        // spoken is still in the collected turn text. Restarting the cursor
        // would say the whole running turn again; it keeps reading from where
        // it stopped, so the user hears only the correction's tail.
        setDelegation(delegationId)
        turnObservedRef.current = true
        submittedAtRef.current = Date.now()
        refreshStatus()

        return
      }

      // The words never reached the running turn (queued as the next one, or
      // refused outright). Say so on the delegation the live model awaits:
      // `think` would be silent, and silence lets the model claim the request
      // went through.
      live.speak(delegationId, outcome === 'queued' ? QUEUED_NOTICE : UNDELIVERED_NOTICE)
    }

    const session = new VoiceLiveSession(
      {
        // The voice model answers a bare "stop" itself (it just goes quiet) and
        // never delegates it, so the spoken stop phrase is judged on the user
        // transcript once the utterance settles.
        onTranscript: fragment => {
          if (fragment.speaker !== 'user') {
            return
          }

          userUtteranceRef.current += fragment.text

          if (utteranceTimerRef.current) {
            window.clearTimeout(utteranceTimerRef.current)
          }

          utteranceTimerRef.current = window.setTimeout(() => {
            utteranceTimerRef.current = null
            const utterance = userUtteranceRef.current
            userUtteranceRef.current = ''

            if (sessionRef.current === session && isVoiceStopCommand(utterance)) {
              void end()
              latest.current.onStopWord?.()
            }
          }, UTTERANCE_SETTLE_MS)
        },
        onClosed: (reason, usageSeconds) => {
          if (sessionRef.current !== session) {
            return
          }

          sessionRef.current = null
          setDelegation(null)
          setStatus('idle')

          if (reason !== 'close_requested') {
            notify({
              kind: 'warning',
              message: liveEndedMessage(reason, usageSeconds, voiceCopy),
              title: voiceCopy.liveEnded
            })
            latest.current.onFatalError?.()
          }
        },
        onDelegation: (delegationId, context) => {
          if (sessionRef.current !== session) {
            return
          }

          const { context: voiceContext, prompt } = delegationPrompt(context)

          // A spoken stop command ends the conversation instead of becoming a turn.
          if (prompt && isVoiceStopCommand(prompt)) {
            void end()
            latest.current.onStopWord?.()

            return
          }

          // A newer request while a turn is running is a correction, not a
          // reason to throw work away: it STEERS the running turn — the same
          // policy the text composer follows (`busy_input_mode: steer`).
          // Nothing on this path interrupts: the old code called onInterrupt
          // here, which killed whatever Hermes was doing mid-tool and let the
          // voice answer a request that had been abandoned.
          if (busyRef.current) {
            void steerRunningTurn(session, delegationId, prompt)

            return
          }

          setDelegation(delegationId)
          spokenResponseIdRef.current = null
          spokenLengthRef.current = 0
          lastToolLabelRef.current = null
          turnObservedRef.current = false
          submittedAtRef.current = Date.now()
          latest.current.consumePendingResponse()
          refreshStatus()
          void Promise.resolve(latest.current.onSubmit(prompt, voiceContext)).catch(error => {
            notifyError(error, voiceCopy.liveDelegationFailed)
            session.speak(delegationId, 'Sorry, I could not reach Hermes for that request.')
            setDelegation(null)
            refreshStatus()
          })
        },
        onError: (message, fatal) => {
          notify({ kind: fatal ? 'error' : 'warning', message, title: voiceCopy.liveError })
        },
        onSpeakingChange: speaking => {
          speakingRef.current = speaking
          setLevel(speaking ? 0.6 : 0)
          refreshStatus()
        }
      },
      ownerRef.current
    )

    sessionRef.current = session
    startingRef.current = false
    setMuted(false)
    setStatus('thinking')

    try {
      await session.start(latest.current.seedHistory())

      if (sessionRef.current !== session || startEpochRef.current !== epoch) {
        session.close()

        return
      }

      refreshStatus()
    } catch (error) {
      if (sessionRef.current === session) {
        sessionRef.current = null
      }

      session.close()

      if (startEpochRef.current !== epoch) {
        return
      }

      // Only a mic DOMException gets the recorder's copy: this catch also
      // takes non-mic start failures ('GPT-Live session already started',
      // 'Missing local SDP offer', API errors) — those keep their own message.
      notifyError(error instanceof DOMException ? micError(error, voiceCopy) : error, voiceCopy.couldNotStartSession)
      setStatus('idle')
      latest.current.onFatalError?.()
    }
  }, [end, refreshStatus, setDelegation, voiceCopy])

  // Drive the reply back into the voice: stream commentary as Hermes writes
  // it (sentence-chunked), quiet tool progress as thinking appends, and clear
  // the delegation when the turn settles.
  // eslint-disable-next-line no-restricted-syntax -- turn-coordination refs (delegation id / spoken cursor), not atom mirrors
  useEffect(() => {
    const session = sessionRef.current
    const delegationId = delegationRef.current

    if (!session || !delegationId) {
      return undefined
    }

    const tick = () => {
      if (sessionRef.current !== session || delegationRef.current !== delegationId) {
        return
      }

      if (busyRef.current) {
        turnObservedRef.current = true
      }

      const tool = latest.current.activeToolLabel?.() ?? null

      if (tool && tool !== lastToolLabelRef.current) {
        lastToolLabelRef.current = tool
        session.think(delegationId, `Hermes is working: ${tool}. Not done yet.`)
      }

      const response = latest.current.pendingResponse()

      if (response) {
        turnObservedRef.current = true

        if (spokenResponseIdRef.current !== response.id) {
          spokenResponseIdRef.current = response.id
          spokenLengthRef.current = 0
        }

        const spoken = sanitizeTextForSpeech(response.text)

        // Append only completed sentences while streaming; the tail lands on settle.
        if (response.pending || busyRef.current) {
          const boundary = spoken.lastIndexOf('. ', spoken.length - 2)
          const cut = boundary > spokenLengthRef.current ? boundary + 1 : spokenLengthRef.current

          if (cut > spokenLengthRef.current) {
            session.speak(delegationId, spoken.slice(spokenLengthRef.current, cut))
            spokenLengthRef.current = cut
          }

          return
        }

        if (spoken.length > spokenLengthRef.current) {
          session.speak(delegationId, spoken.slice(spokenLengthRef.current))
          spokenLengthRef.current = spoken.length
        }

        latest.current.consumePendingResponse()
        setDelegation(null)
        refreshStatus()

        return
      }

      // The submit ack lags: give the turn time to be seen running before
      // reading "idle and no reply" as a finished turn.
      if (
        !busyRef.current &&
        (turnObservedRef.current || Date.now() - submittedAtRef.current > SUBMIT_SETTLE_GRACE_MS)
      ) {
        // Turn settled without a speakable reply (tool-only, error, interrupted).
        // This has to be SPOKEN: `think` is never read aloud, so the live model
        // hears nothing here and fills the silence from its own context —
        // announcing a success Hermes never reported (MS journey 1285, the 404
        // turn, said the update went through).
        if (spokenLengthRef.current === 0) {
          session.speak(delegationId, 'That did not go through.')
        }

        setDelegation(null)
        refreshStatus()
      }
    }

    const timer = window.setInterval(tick, 200)
    tick()

    return () => window.clearInterval(timer)
  }, [activeDelegation, busy, refreshStatus, setDelegation, status])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value
      sessionRef.current?.setMuted(next)

      return next
    })
  }, [])

  /** No explicit turn boundary in full duplex; a nudge tells the voice to answer now. */
  const stopTurn = useCallback(() => {
    sessionRef.current?.instruct('The user has finished speaking. Respond now to what they said.')
  }, [])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (enabled && !wasEnabledRef.current) {
      void start()
    }

    if (!enabled && wasEnabledRef.current) {
      void end()
    }

    wasEnabledRef.current = enabled
  }, [enabled, end, start])

  useEffect(() => () => void end(), [end])

  return { end, level, muted, start, status, stopTurn, toggleMute }
}
