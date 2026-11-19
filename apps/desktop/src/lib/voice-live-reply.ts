import { sanitizeTextForSpeech } from '@/lib/speech-text'

/**
 * The reply contract between Hermes and the live voice model (policy module
 * 48-comms-voice-handoff). A voice turn's final message is
 *
 *     DONE: one or two plain sentences.
 *     ---
 *     detail card: ids, links, tables, timestamps (chat only)
 *
 * The directive is `DONE:`, `FAILED:` or `NOT CONFIRMED:` at the start of a
 * line. The delimiter is a line holding only `---` (a horizontal rule, so the
 * chat row still reads cleanly); with no delimiter the directive ends at the
 * first blank line. Only the directive text is ever spoken. Text with no
 * directive is never spoken, and the turn counts as not confirmed.
 */
export type VoiceReplyKind = 'done' | 'failed' | 'not_confirmed'

export interface VoiceReply {
  kind: VoiceReplyKind
  /** Speakable directive text, already sanitised; empty when the model gave none. */
  spoken: string
}

/** Spoken when the turn went through nothing we can verify. Never contains "done". */
export const NOT_CONFIRMED_LINE = 'I could not confirm that.'
/** Spoken only on an error or a cut-off (stop word, submit failure, interrupted turn). */
export const FAILED_LINE = 'That did not go through.'

const DIRECTIVE_RE = /^[ \t>*_#-]*(DONE|FAILED|NOT[ _-]CONFIRMED)[*_ \t]*:[*_ \t]*/gim
const DELIMITER_RE = /^[ \t]*-{3,}[ \t]*$/m
/** A directive is a sentence or two; more than this is a detail card in the wrong place. */
const SPOKEN_LIMIT = 400

function kindOf(label: string): VoiceReplyKind {
  const key = label.toUpperCase()

  return key === 'DONE' ? 'done' : key === 'FAILED' ? 'failed' : 'not_confirmed'
}

/**
 * Parse a finished voice turn. The LAST directive wins, so narration the model
 * wrote between tool calls ("let me check") never counts. No directive means
 * `not_confirmed` with empty speech: the caller says `NOT_CONFIRMED_LINE`.
 */
export function parseVoiceReply(text: string): VoiceReply {
  let last: null | RegExpExecArray = null

  for (const match of text.matchAll(DIRECTIVE_RE)) {
    last = match
  }

  if (!last) {
    return { kind: 'not_confirmed', spoken: '' }
  }

  const body = text.slice(last.index + last[0].length)
  const delimiter = DELIMITER_RE.exec(body)
  const head = delimiter ? body.slice(0, delimiter.index) : (body.split(/\n\s*\n/, 1)[0] ?? '')
  const spoken = sanitizeTextForSpeech(head).replace(/\s+/g, ' ').trim().slice(0, SPOKEN_LIMIT)

  return { kind: kindOf(last[1]), spoken }
}
