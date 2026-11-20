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
 * first blank line. Only the headline is spoken (ids rewritten out, progress claims dropped); the detail card is a silent reference the voice may speak on request. Text with no
 * directive is never spoken, and the turn counts as not confirmed.
 */
export type VoiceReplyKind = 'done' | 'failed' | 'not_confirmed'

export interface VoiceReply {
  kind: VoiceReplyKind
  /** Speakable headline, already sanitised; empty when the model gave none. */
  spoken: string
  /** The detail card below the delimiter, raw: chat text and silent reference, never spoken on arrival. */
  details: string
}

/** Spoken when the turn went through nothing we can verify. Never contains "done". */
export const NOT_CONFIRMED_LINE = 'I could not confirm that.'
/** Spoken only on an error or a cut-off (stop word, submit failure, interrupted turn). */
export const FAILED_LINE = 'That did not go through.'

const DIRECTIVE_RE = /^[ \t>*_#-]*(DONE|FAILED|NOT[ _-]CONFIRMED)[*_ \t]*:[*_ \t]*/gim
const DELIMITER_RE = /^[ \t]*-{3,}[ \t]*$/m
/** A directive is a sentence or two; more than this is a detail card in the wrong place. */
const SPOKEN_LIMIT = 400
const DETAILS_SPOKEN_LIMIT = 2_800

// Ids mean nothing by ear and the voice garbles them ("card 584", "Filed: 1677"): the noun stays,
// the number goes.
const CARD_ID_RE = /\bt_[0-9a-f]{6,}\b/gi
const NUMBERED_NOUN_RE = /\b(card|proposal|journey|ticket|order|invoice|PR|pull request|issue|row|feedback)s?\s*(?:#|number|no\.?)?\s*\d{3,}\b/gi
const BARE_HASH_NUMBER_RE = /#\d{3,}\b/g
// "issues 1234 and 1235": the noun rewrite only takes the first id of a list.
const ID_LIST_TAIL_RE = /(\bthe (?:cards?|proposals?|journeys?|tickets?|orders?|invoices?|PRs?|pull requests?|issues?|rows?|feedback))((?:\s*(?:,|and|or|&)\s*#?\d{3,}\b)+)/gi
// Letter-prefixed ids by ear ("D354").
const LETTER_ID_RE = /\b[A-Z]{1,3}\d{3,}\b/g
const FILED_NUMBER_RE = /\b(filed|created|opened|logged)\s*:?\s*#?\d{3,}\b/gi

// A progress claim next to a result contradicts it ("not done yet. DONE: ..."); a headline carries the result only.
const PROGRESS_PHRASE = '(?:hermes is working|not done yet|still working(?: on it)?|still on it|working on it|taking a bit longer)'
// Only a sentence made of nothing but progress phrases goes; "The refund is still working through Stripe." is content.
const PROGRESS_SENTENCE_RE = new RegExp(
  `(?:^|(?<=[.!?]\\s))\\s*${PROGRESS_PHRASE}(?:[,;\\s]+(?:and\\s+)?${PROGRESS_PHRASE})*\\s*(?:[.!?]+\\s*|$)`,
  'gi'
)

/** Rewrite ids out of a spoken line and drop progress claims. */
export function scrubSpoken(text: string): string {
  return text
    .replace(LABEL_RE, ' ')
    .replace(PROGRESS_SENTENCE_RE, ' ')
    .replace(CARD_ID_RE, 'the card')
    .replace(NUMBERED_NOUN_RE, (_whole, noun: string) => `the ${noun.toLowerCase()}`)
    .replace(ID_LIST_TAIL_RE, '$1')
    .replace(LETTER_ID_RE, '')
    .replace(FILED_NUMBER_RE, (_whole, verb: string) => verb)
    .replace(BARE_HASH_NUMBER_RE, '')
    .replace(/\s+/g, ' ')
    .trim()
}

const LABEL_RE = /\b(?:DONE|FAILED|NOT[ _-]CONFIRMED)\s*:/g

const DETAILS_REQUEST_RE =
  /^(?:(?:please\s+)?(?:say|tell me|give me|read(?: me)?)\s+(?:the\s+)?(?:more|details?)(?:\s+(?:please|again))?|(?:the\s+)?details?(?:\s+please)?|more(?:\s+please)?|go on|what else|read (?:it|them) (?:to me|out))$/i

/** True when the whole utterance asks for the details of the last result. */
export function isDetailsRequest(utterance: string): boolean {
  return DETAILS_REQUEST_RE.test(utterance.toLowerCase().replace(/[^\p{L}\p{N}\s']/gu, ' ').replace(/\s+/g, ' ').trim())
}

/** Plain-prose version of a detail card for speaking on request: tables, links and ids dropped. */
export function speakableDetails(details: string): string {
  const prose = details
    .split('\n')
    .filter(line => !/^\s*\|/.test(line) && !/^\s*[-:| ]{3,}$/.test(line))
    .join('\n')

  return scrubSpoken(sanitizeTextForSpeech(prose)).slice(0, DETAILS_SPOKEN_LIMIT)
}

// Backchannel: a thinking-aloud fragment while work runs is not a turn ("mm", "got it", "uh").
const BACKCHANNEL_WORDS = new Set([
  'mm', 'mmm', 'hmm', 'hm', 'mhm', 'uh', 'um', 'er', 'ah', 'oh', 'yeah', 'yep', 'yes', 'ok', 'okay', 'right', 'sure',
  'cool', 'alright', 'so', 'and', 'well', 'like', 'got', 'it', 'i', 'see', 'huh', 'nice', 'great', 'good', 'fine'
])

// One bare word is usually a trailing fragment ("journeys", "some"), unless it is a steer.
const STEER_WORDS = new Set(['no', 'wait', 'hold', 'dont', "don't", 'cancel', 'undo', 'skip', 'abort', 'pause', 'resume',
  'status', 'approve', 'approved', 'deny', 'reject', 'confirm', 'refund', 'send', 'retry', 'faster', 'stop'
])

/** A fragment of about three words or fewer that steers nothing. */
export function isBackchannel(utterance: string): boolean {
  const words = utterance
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s'-]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean)

  if (words.length === 0) {
    return true
  }

  if (words.length > 3 || words.some(word => STEER_WORDS.has(word))) {
    return false
  }

  return words.length === 1 || words.every(word => BACKCHANNEL_WORDS.has(word))
}

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
    // A directive after the delimiter of an earlier one is detail-card text, not a new headline.
    if (last && DELIMITER_RE.test(text.slice(last.index + last[0].length, match.index))) {
      break
    }

    last = match
  }

  if (!last) {
    return { details: '', kind: 'not_confirmed', spoken: '' }
  }

  const body = text.slice(last.index + last[0].length)
  const delimiter = DELIMITER_RE.exec(body)
  const head = delimiter ? body.slice(0, delimiter.index) : (body.split(/\n\s*\n/, 1)[0] ?? '')
  const details = (delimiter ? body.slice(delimiter.index + delimiter[0].length) : body.slice(head.length)).trim()
  const spoken = scrubSpoken(sanitizeTextForSpeech(head)).slice(0, SPOKEN_LIMIT)

  return { details, kind: kindOf(last[1]), spoken }
}
