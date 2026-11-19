import { describe, expect, it } from 'vitest'

import { parseVoiceReply } from './voice-live-reply'

describe('parseVoiceReply', () => {
  it('splits the directive from the detail card at the --- line', () => {
    expect(parseVoiceReply('DONE: Staged the refund.\n---\nOrder #4821, id 9f3ab2c1')).toEqual({
      kind: 'done',
      spoken: 'Staged the refund.'
    })
  })

  it('ends the directive at the first blank line when there is no delimiter', () => {
    expect(parseVoiceReply('FAILED: Could not send it.\n\nTrace: 404 at /x').spoken).toBe('Could not send it.')
  })

  it('reads all three kinds, tolerating bold and spacing', () => {
    expect(parseVoiceReply('**NOT CONFIRMED:** Nothing came back.').kind).toBe('not_confirmed')
    expect(parseVoiceReply('not confirmed: nothing').kind).toBe('not_confirmed')
    expect(parseVoiceReply('FAILED : no').kind).toBe('failed')
  })

  it('treats text with no directive as not confirmed with nothing to say', () => {
    expect(parseVoiceReply('All good, I sent it.')).toEqual({ kind: 'not_confirmed', spoken: '' })
    expect(parseVoiceReply('')).toEqual({ kind: 'not_confirmed', spoken: '' })
  })

  it('does not mistake a directive word inside a sentence for a directive', () => {
    expect(parseVoiceReply('I am done: really').kind).toBe('not_confirmed')
  })

  it('takes the last directive and drops dense identifiers from the spoken line', () => {
    const reply = parseVoiceReply('DONE: early.\nFAILED: Order 9f3ab2c1d4e5 was not sent.\n---\nx')

    expect(reply.kind).toBe('failed')
    expect(reply.spoken).not.toContain('9f3ab2c1d4e5')
  })
})
