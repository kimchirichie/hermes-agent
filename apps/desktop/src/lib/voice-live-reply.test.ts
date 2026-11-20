import { describe, expect, it } from 'vitest'

import { isBackchannel, isDetailsRequest, parseVoiceReply, speakableDetails } from './voice-live-reply'

describe('parseVoiceReply', () => {
  it('splits the directive from the detail card at the --- line', () => {
    expect(parseVoiceReply('DONE: Staged the refund.\n---\nOrder #4821, id 9f3ab2c1')).toEqual({
      details: 'Order #4821, id 9f3ab2c1',
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
    expect(parseVoiceReply('All good, I sent it.')).toEqual({ details: '', kind: 'not_confirmed', spoken: '' })
    expect(parseVoiceReply('')).toEqual({ details: '', kind: 'not_confirmed', spoken: '' })
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

describe('headline hygiene', () => {
  it('ignores a directive word that sits inside the detail card', () => {
    const reply = parseVoiceReply('DONE: Negative priorities work.\n\n---\n- DONE: created a card at -5\n- FAILED: nothing')

    expect(reply.kind).toBe('done')
    expect(reply.spoken).toBe('Negative priorities work.')
  })

  it('never speaks a label that appears mid-headline', () => {
    expect(parseVoiceReply('DONE: Moved the cards. NOT CONFIRMED: the cap is live.').spoken).not.toMatch(/DONE|CONFIRMED:/)
  })

  it('rewrites ids out of the headline', () => {
    expect(parseVoiceReply('DONE: Filed: 1677 as card 584 for the nudge build.').spoken).toBe('Filed as the card for the nudge build.')
    expect(parseVoiceReply('DONE: Filed t_54aa843d for you.').spoken).not.toContain('54aa843d')
    expect(parseVoiceReply('DONE: Staged proposal 585 and journey 1199.').spoken).toBe('Staged the proposal and the journey.')
  })

  it('drops a progress claim written next to a result', () => {
    expect(parseVoiceReply('DONE: Hermes is working, not done yet. The quote is ready.').spoken).toBe('The quote is ready.')
  })
})

describe('scrub widening', () => {
  it('drops every id of a list and letter-prefixed ids', () => {
    expect(parseVoiceReply('DONE: Opened issues 1234 and 1235.').spoken).toBe('Opened the issue.')
    expect(parseVoiceReply('DONE: Draft D354 is ready.').spoken).toBe('Draft is ready.')
  })

  it('keeps content sentences that merely mention progress words', () => {
    expect(parseVoiceReply('DONE: Sent the email to Sam. The refund is still working through Stripe.').spoken).toBe(
      'Sent the email to Sam. The refund is still working through Stripe.'
    )
    expect(parseVoiceReply('DONE: Not done yet. Sent the email.').spoken).toBe('Sent the email.')
  })

  it('treats command words as steers, not murmurs', () => {
    for (const cmd of ['status', 'approve', 'refund', 'faster']) {
      expect(isBackchannel(cmd)).toBe(false)
    }
  })
})

describe('details and backchannel', () => {
  it('recognises a request for the details, and only that', () => {
    for (const ok of ['details', 'Details please', 'say more', 'Read me the details.', 'tell me more', 'more']) {
      expect(isDetailsRequest(ok)).toBe(true)
    }

    expect(isDetailsRequest('what are the details of the Taylor quote')).toBe(false)
    expect(isDetailsRequest('send the details to Sam')).toBe(false)
  })

  it('speaks details as prose without tables or ids', () => {
    const spoken = speakableDetails('Total 4,812 dollars.\n| a | b |\n|---|---|\nCard t_54aa843d, proposal 585.')

    expect(spoken).toContain('Total 4,812 dollars.')
    expect(spoken).not.toMatch(/\||585|54aa843d/)
  })

  it('treats short murmurs as backchannel and real words as turns', () => {
    for (const murmur of ['Mm', 'Got it', 'uh', 'yeah', 'journeys', 'some', 'okay yeah', '']) {
      expect(isBackchannel(murmur)).toBe(true)
    }

    for (const turn of ['wait', 'no skip the archive', 'start the import now', 'what is the total today']) {
      expect(isBackchannel(turn)).toBe(false)
    }
  })
})
