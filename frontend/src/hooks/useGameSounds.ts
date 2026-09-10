/**
 * useGameSounds — Web Audio API synthesized sound effects for Bluff game events.
 *
 * No audio files needed. Uses the browser's AudioContext to synthesize
 * short tones inline. Safe: lazy-initialises AudioContext on first call
 * (satisfies autoplay policy — AudioContext only created after a user gesture).
 *
 * Events:
 *  play()       — soft card-swipe tone when human plays cards
 *  callBluff()  — tense rising chord when bluff is called
 *  bluffCaught()— punchy descending tones (challenger was RIGHT)
 *  wrongCall()  — low sad tone (challenger was WRONG, absorbs pile)
 *  win()        — bright ascending fanfare
 *  lose()       — descending diminished triad
 *  draw()       — neutral neutral neutral
 *  pass()       — soft click
 */

import { useCallback, useRef } from "react";

type AudioCtxRef = AudioContext | null;

/** Synthesise a tone. Frequency in Hz, duration in seconds, type = OscillatorType */
function playTone(
  ctx: AudioContext,
  freq: number,
  duration: number,
  startOffset: number = 0,
  type: OscillatorType = "sine",
  gain: number = 0.18
) {
  const osc = ctx.createOscillator();
  const gainNode = ctx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, ctx.currentTime + startOffset);
  gainNode.gain.setValueAtTime(gain, ctx.currentTime + startOffset);
  gainNode.gain.exponentialRampToValueAtTime(
    0.001,
    ctx.currentTime + startOffset + duration
  );
  osc.connect(gainNode);
  gainNode.connect(ctx.destination);
  osc.start(ctx.currentTime + startOffset);
  osc.stop(ctx.currentTime + startOffset + duration + 0.01);
}

export function useGameSounds() {
  const ctxRef = useRef<AudioCtxRef>(null);

  const getCtx = useCallback((): AudioContext | null => {
    if (typeof window === "undefined") return null;
    if (!ctxRef.current) {
      try {
        ctxRef.current = new (window.AudioContext ||
          (window as typeof window & { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      } catch {
        return null;
      }
    }
    // Resume suspended context (browser autoplay policy)
    if (ctxRef.current.state === "suspended") {
      ctxRef.current.resume().catch(() => {});
    }
    return ctxRef.current;
  }, []);

  /** Soft whoosh when a card is played */
  const play = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    playTone(ctx, 520, 0.12, 0, "triangle", 0.12);
    playTone(ctx, 660, 0.10, 0.06, "triangle", 0.08);
  }, [getCtx]);

  /** Rising tension when bluff is called */
  const callBluff = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    playTone(ctx, 400, 0.08, 0, "sawtooth", 0.10);
    playTone(ctx, 500, 0.08, 0.07, "sawtooth", 0.12);
    playTone(ctx, 640, 0.14, 0.14, "sawtooth", 0.14);
  }, [getCtx]);

  /** Punchy descending chord — challenger caught a bluff */
  const bluffCaught = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    playTone(ctx, 880, 0.15, 0, "square", 0.14);
    playTone(ctx, 660, 0.15, 0.08, "square", 0.12);
    playTone(ctx, 440, 0.20, 0.16, "sine", 0.16);
  }, [getCtx]);

  /** Sad low tone — wrong call, pile absorbed */
  const wrongCall = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    playTone(ctx, 300, 0.25, 0, "sine", 0.18);
    playTone(ctx, 240, 0.30, 0.12, "sine", 0.14);
  }, [getCtx]);

  /** Ascending fanfare — human wins */
  const win = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    [523, 659, 784, 1047].forEach((freq, i) => {
      playTone(ctx, freq, 0.18, i * 0.10, "triangle", 0.18);
    });
  }, [getCtx]);

  /** Descending diminished — human loses */
  const lose = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    [440, 370, 311, 261].forEach((freq, i) => {
      playTone(ctx, freq, 0.22, i * 0.12, "sine", 0.15);
    });
  }, [getCtx]);

  /** Neutral flat tone — draw */
  const draw = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    playTone(ctx, 440, 0.40, 0, "sine", 0.12);
    playTone(ctx, 440, 0.30, 0.45, "sine", 0.08);
  }, [getCtx]);

  /** Soft tick — pass / draw card */
  const pass = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    playTone(ctx, 800, 0.06, 0, "triangle", 0.08);
  }, [getCtx]);

  return { play, callBluff, bluffCaught, wrongCall, win, lose, draw, pass };
}
