"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { MotionConfig, motion, AnimatePresence } from "motion/react";
import Balatro from "@/components/Balatro";
import RulesModal from "@/components/RulesModal";
import ProfileMemorySettings from "@/components/PrivacyPreferencesModal";
import { useGameSounds } from "@/hooks/useGameSounds";


// ---------------------------------------------------------------------------
// Env config
// ---------------------------------------------------------------------------

const API_URL = process.env.NEXT_PUBLIC_API_URL === "same-origin"
  ? "/api"
  : process.env.NEXT_PUBLIC_API_URL || "http://localhost:8768";
const LOCAL_HISTORY_PREFIX = "bluff-public-history:v1:";

function getWebSocketUrl(path: string) {
  const url = new URL(`${API_URL}${path}`, window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

function createHistoryId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// ---------------------------------------------------------------------------
// Game layout wrapper
// ---------------------------------------------------------------------------

function GameLayout({
  children,
  username,
  onLogout,
  onOpenRules,
  onOpenPrivacy,
  showNav = true,
}: {
  children: React.ReactNode;
  username: string;
  onLogout: () => void;
  onOpenRules?: () => void;
  onOpenPrivacy: () => void;
  showNav?: boolean;
}) {

  return (
    <MotionConfig reducedMotion="user">
    <div className="relative flex h-dvh w-full flex-col overflow-hidden bg-ctp-base">
      {/* Full-page Balatro shader background */}
      <div className="pointer-events-none absolute inset-0 opacity-25" aria-hidden="true">
        <Balatro
          color1="#c6a4e4"
          color2="#1c1722"
          color3="#0d0a12"
          spinSpeed={0.18}
          contrast={2.5}
          lighting={0.35}
          spinAmount={0.22}
          mouseInteraction={true}
        />
      </div>
      <div className="pointer-events-none ambient-scrim absolute inset-0" aria-hidden="true" />

      {/* Header */}
      <header className="z-20 flex h-14 shrink-0 items-center justify-between border-b border-ctp-surface1/70 bg-ctp-crust/90 px-3 backdrop-blur-xl sm:px-6">
        <div className="flex items-center gap-3">
          <span className="editorial-display text-3xl leading-none text-ctp-peach">
            Bluff
          </span>
        </div>
        <div className="flex items-center gap-2 sm:gap-3">
          {showNav && (
            <div className="flex items-center gap-1">
              <button
                onClick={onOpenRules}
                aria-label="Rules"
                className="flex min-h-11 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-ctp-subtext0 transition-colors hover:bg-ctp-surface0 hover:text-ctp-peach"
              >

                <span>Rules</span>
              </button>
              <button
                onClick={onOpenPrivacy}
                title="Data and learning preferences"
                className="min-h-11 rounded-full px-3 text-xs font-medium text-ctp-subtext0 transition-colors hover:bg-ctp-surface0 hover:text-ctp-peach"
              >
                Data
              </button>
              <a
                href="https://github.com/oGhostyyy/Bluff"
                target="_blank"
                rel="noopener noreferrer"
                className="hidden min-h-11 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-ctp-subtext0 transition-colors hover:bg-ctp-surface0 hover:text-ctp-peach sm:flex"
              >
                <span>Repo</span>
              </a>
              <a
                href="https://linktr.ee/oGhostyyy"
                target="_blank"
                rel="noopener noreferrer"
                className="hidden min-h-11 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-ctp-subtext0 transition-colors hover:bg-ctp-surface0 hover:text-ctp-peach sm:flex"
              >
                <span>Linktree</span>
              </a>
            </div>
          )}
          <div className="flex items-center gap-2">
            <span className="max-w-[64px] truncate text-xs font-semibold text-ctp-peach sm:max-w-[100px]">
              {username}
            </span>
            <button
              onClick={onLogout}
              title="Leave game"
              className="min-h-11 rounded-full border border-ctp-surface1 bg-ctp-surface0/70 px-3 text-xs text-ctp-subtext0 transition-colors hover:border-ctp-red/50 hover:bg-ctp-red/10 hover:text-ctp-red"
            >
              <span className="sm:hidden">Leave</span>
              <span className="hidden sm:inline">Leave game</span>
            </button>
          </div>
        </div>
      </header>

      {children}
    </div>
    </MotionConfig>
  );
}

// ---------------------------------------------------------------------------
// Bot selector
// ---------------------------------------------------------------------------

const BOT_OPTIONS = [
  {
    id: "flagship",
    name: "Pranjol",
    role: "Adaptive Flagship",
    description: "Tracks your calls and revealed bluffs within this room, including rematches. Cross-game guest memory is used only when enabled in Data and confirmed by the server.",
  },
  {
    id: "math",
    name: "Sanja",
    role: "Math baseline",
    description: "Uses approximate card-count beliefs and short-horizon card-advantage estimates. Assumes a 30% bluff prior and a 35% call prior. Not optimal and not fitted to human data.",
  },
  {
    id: "honest",
    name: "Sudhnashu",
    role: "Honest baseline",
    description: "Never bluffs. Calls only when card knowledge proves a bluff; otherwise plays truthfully or passes.",
  },
  {
    id: "random",
    name: "Tanmoy",
    role: "Random baseline",
    description: "Chooses among legal actions at random.",
  },
] as const;

function BotSelector({
  onSelect,
}: {
  onSelect: (botId: string) => void;
}) {
  return (
    <div className="flex w-full flex-col items-center gap-9 py-9 sm:py-12">
      <motion.div
        className="px-5 text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="editorial-display-tight text-5xl leading-[0.98] text-ctp-text sm:text-6xl">
          Choose your opponent
        </h1>
        <p className="mx-auto mt-3 max-w-lg text-sm leading-6 text-ctp-subtext0">
          One adaptive opponent and three baselines.
        </p>
      </motion.div>

      <div className="grid w-full max-w-6xl grid-cols-1 gap-4 px-4 sm:grid-cols-2 sm:gap-5 xl:grid-cols-3">
        {BOT_OPTIONS.map((bot, i) => (
          <motion.button
            key={bot.id}
            onClick={() => onSelect(bot.id)}
            className={`bot-choice group relative cursor-pointer rounded-2xl border p-5 text-left hover:-translate-y-0.5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ctp-peach ${
              bot.id === "flagship"
                ? "bot-choice-featured marble-panel sm:col-span-2 sm:p-7 xl:col-span-3"
                : "surface-card"
            }`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 + i * 0.08 }}
          >
            <div className="mb-5 flex items-center justify-between gap-3">
              <p className={`eyebrow ${bot.id === "flagship" ? "text-ctp-peach" : "text-ctp-lavender"}`}>
                {bot.role}
              </p>
            </div>
            <div className="flex min-w-0 flex-1 items-start sm:items-center">
              <div className={`min-w-0 flex-1 ${bot.id === "flagship" ? "max-w-3xl" : ""}`}>
                <p className="bot-choice-title text-ctp-text">
                  {bot.name}
                </p>
                <p className="mt-3 max-w-2xl text-sm leading-6 text-ctp-subtext1">
                  {bot.description}
                </p>
              </div>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Suit = "♠" | "♥" | "♦" | "♣";
type Rank =
  | "2"
  | "3"
  | "4"
  | "5"
  | "6"
  | "7"
  | "8"
  | "9"
  | "10"
  | "J"
  | "Q"
  | "K"
  | "A";

const SUIT_LABELS: Record<Suit, string> = {
  "♠": "spades",
  "♥": "hearts",
  "♦": "diamonds",
  "♣": "clubs",
};

interface CardData {
  suit: Suit;
  rank: Rank;
}

interface LogEntry {
  time: string;
  text: string;
  kind?: "bluff" | "honest" | "normal" | "draw";
}

interface GameOverResult {
  humanWon: boolean;
  isDraw: boolean;
  message: string;
}

interface BotAdaptation {
  scope?: string;
  profile_status?: string;
  persistent_profiles_available?: boolean;
  research_logging?: boolean;
  log_status?: string;
}

function guestStatusText(adaptation: BotAdaptation) {
  const profileStatus = adaptation.profile_status ?? "unavailable";
  const profile = profileStatus === "saved"
    ? "Guest profile status: saved by the server."
    : profileStatus === "save_failed"
      ? "Guest profile status: save failed."
      : profileStatus === "not_consented"
        ? "Guest profile status: not saved because consent is off."
        : profileStatus === "unavailable"
          ? "Guest profile status: unavailable."
          : `Guest profile status: ${profileStatus}.`;
  const research = adaptation.log_status === "recording"
    ? "Research log status: recording."
    : adaptation.log_status === "saved"
      ? "Research log status: saved by the server."
      : adaptation.log_status === "not_consented"
        ? "Research log status: not saved because consent is off."
        : adaptation.log_status === "save_failed"
          ? "Research log status: save failed."
          : adaptation.log_status === "unavailable"
            ? "Research log status: unavailable."
            : `Research log status: ${adaptation.log_status ?? "not reported"}.`;
  return `${profile} ${research}`;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RANKS: Rank[] = [
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "10",
  "J",
  "Q",
  "K",
  "A",
];

const RANK_LABELS: Record<Rank, string> = {
  "2": "Twos",
  "3": "Threes",
  "4": "Fours",
  "5": "Fives",
  "6": "Sixes",
  "7": "Sevens",
  "8": "Eights",
  "9": "Nines",
  "10": "Tens",
  J: "Jacks",
  Q: "Queens",
  K: "Kings",
  A: "Aces",
};

const RED_SUITS: Set<Suit> = new Set(["♥", "♦"]);

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

// Hand is populated by backend game_state message (no mock data).

// Game log is now driven entirely by backend game_state messages (no mock data).

// ---------------------------------------------------------------------------
// PlayingCard
// ---------------------------------------------------------------------------

function PlayingCard({
  card,
  faceDown = false,
  size = "normal",
  selected = false,
  onClick,
}: {
  card?: CardData;
  faceDown?: boolean;
  size?: "small" | "normal";
  selected?: boolean;
  onClick?: () => void;
}) {
  const isSmall = size === "small";

  if (faceDown || !card) {
    return (
      <div
        className={[
          "rounded-lg border border-ctp-surface1 bg-ctp-surface0",
          isSmall ? "h-[56px] w-[40px]" : "h-[126px] w-[90px] max-md:h-[91px] max-md:w-[65px]",
        ].join(" ")}
        style={{
          backgroundImage: `
            repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(223,193,132,0.06) 4px, rgba(223,193,132,0.06) 5px),
            repeating-linear-gradient(-45deg, transparent, transparent 4px, rgba(223,193,132,0.06) 4px, rgba(223,193,132,0.06) 5px)
          `,
        }}
      >
        {!isSmall && (
          <div className="flex h-full items-center justify-center opacity-[0.08]">
            <span className="text-[28px] text-ctp-peach">♠</span>
          </div>
        )}
      </div>
    );
  }

  const red = RED_SUITS.has(card.suit);

  return (
    <button
      type="button"
      aria-label={`${card.rank} of ${SUIT_LABELS[card.suit]}${selected ? ", selected" : ""}`}
      aria-pressed={selected}
      onClick={onClick}
      className={[
        "playing-card-face rounded-lg border transition-transform duration-200",
        isSmall ? "h-[56px] w-[40px]" : "h-[126px] w-[90px] max-md:h-[91px] max-md:w-[65px]",
        red ? "card-red" : "card-ink",
        selected ? "is-selected" : "",
        "cursor-pointer",
      ].join(" ")}
    >
      <div className="flex h-full flex-col justify-between p-1.5">
        {/* top-left corner */}
        <div className="flex flex-col items-center leading-none">
          <span className={isSmall ? "text-[9px]" : "max-md:text-[10px] text-[13px]"}>
            {card.rank}
          </span>
          <span className={isSmall ? "text-[9px]" : "max-md:text-[11px] text-[14px]"}>
            {card.suit}
          </span>
        </div>

        {/* center suit */}
        <div className="flex items-center justify-center">
          <span className={isSmall ? "text-[16px]" : "max-md:text-[22px] text-[30px]"}>
            {card.suit}
          </span>
        </div>

        {/* bottom-right corner (inverted) */}
        <div className="flex flex-col items-center leading-none rotate-180">
          <span className={isSmall ? "text-[9px]" : "max-md:text-[10px] text-[13px]"}>
            {card.rank}
          </span>
          <span className={isSmall ? "text-[9px]" : "max-md:text-[11px] text-[14px]"}>
            {card.suit}
          </span>
        </div>
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// OpponentArea (top ~20%)
// ---------------------------------------------------------------------------

function OpponentArea({ cardCount, botName, botRole, lastBy, lastCount, lastRank }: { cardCount: number; botName: string; botRole: string; lastBy?: string; lastCount?: number; lastRank?: string }) {
  const visibleCards = Math.min(cardCount, 7);

  return (
    <div className="opponent-stage flex flex-col items-center gap-2 px-4 py-3 md:py-4">
      {/* avatar + name */}
      <div className="flex items-center gap-3">
        <div className="text-center sm:text-left">
          <p className="game-bot-name leading-tight text-ctp-text">Bot ({botName})</p>
          <p className="text-[11px] text-ctp-overlay1">{botRole}</p>
          <p
            className={`text-[13px] ${
              cardCount <= 5 ? "text-ctp-red" : "text-ctp-subtext0"
            }`}
          >
            {cardCount} card{cardCount !== 1 ? "s" : ""} left
          </p>
        </div>
      </div>

      {/* fanned face-down cards */}
      <div className="opponent-card-fan flex items-end pt-1">
        {Array.from({ length: visibleCards }).map((_, i) => {
          const mid = (visibleCards - 1) / 2;
          const maxAngle = Math.min(20, visibleCards * 3);
          const rot = visibleCards > 1 ? (i - mid) * (maxAngle * 2) / (visibleCards - 1) : 0;
          const archY = visibleCards > 1 ? 18 * (1 - Math.cos(((i - mid) / mid) * (Math.PI / 2))) : 0;
          return (
            <div
              key={i}
              className="-ml-[12px] first:ml-0 relative"
              style={{
                transform: `translateY(${archY}px)`,
                transformOrigin: "bottom center",
                zIndex: i,
              }}
            >
              <div style={{ transform: `rotate(${rot}deg)`, transformOrigin: "bottom center" }}>
                <PlayingCard faceDown size="small" />
              </div>
            </div>
          );
        })}
      </div>

      {/* last action */}
      <p className="text-[13px] italic text-ctp-subtext0">
        {lastBy ? `${lastBy} played ${lastCount} card${lastCount !== 1 ? "s" : ""} as ${lastRank}` : "Awaiting first play"}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CenterTable (middle ~50%)
// ---------------------------------------------------------------------------

function CenterTable({
  round,
  playerCount,
  botCount,
  deckCount,
  claimedRank,
  claimedBy,
  playedCount,
  discardCount,
  turnLabel,
}: {
  round: number;
  playerCount: number;
  botCount: number;
  deckCount: number;
  claimedRank: Rank;
  claimedBy: "You" | "Bot" | null;
  playedCount: number;
  discardCount: number;
  turnLabel: string;
}) {
  return (
    <div className="game-table-surface relative mx-3 my-2 flex min-h-[220px] w-[calc(100%-1.5rem)] max-w-6xl flex-1 flex-col items-center justify-center gap-2 self-center rounded-3xl border px-4 py-2">


      {/* radial glow overlay */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 50%, rgba(223,193,132,0.06) 0%, transparent 70%)",
        }}
      />

      <p role="status" aria-live="polite" className="relative z-10 mb-2 text-sm font-semibold text-ctp-peach">
        {turnLabel}
      </p>

      {/* status bar */}
      <motion.div
        className="game-status-pill relative z-10 flex shrink-0 max-w-[calc(100vw-24px)] items-center gap-2 rounded-xl border px-3 py-2.5 text-xs shadow-lg sm:gap-3 sm:rounded-full sm:px-5 sm:text-sm md:gap-4"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15 }}
      >
        <span className="text-ctp-subtext0">Round {round}</span>
        <span className="h-3 w-px bg-ctp-surface2" />
        <span className="font-medium text-ctp-peach">You: {playerCount}</span>
        <span className="h-3 w-px bg-ctp-surface2" />
        <span className="font-medium text-ctp-lavender">Bot: {botCount}</span>
        <span className="h-3 w-px bg-ctp-surface2" />
        <span className="text-ctp-subtext0">Deck: {deckCount}</span>
      </motion.div>

      {/* claimed rank */}
      <motion.div
        className="relative z-10 shrink-0 text-center"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.25 }}
      >
        <p
          className="editorial-display text-4xl text-ctp-peach sm:text-5xl"
          style={{
            textShadow: "0 0 30px rgba(223,193,132,0.15)",
          }}
        >
          {claimedBy ? RANK_LABELS[claimedRank] : "Awaiting first play"}
        </p>
        <p className="text-sm text-ctp-subtext0">
          {claimedBy ? `Claimed by ${claimedBy}` : "No cards played yet"}
        </p>
      </motion.div>

      {/* played cards fan */}
      <motion.div
        className="center-packet-visual relative z-10 flex shrink-0 items-end"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.35 }}
      >
        {Array.from({ length: playedCount }).map((_, i) => {
          const mid = (playedCount - 1) / 2;
          const rot = (i - mid) * 7;
          return (
            <div
              key={i}
              className="-ml-[12px] first:ml-0"
              style={{
                transform: `rotate(${rot}deg)`,
                transformOrigin: "bottom center",
              }}
            >
              <PlayingCard faceDown size="small" />
            </div>
          );
        })}
      </motion.div>

      {/* discard pile */}
      <motion.div
        className="relative z-10 flex shrink-0 items-center gap-3 pt-1"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.45 }}
      >
        {/* stacked mini cards */}
        <div className="relative h-[32px] w-[22px]">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="absolute rounded border border-ctp-surface1 bg-ctp-surface0"
              style={{
                width: 22,
                height: 32,
                top: -i * 2,
                left: i * 1.5,
              }}
            />
          ))}
        </div>
        <span className="text-xs text-ctp-subtext0">
          Center pile: {discardCount} cards
        </span>
      </motion.div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// RankSelector
// ---------------------------------------------------------------------------

function RankSelector({
  selected,
  onSelect,
  onConfirm,
  onCancel,
  activeRank,
}: {
  selected: Rank | null;
  onSelect: (r: Rank) => void;
  onConfirm: () => void;
  onCancel: () => void;
  activeRank: Rank | null;
}) {
  const availableRanks = activeRank ? [activeRank] : RANKS;
  return (
    <motion.div
      className="flex flex-col items-center gap-3"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 12 }}
      transition={{ duration: 0.25 }}
    >
      <p className="text-xs font-medium uppercase tracking-[0.15em] text-ctp-subtext0">
        {activeRank ? `Fixed round rank: ${activeRank}` : "Choose this round's rank"}
      </p>

      <div className="flex flex-wrap justify-center gap-1.5">
        {availableRanks.map((rank) => (
          <button
            key={rank}
            onClick={() => onSelect(rank)}
            className={[
              "flex h-11 w-11 items-center justify-center rounded-lg border text-sm font-medium transition-all duration-150",
              "md:h-12 md:w-12",
              selected === rank
                ? "border-ctp-peach bg-ctp-peach text-ctp-base shadow-[0_0_12px_rgba(223,193,132,0.2)]"
                : "border-ctp-surface1 bg-ctp-surface0 text-ctp-text hover:border-ctp-overlay1 hover:bg-ctp-surface1",
            ].join(" ")}
          >
            {rank}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={onCancel}
          className="min-h-11 rounded-full border border-ctp-surface1 bg-transparent px-4 py-2 text-[13px] text-ctp-subtext0 transition-all hover:border-ctp-overlay1 hover:text-ctp-text"
        >
          Cancel
        </button>
        <button
          onClick={onConfirm}
          disabled={!selected}
          className="button-primary min-h-11 rounded-full px-6 py-2 text-[13px] font-semibold"
        >
          Confirm Play
        </button>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// PlayerHand (bottom ~30%)
// ---------------------------------------------------------------------------

function PlayerHand({
  hand,
  selected,
  onToggle,
  showSelector,
  selectorRank,
  onSelectRank,
  onOpenSelector,
  onConfirmPlay,
  onCancelPlay,
  onCallBluff,
  onPass,
  canPlay,
  canPass,
  canCallBluff,
  activeRank,
}: {
  hand: CardData[];
  selected: Set<number>;
  onToggle: (i: number) => void;
  showSelector: boolean;
  selectorRank: Rank | null;
  onSelectRank: (r: Rank) => void;
  onOpenSelector: () => void;
  onConfirmPlay: () => void;
  onCancelPlay: () => void;
  onCallBluff: () => void;
  onPass: () => void;
  canPlay: boolean;
  canPass: boolean;
  canCallBluff: boolean;
  activeRank: Rank | null;
}) {
  const n = hand.length;

  return (
    <div className="hand-dock-surface relative z-20 flex w-full min-w-0 flex-col items-center gap-3 border-t px-4 pb-4 pt-2 shadow-[0_-16px_40px_rgba(7,7,15,0.2)] backdrop-blur-xl md:pb-6">
      {/* Flat row avoids transformed cards being clipped by the scroll area. */}
      <div role="group" aria-label="Your hand" tabIndex={0} className="w-full max-w-full overflow-x-auto overscroll-x-contain pt-6 pb-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ctp-peach">
        <div className="mx-auto flex w-max min-w-full items-end justify-center gap-2 px-6">
          {hand.map((card, i) => {
            const isSelected = selected.has(i);

            return (
              <div
                key={i}
                className="relative shrink-0"
                style={{
                  zIndex: isSelected ? n + 1 : i,
                }}
              >
                <div>
                  <div
                    className="transition-transform duration-200 hover:-translate-y-2"
                    style={{
                      transform: isSelected ? "translateY(-20px)" : undefined,
                    }}
                  >
                    <PlayingCard
                      card={card}
                      selected={isSelected}
                      onClick={() => onToggle(i)}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* action buttons */}
      {canPlay && (
        <p role="status" aria-live="polite" className="max-w-2xl text-center text-sm text-ctp-subtext0">
          {selected.size === 0
            ? "Your turn: choose Play, Call Bluff, or Pass when available. Select 1–4 cards to play."
            : `${selected.size} card${selected.size === 1 ? "" : "s"} selected. Choose Play Cards to declare a rank.`}
        </p>
      )}
      {canCallBluff && !canPlay && (
        <p role="status" aria-live="polite" className="max-w-2xl text-center text-sm text-ctp-subtext0">
          {canPass
            ? "Choose to play, challenge the latest packet, or pass and draw 1 card."
            : "Choose to play or challenge the latest packet."}
        </p>
      )}
      <div className="flex w-full flex-wrap items-center justify-center gap-2.5 px-2 sm:gap-3">
        <button
          onClick={onOpenSelector}
          disabled={(selected.size === 0 && !showSelector) || !canPlay}
          className="button-primary flex min-h-11 items-center justify-center gap-2 rounded-full px-5 text-sm font-semibold"
        >
          Play Cards
        </button>

        <button
          onClick={onCallBluff}
          disabled={!canCallBluff}
          className="button-danger flex min-h-11 items-center justify-center gap-2 rounded-full border-2 px-5 text-sm font-semibold"
        >
          Call Bluff!
        </button>
        <div className="relative group">
          <button
            onClick={onPass}
            disabled={!canPass}
            className="button-secondary flex min-h-11 items-center justify-center gap-2 rounded-full px-5 text-sm font-medium"
          >
            Pass
          </button>
          {!canPass && canCallBluff && (
            <div className="pointer-events-none absolute -top-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-lg border border-ctp-surface1/60 bg-ctp-mantle/95 px-2.5 py-1 text-[11px] text-ctp-overlay1 opacity-0 shadow-lg backdrop-blur-sm transition-opacity group-hover:opacity-100">
              Passing unavailable. Play or call bluff.
            </div>
          )}
        </div>

      </div>

      {/* rank selector */}
      <AnimatePresence>
        {showSelector && (
          <RankSelector
            selected={selectorRank}
            onSelect={onSelectRank}
            onConfirm={onConfirmPlay}
            onCancel={onCancelPlay}
            activeRank={activeRank}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

// ---------------------------------------------------------------------------
// GameOverOverlay
// ---------------------------------------------------------------------------

function GameOverOverlay({
  result,
  adaptation,
  onPlayAgain,
}: {
  result: GameOverResult;
  adaptation: BotAdaptation | null;
  onPlayAgain: () => void;
}) {
  const { humanWon, isDraw, message } = result;

  const headline = isDraw
    ? "Draw!"
    : humanWon
      ? "You Won!"
      : "Bot Wins!";

  const subColor = isDraw
    ? "text-ctp-yellow"
    : humanWon
      ? "text-ctp-green"
      : "text-ctp-red";

  const glowColor = isDraw
    ? "shadow-[0_0_60px_rgba(234,210,147,0.12)]"
    : humanWon
      ? "shadow-[0_0_60px_rgba(159,190,161,0.12)]"
      : "shadow-[0_0_60px_rgba(225,140,145,0.12)]";

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      {/* backdrop */}
      <div className="absolute inset-0 bg-ctp-crust/70 backdrop-blur-md" />

      {/* panel */}
      <motion.div
        className={`relative z-10 w-[320px] max-w-[90vw] rounded-2xl border border-ctp-surface1/60 bg-ctp-mantle/95 p-6 text-center ${glowColor}`}
        initial={{ scale: 0.88, y: 20, opacity: 0 }}
        animate={{ scale: 1, y: 0, opacity: 1 }}
        transition={{ type: "spring", damping: 22, stiffness: 260, delay: 0.05 }}
      >
        {/* result headline */}
        <p className={`editorial-display text-4xl leading-tight ${subColor}`}>
          {headline}
        </p>
        <p className="mt-2 text-[13px] text-ctp-subtext0 leading-snug">
          {message}
        </p>

        {/* bot adaptation summary if available */}
        {adaptation && (
          <div className="mt-4 rounded-xl border border-ctp-lavender/25 bg-ctp-surface0/60 p-3 text-left">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-ctp-lavender">
              Prototype data scope
            </p>
            <p className="text-xs text-ctp-subtext0">{guestStatusText(adaptation)}</p>
          </div>
        )}

        {/* play again */}
        <button
          onClick={onPlayAgain}
          className="mt-5 w-full rounded-full bg-ctp-peach py-2.5 text-[14px] font-semibold text-ctp-base transition-all hover:brightness-110 hover:-translate-y-0.5 active:translate-y-0"
        >
          Play Again
        </button>
      </motion.div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// GameLogSidebar
// ---------------------------------------------------------------------------

function GameLogSidebar({
  open,
  collapsed,
  onClose,
  logs,
  adaptation,
}: {

  open: boolean;
  collapsed: boolean;
  onClose: () => void;
  logs: LogEntry[];
  adaptation?: BotAdaptation | null;
}) {
  return (
    <>
      {/* ── Desktop sidebar ── */}
      <motion.aside
        className="game-log-surface hidden h-full shrink-0 flex-col overflow-hidden border-l border-ctp-surface1/70 backdrop-blur-xl md:flex"
        animate={{ width: collapsed ? 0 : 280, opacity: collapsed ? 0 : 1 }}
        transition={{ duration: 0.2, ease: "easeInOut" }}
      >
        <div className="flex items-center justify-between border-b border-ctp-surface1/60 px-4 py-3 w-[280px]">
          <div>
            <p className="editorial-section-title text-ctp-text">Game log</p>
            <p className="text-xs text-ctp-overlay1">Latest at the top</p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close game log"
            className="min-h-11 rounded-xl px-3 text-xs text-ctp-overlay0 transition-colors hover:bg-ctp-surface0 hover:text-ctp-text"
          >
            Close
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-3 w-[280px]">
          {adaptation && (
            <div className="mb-3 rounded-lg border border-ctp-lavender/30 bg-ctp-surface0/60 p-2.5 text-left">
              <div className="pb-1.5 border-b border-ctp-surface1/40">
                <span className="text-xs font-semibold uppercase tracking-wider text-ctp-lavender">
                  Session memory
                </span>
              </div>
              <p className="mt-2 text-xs text-ctp-subtext0">{guestStatusText(adaptation)}</p>
            </div>
          )}

          <div className="space-y-3">
            {logs.map((entry, i) => (
              <div key={i} className="game-log-entry">
                <span className="block text-xs text-ctp-subtext0">
                  {entry.time}
                </span>
                <p
                  className={[
                    "text-sm leading-5",
                    entry.kind === "bluff"
                      ? "font-semibold text-ctp-red"
                      : entry.kind === "honest"
                        ? "text-ctp-green"
                        : entry.kind === "draw"
                          ? "font-semibold text-ctp-yellow"
                          : "text-ctp-subtext0",
                  ].join(" ")}
                >
                  {entry.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      </motion.aside>

      {/* ── Mobile overlay ── */}
      <AnimatePresence>
        {open && (
          <motion.div
            className="fixed inset-0 z-50 md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* backdrop */}
            <button
              className="absolute inset-0 bg-ctp-crust/60 backdrop-blur-sm"
              onClick={onClose}
              aria-label="Close log"
            />

            {/* panel */}
            <motion.aside
              className="game-log-surface absolute right-0 top-14 flex h-[calc(100dvh-3.5rem)] w-[280px] flex-col border-l border-ctp-surface1"
              initial={{ x: 280 }}
              animate={{ x: 0 }}
              exit={{ x: 280 }}
              transition={{ type: "spring", damping: 28, stiffness: 320 }}
            >
              <div className="flex items-center justify-between border-b border-ctp-surface1 px-4 py-3">
                <div>
                  <p className="editorial-section-title text-ctp-text">Game log</p>
                  <p className="text-xs text-ctp-overlay1">Latest at the top</p>
                </div>
                <button
                  onClick={onClose}
                  aria-label="Close game log"
                  className="min-h-11 rounded-xl px-3 text-xs text-ctp-overlay1 transition-colors hover:bg-ctp-surface0 hover:text-ctp-text"
                >
                  Close
                </button>
              </div>

              <div className="flex-1 overflow-y-auto px-4 py-3">
                {adaptation && (
                  <div className="mb-3 rounded-lg border border-ctp-lavender/30 bg-ctp-surface1/60 p-2.5 text-left">
                    <div className="flex items-center justify-between pb-1.5 border-b border-ctp-surface2/40">
                      <span className="text-xs font-semibold uppercase tracking-wider text-ctp-lavender">
                        Session memory
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-ctp-subtext0">{guestStatusText(adaptation)}</p>
                  </div>
                )}

                <div className="space-y-3">
                  {logs.map((entry, i) => (
                    <div key={i}>
                      <span className="block text-xs text-ctp-subtext0">
                        {entry.time}
                      </span>
                      <p
                        className={[
                          "text-sm leading-5",
                          entry.kind === "bluff"
                            ? "font-semibold text-ctp-red"
                            : entry.kind === "honest"
                              ? "text-ctp-green"
                              : entry.kind === "draw"
                                ? "font-semibold text-ctp-yellow"
                                : "text-ctp-subtext0",
                        ].join(" ")}
                      >
                        {entry.text}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.aside>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function GamePage() {
  return <GameSession />;
}

function GameSession() {
  const router = useRouter();
  const sounds = useGameSounds();

  const username = "You";

  // bot selection
  const [selectedBot, setSelectedBot] = useState<string | null>(null);
  const [botSelected, setBotSelected] = useState(false);

  // game state
  const [hand, setHand] = useState<CardData[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [showSelector, setShowSelector] = useState(false);
  const [selectorRank, setSelectorRank] = useState<Rank | null>(null);
  const [botCards, setBotCards] = useState(14);
  const [round, setRound] = useState(1);
  const [activeRank, setActiveRank] = useState<Rank | null>(null);
  const [pileCount, setPileCount] = useState(0);
  const [deckCount, setDeckCount] = useState(24); // HOTFIX Tess 2026-09-11: was 28, actual is 24 (52-14*2); WS corrects it anyway
  const [claimedRank, setClaimedRank] = useState<Rank>("7");
  const [claimedBy, setClaimedBy] = useState<"You" | "Bot" | null>(null);
  const [playedCount, setPlayedCount] = useState(0);
  const [canCallBluff, setCanCallBluff] = useState(false);
  const [canPlay, setCanPlay] = useState(false);
  const [canPass, setCanPass] = useState(false);
  const [movePending, setMovePending] = useState(false);
  const [adaptation, setAdaptation] = useState<BotAdaptation | null>(null);

  const [logOpen, setLogOpen] = useState(false);
  const [logCollapsed, setLogCollapsed] = useState(false);
  const [rulesOpen, setRulesOpen] = useState(false);
  const [privacyOpen, setPrivacyOpen] = useState(false);

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [historyId, setHistoryId] = useState("");
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [gameOver, setGameOver] = useState<GameOverResult | null>(null);

  useEffect(() => {
    if (!historyId) return;
    try {
      localStorage.setItem(`${LOCAL_HISTORY_PREFIX}${historyId}`, JSON.stringify({
        saved_at: Date.now(),
        entries: logs.slice(0, 200),
      }));
      const histories: { key: string; savedAt: number }[] = [];
      for (let index = 0; index < localStorage.length; index += 1) {
        const key = localStorage.key(index);
        if (!key?.startsWith(LOCAL_HISTORY_PREFIX)) continue;
        const stored: unknown = JSON.parse(localStorage.getItem(key) || "null");
        const savedAt = typeof stored === "object" && stored !== null && "saved_at" in stored && typeof stored.saved_at === "number" ? stored.saved_at : 0;
        histories.push({ key, savedAt });
      }
      histories.sort((a, b) => b.savedAt - a.savedAt);
      for (const old of histories.slice(20)) localStorage.removeItem(old.key);
    } catch {
      // Local public history is best effort; gameplay does not depend on it.
    }
  }, [historyId, logs]);


  // ── handlers ──
  const MAX_PLAY = 4;
  const toggleCard = useCallback((idx: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else if (next.size < MAX_PLAY) next.add(idx);
      return next;
    });
  }, []);

  const handlePlayClick = useCallback(() => {
    if (selected.size === 0) return;
    setSelectorRank(activeRank);
    setShowSelector(true);
  }, [selected.size, activeRank]);

  const handleConfirmPlay = useCallback(() => {
    const rank = activeRank ?? selectorRank;
    if (!rank || !canPlay || movePending) return;

    const selectedIndices = Array.from(selected);

    if (ws && connected && ws.readyState === WebSocket.OPEN) {
      setMovePending(true);
      sounds.play();
      ws.send(
        JSON.stringify({
          action: "play",
          cards: selectedIndices,
          rank,
        })
      );
    }

    setSelected(new Set());
    setShowSelector(false);
    setSelectorRank(null);
  }, [activeRank, selectorRank, selected, ws, connected, sounds, canPlay, movePending]);

  const handleCancelPlay = useCallback(() => {
    setShowSelector(false);
    setSelectorRank(null);
  }, []);

  const handleCallBluff = useCallback(() => {
    if (!canCallBluff || movePending) return;
    if (ws && connected && ws.readyState === WebSocket.OPEN) {
      setMovePending(true);
      sounds.callBluff();
      ws.send(JSON.stringify({ action: "call_bluff" }));
    }
    setSelected(new Set());
    setShowSelector(false);
    setSelectorRank(null);
  }, [ws, connected, sounds, canCallBluff, movePending]);

  const handlePass = useCallback(() => {
    if (!canPass || movePending) return;
    if (ws && connected && ws.readyState === WebSocket.OPEN) {
      setMovePending(true);
      sounds.pass();
      ws.send(JSON.stringify({ action: "pass" }));
    }
    setSelected(new Set());
    setShowSelector(false);
    setSelectorRank(null);
  }, [ws, connected, sounds, canPass, movePending]);


  const handleLogout = useCallback(() => {
    router.push("/");
  }, [router]);

  const handleOpenPrivacy = useCallback(() => {
    setPrivacyOpen(true);
  }, []);

  const handleProfileRevoked = useCallback(() => {
    try {
      const localHistoryKeys: string[] = [];
      for (let index = 0; index < localStorage.length; index += 1) {
        const key = localStorage.key(index);
        if (key?.startsWith(LOCAL_HISTORY_PREFIX)) localHistoryKeys.push(key);
      }
      for (const key of localHistoryKeys) localStorage.removeItem(key);
    } catch {
      // Server data deletion remains complete if browser storage is unavailable.
    }
    setHistoryId("");
    setLogs([]);
    setAdaptation((previous) => ({
      ...previous, profile_status: "not_consented",
      persistent_profiles_available: false, research_logging: false,
      log_status: "not_consented", scope: "room_only",
    }));
  }, []);

  const handleBotSelect = useCallback((botId: string) => {
    setSelectedBot(botId);
    setBotSelected(true);
  }, []);

  const handlePlayAgain = useCallback(() => {
    setMovePending(false);
    // A rematch keeps the room's behavioral memory, not its old card beliefs.
    if (ws && connected && ws.readyState === WebSocket.OPEN) {
      setHistoryId(createHistoryId());
      setLogs([]);
      setGameOver(null);
      setSelected(new Set());
      setShowSelector(false);
      setSelectorRank(null);
      setCanPlay(false);
      setCanCallBluff(false);
      setCanPass(false);
      ws.send(JSON.stringify({ action: "new_game" }));
      return;
    }
    // A disconnected room cannot be resumed; select a fresh opponent instead.
    setGameOver(null);
    if (ws) ws.close();
    setWs(null);
    setConnected(false);
    setBotSelected(false);
    setSelectedBot(null);
    setHand([]);
    setHistoryId("");
    setSelected(new Set());
    setLogs([]);
    setAdaptation(null);
    setPileCount(0);
    setClaimedBy(null);
    setActiveRank(null);
    setPlayedCount(0);
    setDeckCount(24);
    setRound(1);
    setBotCards(14);
    setCanCallBluff(false);
    setCanPlay(false);
    setCanPass(false);
  }, [ws, connected]);

  // ── Connect to WebSocket after bot selected ──
  useEffect(() => {
    if (!botSelected || !selectedBot) return;

    let socket: WebSocket | null = null;
    let isMounted = true;

    async function initWs() {
      try {
        const res = await fetch(`${API_URL}/rooms?bot_name=${selectedBot}`, {
          method: "POST",
          credentials: "include",
        });
        if (!res.ok) {
          const payload = await res.json().catch(() => null);
          throw new Error(payload?.detail || `Room creation failed (${res.status})`);
        }
        const data = await res.json();
        if (!isMounted) return;
        const roomId = data.room_id;
        if (!roomId) return;
        setHistoryId(createHistoryId());
        socket = new WebSocket(getWebSocketUrl(`/ws/${roomId}`));

        socket.onopen = () => {
          if (!isMounted) return;
          setConnected(true);
          setWs(socket);
          const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          setLogs((prev) => [
            { time: now, text: `Connected. Playing ${selectedBot}.`, kind: "normal" },
            ...prev,
          ]);
        };

        socket.onmessage = (event) => {
          if (!isMounted) return;
          try {
            const payload = typeof event.data === "string" ? JSON.parse(event.data) : event.data;
            const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            if (payload.type === "game_state") {
              setMovePending(false);
              setCanPlay(payload.can_play === true);
              if (payload.hand) {
                setHand(
                  payload.hand.map((c: { suit: Suit; rank: Rank }) => ({
                    suit: c.suit as Suit,
                    rank: c.rank as Rank,
                  }))
                );
              }
              if (payload.opponent_hand_size !== undefined) {
                setBotCards(payload.opponent_hand_size);
              }
              if (payload.pile_size !== undefined) {
                setPileCount(payload.pile_size);
              }
              if (payload.draw_pile_size !== undefined) {
                setDeckCount(payload.draw_pile_size);
              }
              if (payload.round_number !== undefined) {
                setRound(payload.round_number);
              }
              const nextActiveRank = RANKS.includes(payload.active_rank as Rank)
                ? payload.active_rank as Rank
                : null;
              setActiveRank(nextActiveRank);
              setSelectorRank(nextActiveRank);
              if (payload.can_call_bluff !== undefined) {
                setCanCallBluff(payload.can_call_bluff);
              }
              if (payload.can_pass !== undefined) {
                setCanPass(payload.can_pass);
              }
              if (payload.last_action) {
                const la = payload.last_action;
                setClaimedRank(la.claimed_rank || "7");
                setClaimedBy(la.player === 0 ? "You" : "Bot");
                // `cards` is populated only once a challenge reveals the play
                // (game-rules.md §5); the played count is always public, so the
                // server sends it separately.
                setPlayedCount(
                  la.cards_played_count ?? (la.cards?.length || 1)
                );
              } else {
                setClaimedBy(null);
                setPlayedCount(0);
              }
              if (payload.resolution_action?.revealed) {
                const resolved = payload.resolution_action;
                const cards = Array.isArray(resolved.cards)
                  ? resolved.cards.map((card: CardData) => `${card.rank}${card.suit}`).join(", ")
                  : "";
                const outcome = resolved.was_bluff ? "was a bluff" : "was truthful";
                setLogs((prev) => [{
                  time: now,
                  text: `Revealed ${resolved.player === 0 ? "your" : "bot's"} ${resolved.cards_played_count}-card ${resolved.claimed_rank} packet (${outcome})${cards ? `: ${cards}` : ""}.`,
                  kind: resolved.was_bluff ? "bluff" : "honest",
                }, ...prev]);
              }
              if (payload.bot_adaptation) {
                setAdaptation(payload.bot_adaptation);
              }
              if (payload.message) {
                const msg = payload.message.toLowerCase();
                const isBluff = msg.includes("bluff");
                const isHonest = msg.includes("honest") || msg.includes("win");
                // Contextual sounds based on message content
                if (msg.includes("caught") || msg.includes("bluff! correct") || msg.includes("was a bluff")) {
                  sounds.bluffCaught();
                } else if (msg.includes("wrong") || msg.includes("honest play") || msg.includes("not a bluff")) {
                  sounds.wrongCall();
                }
                setLogs((prev) => [
                  {
                    time: now,
                    text: payload.message,
                    kind: isBluff ? "bluff" : isHonest ? "honest" : "normal",
                  },
                  ...prev,
                ]);
              }
            } else if (payload.type === "game_over") {
              setMovePending(false);
              setCanPlay(false);
              setActiveRank(null);
              setCanCallBluff(false);
              setCanPass(false);
              // Display storage outcomes reported by the server, not estimates.
              setAdaptation(payload.bot_adaptation ?? null);
              const isDraw = Boolean(payload.draw);
              const humanWon = Boolean(payload.human_won);
              const logKind: LogEntry["kind"] = isDraw ? "draw" : humanWon ? "honest" : "bluff";
              // Game-over sound
              if (isDraw) sounds.draw();
              else if (humanWon) sounds.win();
              else sounds.lose();
              setLogs((prev) => [
                {
                  time: now,
                  text: payload.message || (humanWon ? "You won!" : "Bot wins!"),
                  kind: logKind,
                },
                ...prev,
              ]);
              setGameOver({
                humanWon,
                isDraw,
                message: payload.message || (humanWon ? "You emptied your hand first!" : "Bot emptied its hand first."),
              });

            } else if (payload.type === "error") {
              setMovePending(false);
              setLogs((prev) => [
                { time: now, text: `Error: ${payload.message}`, kind: "bluff" },
                ...prev,
              ]);
            }
          } catch {
            // ignore non-json
          }
        };

        socket.onclose = () => {
          if (!isMounted) return;
          setConnected(false);
          setWs(null);
        };
      } catch (error) {
        if (!isMounted) return;
        const message = error instanceof Error ? error.message : "Backend offline";
        const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        setLogs((prev) => [
          { time: now, text: message, kind: "bluff" },
          ...prev,
        ]);
      }
    }

    initWs();

    return () => {
      isMounted = false;
      if (socket) socket.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [botSelected, selectedBot]);

  // ── loading ──
  // ── bot selection screen ──
  if (!botSelected) {
    return (
      <>
        <GameLayout username={username} onLogout={handleLogout} onOpenRules={() => setRulesOpen(true)} onOpenPrivacy={handleOpenPrivacy} showNav={true}>
          <main id="main-content" className="relative z-10 min-h-0 flex-1 overflow-y-auto">
            <BotSelector onSelect={handleBotSelect} />
          </main>
          <RulesModal isOpen={rulesOpen} onClose={() => setRulesOpen(false)} />
        </GameLayout>
        <ProfileMemorySettings open={privacyOpen} onClose={() => setPrivacyOpen(false)} apiUrl={API_URL} onRevoked={handleProfileRevoked} />
      </>
    );
  }

  return (
    <>
      <GameLayout username={username} onLogout={handleLogout} onOpenRules={() => setRulesOpen(true)} onOpenPrivacy={handleOpenPrivacy}>

      <main id="main-content" className="relative z-10 flex min-h-0 flex-1 flex-col">
      <h1 className="sr-only">Bluff game</h1>
      <div className="flex flex-1 min-h-0 w-full overflow-hidden">
        {/* ── main game area ── */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto">
          {/* opponent - top */}
          <div className="shrink-0">
            <OpponentArea cardCount={botCards} botName={BOT_OPTIONS.find(b => b.id === selectedBot)?.name || "Opponent"} botRole={BOT_OPTIONS.find(b => b.id === selectedBot)?.role || "Baseline"} lastBy={claimedBy ?? undefined} lastCount={playedCount} lastRank={claimedRank} />
          </div>

          {/* center table - middle */}
          <CenterTable
            round={round}
            playerCount={hand.length}
            botCount={botCards}
            deckCount={deckCount}
            claimedRank={claimedRank}
            claimedBy={claimedBy}
            playedCount={playedCount}
            discardCount={pileCount}
            turnLabel={gameOver ? "Game over" : !connected ? "Connecting..." : movePending ? "Submitting your move..." : canPlay ? (activeRank ? "Your turn: play, call bluff, or pass" : "Your turn: play cards and choose a rank") : `${BOT_OPTIONS.find(b => b.id === selectedBot)?.name || "Bot"}'s turn`}
          />

          {/* player hand - bottom */}
          <div className="shrink-0">
            <PlayerHand
              hand={hand}
              selected={selected}
              onToggle={toggleCard}
              showSelector={showSelector}
              selectorRank={selectorRank}
              onSelectRank={setSelectorRank}
              onOpenSelector={handlePlayClick}
              onConfirmPlay={handleConfirmPlay}
              onCancelPlay={handleCancelPlay}
              onCallBluff={handleCallBluff}
              onPass={handlePass}
              canPlay={connected && !movePending && canPlay}
              canPass={connected && !movePending && canPass}
              canCallBluff={connected && !movePending && canCallBluff}
              activeRank={activeRank}
            />
          </div>
        </div>

        {/* ── game log sidebar ── */}
        <GameLogSidebar
          open={logOpen}
          collapsed={logCollapsed}
          onClose={() => setLogCollapsed(true)}
          logs={logs}
          adaptation={adaptation}
        />
      </div>

      {/* ── Rules Modal ── */}
      <RulesModal isOpen={rulesOpen} onClose={() => setRulesOpen(false)} />

      {/* ── log toggle buttons ── */}
      {/* Desktop: toggle sidebar */}
      <button
        onClick={() => setLogCollapsed(!logCollapsed)}
        className="fixed bottom-4 right-4 z-40 hidden min-h-11 items-center justify-center rounded-full border border-ctp-surface1/60 bg-ctp-crust/90 px-4 text-xs text-ctp-subtext0 shadow-lg backdrop-blur-sm transition-colors hover:border-ctp-overlay1 hover:text-ctp-text md:flex"
        aria-label="Toggle game log"
      >
        Game log
      </button>
      {/* Mobile: open overlay */}
      <button
        onClick={() => setLogOpen(true)}
        className="fixed bottom-4 right-4 z-40 flex min-h-11 items-center justify-center rounded-full border border-ctp-surface1/60 bg-ctp-crust/90 px-4 text-xs text-ctp-subtext0 shadow-lg backdrop-blur-sm transition-colors hover:border-ctp-overlay1 hover:text-ctp-text md:hidden"
        aria-label="Open game log"
      >
        Game log
      </button>
      {/* ── Game Over Overlay ── */}
      <AnimatePresence>
        {gameOver && (
          <GameOverOverlay result={gameOver} adaptation={adaptation} onPlayAgain={handlePlayAgain} />
        )}
      </AnimatePresence>
      </main>
      </GameLayout>
      <ProfileMemorySettings open={privacyOpen} onClose={() => setPrivacyOpen(false)} apiUrl={API_URL} onRevoked={handleProfileRevoked} />
    </>
  );
}
