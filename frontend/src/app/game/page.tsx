"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useUser } from "@clerk/nextjs";
import { motion, AnimatePresence } from "motion/react";
import { AlertTriangle, Bot, Play, ScrollText, X, BookOpen, ExternalLink, LogOut } from "lucide-react";
import Balatro from "@/components/Balatro";
import RulesModal from "@/components/RulesModal";
import { useGameSounds } from "@/hooks/useGameSounds";


function GithubIcon({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24">
      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Env config
// ---------------------------------------------------------------------------

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL = API_URL.replace(/^http/, "ws");

// ---------------------------------------------------------------------------
// Game layout wrapper
// ---------------------------------------------------------------------------

function GameLayout({
  children,
  username,
  onLogout,
  onOpenRules,
  showNav = true,
}: {
  children: React.ReactNode;
  username: string;
  onLogout: () => void;
  onOpenRules?: () => void;
  showNav?: boolean;
}) {

  return (
    <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-ctp-base">
      {/* Full-page Balatro shader background */}
      <div className="absolute inset-0 pointer-events-none opacity-25">
        <Balatro
          color1="#fab387"
          color2="#1e1e2e"
          color3="#11111b"
          spinSpeed={3.0}
          contrast={2.5}
          lighting={0.35}
          spinAmount={0.22}
          mouseInteraction={true}
        />
      </div>
      <div className="absolute inset-0 pointer-events-none bg-ctp-base/60" />

      {/* Header */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-ctp-surface1/60 bg-ctp-crust/80 px-4 z-20 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <span className="font-[family-name:var(--font-petit-formal-script)] text-[22px] font-bold text-ctp-peach tracking-wider">
            Bluff
          </span>
        </div>
        <div className="flex items-center gap-2 sm:gap-3">
          {showNav && (
            <div className="flex items-center gap-1">
              <button
                onClick={onOpenRules}
                className="flex items-center gap-1.5 px-2 py-1.5 text-[12px] font-medium text-ctp-subtext0 transition-colors hover:text-ctp-peach"
              >

                <BookOpen className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Rules</span>
              </button>
              <a
                href="https://github.com/oGhostyyy/Bluff"
                target="_blank"
                rel="noopener noreferrer"
                className="hidden sm:flex items-center gap-1.5 px-2 py-1.5 text-[12px] font-medium text-ctp-subtext0 transition-colors hover:text-ctp-peach"
              >
                <GithubIcon className="h-3.5 w-3.5" />
                <span>Repo</span>
              </a>
              <a
                href="https://linktr.ee/oGhostyyy"
                target="_blank"
                rel="noopener noreferrer"
                className="hidden sm:flex items-center gap-1.5 px-2 py-1.5 text-[12px] font-medium text-ctp-subtext0 transition-colors hover:text-ctp-peach"
              >
                <ExternalLink className="h-3.5 w-3.5" />
                <span>Linktree</span>
              </a>
            </div>
          )}
          <div className="flex items-center gap-2">
            <span className="text-[12px] font-semibold text-ctp-peach max-w-[100px] truncate">
              {username}
            </span>
            <button
              onClick={onLogout}
              title="Leave Game"
              className="flex h-7 w-7 items-center justify-center rounded-lg border border-ctp-surface1 bg-ctp-surface0/50 text-ctp-overlay0 transition-colors hover:border-ctp-red/50 hover:bg-ctp-red/10 hover:text-ctp-red"
            >
              <LogOut className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </header>

      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Bot selector
// ---------------------------------------------------------------------------

const BOT_OPTIONS = [
  {
    id: "random",
    name: "Beginner",
    bot: "Random",
    difficulty: 1,
    description: "Random moves — easy win",
    dotColor: "bg-ctp-green",
  },
  {
    id: "honest",
    name: "Easy",
    bot: "Honest",
    difficulty: 2,
    description: "Never bluffs — learnable",
    dotColor: "bg-ctp-yellow",
  },
  {
    id: "cardcount",
    name: "Medium",
    bot: "CardCount",
    difficulty: 3,
    description: "Mathematical play — challenging",
    dotColor: "bg-ctp-peach",
  },
  {
    id: "bayesian",
    name: "Hard",
    bot: "Bayesian",
    difficulty: 4,
    description: "Learns your patterns — tough",
    dotColor: "bg-ctp-red",
  },
  {
    id: "purenn",
    name: "Expert",
    bot: "PureNN",
    difficulty: 5,
    description: "Deep RL policy — high skill",
    dotColor: "bg-ctp-mauve",
  },
  {
    id: "hybrid",
    name: "Master",
    bot: "Hybrid",
    difficulty: 6,
    description: "NN + Bayesian adaptation — supreme",
    dotColor: "bg-ctp-sapphire",
  },
  {
    id: "beast",
    name: "Grandmaster",
    bot: "AcademicBeast",
    difficulty: 7,
    description: "TD-MoE + Archetype Classifier + Dewey EV — ultimate",
    dotColor: "bg-ctp-lavender",
  },
] as const;

function BotSelector({
  onSelect,
}: {
  onSelect: (botId: string) => void;
}) {
  return (
    <div className="flex flex-col items-center gap-8 py-12">
      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="font-[family-name:var(--font-petit-formal-script)] text-[36px] text-ctp-peach">
          Choose Your Opponent
        </h2>
        <p className="mt-2 text-[13px] text-ctp-subtext0">
          Select a bot difficulty to begin
        </p>
      </motion.div>

      <div className="grid grid-cols-2 gap-3 max-w-[500px] w-full px-4">
        {BOT_OPTIONS.map((bot, i) => (
          <motion.button
            key={bot.id}
            onClick={() => onSelect(bot.id)}
            className="group relative rounded-xl border border-ctp-surface1/60 bg-ctp-surface0/40 p-5 text-left transition-all duration-200 hover:border-ctp-peach/40 hover:bg-ctp-surface0/60 hover:shadow-[0_4px_20px_rgba(250,179,135,0.08)]"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 + i * 0.08 }}
          >
            {/* Difficulty dots */}
            <div className="mb-3 flex gap-1">
              {Array.from({ length: 4 }).map((_, j) => (
                <div
                  key={j}
                  className={[
                    "h-1.5 w-1.5 rounded-full transition-colors",
                    j < bot.difficulty
                      ? bot.dotColor
                      : "bg-ctp-surface2",
                  ].join(" ")}
                />
              ))}
            </div>

            <p className="text-[15px] font-semibold text-ctp-text group-hover:text-ctp-peach transition-colors">
              {bot.name}
            </p>
            <p className="mt-0.5 text-[11px] text-ctp-overlay0">
              {bot.bot} Bot
            </p>
            <p className="mt-2 text-[12px] text-ctp-subtext0 leading-relaxed">
              {bot.description}
            </p>
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
  adaptation: BotAdaptation | null;
}

interface BotAdaptation {
  estimated_bluff_rate: number;
  estimated_call_frequency: number;
  actions_observed: number;
  model_loaded: boolean;
  inferred_archetype?: string;
  archetype_confidence?: number;
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
            repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(250,179,135,0.06) 4px, rgba(250,179,135,0.06) 5px),
            repeating-linear-gradient(-45deg, transparent, transparent 4px, rgba(250,179,135,0.06) 4px, rgba(250,179,135,0.06) 5px)
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
  const color = red ? "text-ctp-red" : "text-ctp-text";

  return (
    <div
      className={[
        "rounded-lg border bg-ctp-surface0 transition-shadow duration-200",
        isSmall ? "h-[56px] w-[40px]" : "h-[126px] w-[90px] max-md:h-[91px] max-md:w-[65px]",
        selected
          ? "border-ctp-peach shadow-[0_0_20px_rgba(250,179,135,0.25),0_8px_24px_rgba(250,179,135,0.15)]"
          : "border-ctp-surface1",
        onClick && "cursor-pointer",
      ].join(" ")}
      onClick={onClick}
    >
      <div className="flex h-full flex-col justify-between p-1.5">
        {/* top-left corner */}
        <div className={`flex flex-col items-center leading-none ${color}`}>
          <span className={isSmall ? "text-[9px]" : "max-md:text-[10px] text-[13px]"}>
            {card.rank}
          </span>
          <span className={isSmall ? "text-[9px]" : "max-md:text-[11px] text-[14px]"}>
            {card.suit}
          </span>
        </div>

        {/* center suit */}
        <div className={`flex items-center justify-center ${color}`}>
          <span className={isSmall ? "text-[16px]" : "max-md:text-[22px] text-[30px]"}>
            {card.suit}
          </span>
        </div>

        {/* bottom-right corner (inverted) */}
        <div className={`flex flex-col items-center leading-none rotate-180 ${color}`}>
          <span className={isSmall ? "text-[9px]" : "max-md:text-[10px] text-[13px]"}>
            {card.rank}
          </span>
          <span className={isSmall ? "text-[9px]" : "max-md:text-[11px] text-[14px]"}>
            {card.suit}
          </span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// OpponentArea (top ~20%)
// ---------------------------------------------------------------------------

function OpponentArea({ cardCount, botName, lastBy, lastCount, lastRank }: { cardCount: number; botName: string; lastBy?: string; lastCount?: number; lastRank?: string }) {
  const visibleCards = Math.min(cardCount, 7);

  return (
    <div className="flex flex-col items-center gap-2 px-4 py-3 md:py-4">
      {/* avatar + name */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-ctp-lavender bg-ctp-surface0">
          <Bot className="h-5 w-5 text-ctp-lavender" />
        </div>
        <div className="text-center sm:text-left">
          <p className="text-[15px] font-medium text-ctp-lavender">{botName}</p>
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
      <div className="flex items-end pt-1">
        {Array.from({ length: visibleCards }).map((_, i) => {
          const mid = (visibleCards - 1) / 2;
          const maxAngle = Math.min(20, visibleCards * 3);
          const rot = visibleCards > 1 ? (i - mid) * (maxAngle * 2) / (visibleCards - 1) : 0;
          const archY = visibleCards > 1 ? 50 * (1 - Math.cos(((i - mid) / mid) * (Math.PI / 2))) : 0;
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
      <p className="text-[12px] italic text-ctp-overlay0">
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
}: {
  round: number;
  playerCount: number;
  botCount: number;
  deckCount: number;
  claimedRank: Rank;
  claimedBy: "You" | "Bot";
  playedCount: number;
  discardCount: number;
}) {
  return (
    <div className="relative flex flex-1 flex-col items-center justify-center gap-4 overflow-hidden px-4 py-2">


      {/* radial glow overlay */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 50%, rgba(250,179,135,0.06) 0%, transparent 70%)",
        }}
      />

      {/* status bar */}
      <motion.div
        className="relative z-10 flex items-center gap-2.5 rounded-full border border-ctp-surface1 bg-ctp-mantle px-5 py-2 text-[12px] sm:text-[13px] md:gap-3"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15 }}
      >
        <span className="text-ctp-overlay1">Round {round}</span>
        <span className="h-3 w-px bg-ctp-surface2" />
        <span className="font-medium text-ctp-peach">You: {playerCount}</span>
        <span className="h-3 w-px bg-ctp-surface2" />
        <span className="font-medium text-ctp-lavender">Bot: {botCount}</span>
        <span className="h-3 w-px bg-ctp-surface2" />
        <span className="text-ctp-overlay0">Deck: {deckCount}</span>
      </motion.div>

      {/* claimed rank */}
      <motion.div
        className="relative z-10 text-center"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.25 }}
      >
        <p
          className="font-[family-name:var(--font-petit-formal-script)] text-[32px] text-ctp-peach"
          style={{
            textShadow: "0 0 30px rgba(250,179,135,0.15)",
          }}
        >
          {RANK_LABELS[claimedRank]}
        </p>
        <p className="text-[13px] text-ctp-overlay1">
          Claimed by {claimedBy}
        </p>
      </motion.div>

      {/* played cards fan */}
      <motion.div
        className="relative z-10 flex items-end"
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
              className="-ml-[32px] first:ml-0"
              style={{
                transform: `rotate(${rot}deg)`,
                transformOrigin: "bottom center",
              }}
            >
              <PlayingCard faceDown />
            </div>
          );
        })}
      </motion.div>

      {/* discard pile */}
      <motion.div
        className="relative z-10 flex items-center gap-3 pt-1"
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
        <span className="text-[11px] text-ctp-overlay0">
          Discard: {discardCount} cards
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
}: {
  selected: Rank | null;
  onSelect: (r: Rank) => void;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <motion.div
      className="flex flex-col items-center gap-3"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 12 }}
      transition={{ duration: 0.25 }}
    >
      <p className="text-[11px] font-medium uppercase tracking-[0.15em] text-ctp-overlay1">
        Declare rank
      </p>

      <div className="flex flex-wrap justify-center gap-1.5">
        {RANKS.map((rank) => (
          <button
            key={rank}
            onClick={() => onSelect(rank)}
            className={[
              "flex h-[40px] w-[40px] items-center justify-center rounded-lg border text-[13px] font-medium transition-all duration-150",
              "md:h-[44px] md:w-[44px]",
              selected === rank
                ? "border-ctp-peach bg-ctp-peach text-ctp-base shadow-[0_0_12px_rgba(250,179,135,0.2)]"
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
          className="rounded-full border border-ctp-surface1 bg-transparent px-4 py-2 text-[13px] text-ctp-subtext0 transition-all hover:border-ctp-overlay1 hover:text-ctp-text"
        >
          Cancel
        </button>
        <button
          onClick={onConfirm}
          disabled={!selected}
          className="rounded-full bg-ctp-peach px-6 py-2 text-[13px] font-semibold text-ctp-base transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
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
  canPass,
  canCallBluff,
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
  canPass: boolean;
  canCallBluff: boolean;
}) {
  const n = hand.length;
  const mid = (n - 1) / 2;
  // Calculate angle spread — wider for fewer cards, narrower for many
  const maxAngle = Math.min(25, n * 3.5);
  const angleStep = n > 1 ? (maxAngle * 2) / (n - 1) : 0;
  // Arch lift in screen space (center highest, edges lowest)
  const ARCH_HEIGHT = 70;
  // Negative margin to overlap cards — tighter for more cards
  const overlap = n > 6 ? "-ml-[26px] md:-ml-[32px]" : n > 3 ? "-ml-[20px] md:-ml-[26px]" : "-ml-[14px] md:-ml-[18px]";

  return (
    <div className="flex flex-col items-center gap-3 px-4 pb-4 pt-2 md:pb-6">
      {/* card fan */}
      <div className="flex items-end justify-center">
        {hand.map((card, i) => {
          const rot = (i - mid) * angleStep;
          const isSelected = selected.has(i);
          // Arch: lift edges down in screen space so the fan reads as a clean arc
          const archY = n > 1 ? ARCH_HEIGHT * (1 - Math.cos(((i - mid) / mid) * (Math.PI / 2))) : 0;

          return (
            <div
              key={i}
              className={`${overlap} first:ml-0 relative`}
              style={{
                transform: `translateY(${archY}px)`,
                transformOrigin: "bottom center",
                zIndex: i,
              }}
            >
              <div style={{ transform: `rotate(${rot}deg)`, transformOrigin: "bottom center" }}>
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

      {/* action buttons */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenSelector}
          disabled={(selected.size === 0 && !showSelector) || canCallBluff}
          className="flex items-center gap-2 rounded-full bg-ctp-peach px-5 py-2.5 text-[14px] font-semibold text-ctp-base transition-all hover:brightness-110 hover:-translate-y-0.5 hover:shadow-[0_4px_16px_rgba(250,179,135,0.3)] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0 disabled:hover:shadow-none"
        >
          <Play className="h-4 w-4" />
          Play Cards
        </button>

        <button
          onClick={onCallBluff}
          disabled={!canCallBluff}
          className="flex items-center gap-2 rounded-full border-2 border-ctp-red px-5 py-2.5 text-[14px] font-semibold text-ctp-red transition-all hover:-translate-y-0.5 hover:bg-ctp-red/10 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
        >
          <AlertTriangle className="h-4 w-4" />
          Call Bluff!
        </button>
        <div className="relative group">
          <button
            onClick={onPass}
            disabled={!canPass}
            className="flex items-center gap-2 rounded-full border border-ctp-surface1 bg-ctp-surface0/50 px-5 py-2.5 text-[14px] font-medium text-ctp-subtext0 transition-all hover:-translate-y-0.5 hover:border-ctp-overlay1 hover:text-ctp-text disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
          >
            Pass
          </button>
          {!canPass && (
            <div className="pointer-events-none absolute -top-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-lg border border-ctp-surface1/60 bg-ctp-mantle/95 px-2.5 py-1 text-[11px] text-ctp-overlay1 opacity-0 shadow-lg backdrop-blur-sm transition-opacity group-hover:opacity-100">
              Draw pile empty — call bluff instead
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
  onPlayAgain,
}: {
  result: GameOverResult;
  onPlayAgain: () => void;
}) {
  const { humanWon, isDraw, message, adaptation } = result;

  const headline = isDraw
    ? "Draw!"
    : humanWon
      ? "You Won! 🎉"
      : "Bot Wins!";

  const subColor = isDraw
    ? "text-ctp-yellow"
    : humanWon
      ? "text-ctp-green"
      : "text-ctp-red";

  const glowColor = isDraw
    ? "shadow-[0_0_60px_rgba(249,226,175,0.12)]"
    : humanWon
      ? "shadow-[0_0_60px_rgba(166,227,161,0.12)]"
      : "shadow-[0_0_60px_rgba(243,139,168,0.12)]";

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
        <p className={`text-[32px] font-bold leading-tight ${subColor}`}>
          {headline}
        </p>
        <p className="mt-2 text-[13px] text-ctp-subtext0 leading-snug">
          {message}
        </p>

        {/* bot adaptation summary if available */}
        {adaptation && (
          <div className="mt-4 rounded-xl border border-ctp-lavender/25 bg-ctp-surface0/60 p-3 text-left">
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-ctp-lavender">
              AI Mental Model — Final State
            </p>
            <div className="grid grid-cols-2 gap-2 text-[12px]">
              <div>
                <span className="text-[10px] text-ctp-overlay0 block">Est. Bluff Rate</span>
                <span className="font-semibold text-ctp-peach">
                  {(adaptation.estimated_bluff_rate * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="text-[10px] text-ctp-overlay0 block">Est. Call Rate</span>
                <span className="font-semibold text-ctp-teal">
                  {(adaptation.estimated_call_frequency * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            {adaptation.inferred_archetype && (
              <div className="mt-2 pt-1.5 border-t border-ctp-surface2/30 flex items-center justify-between text-[11px]">
                <span className="text-ctp-overlay0">Inferred Archetype</span>
                <span className="font-semibold text-ctp-lavender">
                  {adaptation.inferred_archetype} ({((adaptation.archetype_confidence || 0) * 100).toFixed(0)}%)
                </span>
              </div>
            )}
            <p className="mt-1.5 text-[10px] text-ctp-overlay1">
              Built from {adaptation.actions_observed} observed action{adaptation.actions_observed !== 1 ? "s" : ""}
              {adaptation.model_loaded ? " · Saved to S3 memory" : ""}
            </p>
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
  onToggleCollapse,
  onClose,
  logs,
  adaptation,
}: {

  open: boolean;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onClose: () => void;
  logs: LogEntry[];
  adaptation?: BotAdaptation | null;
}) {
  return (
    <>
      {/* ── Desktop sidebar ── */}
      <motion.aside
        className="hidden md:flex h-full shrink-0 flex-col border-l border-ctp-surface1/60 bg-ctp-crust/80 backdrop-blur-sm overflow-hidden"
        animate={{ width: collapsed ? 0 : 280, opacity: collapsed ? 0 : 1 }}
        transition={{ duration: 0.2, ease: "easeInOut" }}
      >
        <div className="flex items-center justify-between border-b border-ctp-surface1/60 px-4 py-3 w-[280px]">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ctp-overlay1">
            Game Log
          </p>
          <button
            onClick={onClose}
            className="flex h-6 w-6 items-center justify-center rounded-md text-ctp-overlay0 transition-colors hover:bg-ctp-surface0 hover:text-ctp-text"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-3 w-[280px]">
          {adaptation && (
            <div className="mb-3 rounded-lg border border-ctp-lavender/30 bg-ctp-surface0/60 p-2.5 text-left">
              <div className="flex items-center justify-between pb-1.5 border-b border-ctp-surface1/40">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-ctp-lavender">
                  AI Mental Model
                </span>
                {adaptation.model_loaded && (
                  <span className="rounded bg-ctp-green/20 px-1 py-0.5 text-[9px] font-medium text-ctp-green">
                    S3 Memory
                  </span>
                )}
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                <div>
                  <span className="text-[10px] text-ctp-overlay0 block">Estimated Bluff</span>
                  <span className="font-semibold text-ctp-peach">
                    {(adaptation.estimated_bluff_rate * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-ctp-overlay0 block">Estimated Call</span>
                  <span className="font-semibold text-ctp-teal">
                    {(adaptation.estimated_call_frequency * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="mt-1.5 text-[10px] text-ctp-overlay1">
                Learned from {adaptation.actions_observed} action{adaptation.actions_observed !== 1 ? "s" : ""}
              </div>
            </div>
          )}

          <div className="space-y-3">
            {logs.map((entry, i) => (
              <div key={i}>
                <span className="block text-[10px] text-ctp-overlay0">
                  {entry.time}
                </span>
                <p
                  className={[
                    "text-[12px] leading-snug",
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
              className="absolute right-0 top-0 flex h-full w-[280px] flex-col border-l border-ctp-surface1 bg-ctp-surface0"
              initial={{ x: 280 }}
              animate={{ x: 0 }}
              exit={{ x: 280 }}
              transition={{ type: "spring", damping: 28, stiffness: 320 }}
            >
              <div className="flex items-center justify-between border-b border-ctp-surface1 px-4 py-3">
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ctp-overlay1">
                  Game Log
                </p>
                <button
                  onClick={onClose}
                  className="text-ctp-overlay1 transition-colors hover:text-ctp-text"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto px-4 py-3">
                {adaptation && (
                  <div className="mb-3 rounded-lg border border-ctp-lavender/30 bg-ctp-surface1/60 p-2.5 text-left">
                    <div className="flex items-center justify-between pb-1.5 border-b border-ctp-surface2/40">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-ctp-lavender">
                        AI Mental Model
                      </span>
                      {adaptation.model_loaded && (
                        <span className="rounded bg-ctp-green/20 px-1 py-0.5 text-[9px] font-medium text-ctp-green">
                          S3 Memory
                        </span>
                      )}
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                      <div>
                        <span className="text-[10px] text-ctp-overlay0 block">Estimated Bluff</span>
                        <span className="font-semibold text-ctp-peach">
                          {(adaptation.estimated_bluff_rate * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-ctp-overlay0 block">Estimated Call</span>
                        <span className="font-semibold text-ctp-teal">
                          {(adaptation.estimated_call_frequency * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    {adaptation.inferred_archetype && (
                      <div className="mt-1.5 pt-1.5 border-t border-ctp-surface2/30 flex items-center justify-between text-[10px]">
                        <span className="text-ctp-overlay0">Archetype:</span>
                        <span className="font-medium text-ctp-lavender">
                          {adaptation.inferred_archetype} ({((adaptation.archetype_confidence || 0) * 100).toFixed(0)}%)
                        </span>
                      </div>
                    )}
                    <div className="mt-1.5 text-[10px] text-ctp-overlay1">
                      Learned from {adaptation.actions_observed} action{adaptation.actions_observed !== 1 ? "s" : ""}
                    </div>
                  </div>
                )}

                <div className="space-y-3">
                  {logs.map((entry, i) => (
                    <div key={i}>
                      <span className="block text-[10px] text-ctp-overlay0">
                        {entry.time}
                      </span>
                      <p
                        className={[
                          "text-[12px] leading-snug",
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

// Reports the Clerk identity WITHOUT suspending the page tree: while Clerk
// is still loading (or unreachable, e.g. blocked identity domain), this
// renders null and callers fall back to the device UUID. When Clerk
// resolves, rooms created afterwards upgrade to the Clerk id.
function ClerkIdentity({ onId }: { onId: (id: string | null) => void }) {
  const { user } = useUser();
  const id = user?.id ?? null;
  useEffect(() => {
    onId(id);
  }, [id, onId]);
  return null;
}

export default function GamePage() {
  const router = useRouter();
  const sounds = useGameSounds();
  const [clerkUserId, setClerkUserId] = useState<string | null>(null);

  const [username, setUsername] = useState<string | null>(null);
  const [checking, setChecking] = useState(true);

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
  const [pileCount, setPileCount] = useState(0);
  const [deckCount, setDeckCount] = useState(28);
  const [claimedRank, setClaimedRank] = useState<Rank>("7");
  const [claimedBy, setClaimedBy] = useState<"You" | "Bot">("You");
  const [playedCount, setPlayedCount] = useState(1);
  const [canCallBluff, setCanCallBluff] = useState(false);
  const [canPass, setCanPass] = useState(false);
  const [adaptation, setAdaptation] = useState<BotAdaptation | null>(null);

  const [logOpen, setLogOpen] = useState(false);
  const [logCollapsed, setLogCollapsed] = useState(false);
  const [rulesOpen, setRulesOpen] = useState(false);

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [gameOver, setGameOver] = useState<GameOverResult | null>(null);


  // ── auth check ──
  useEffect(() => {
    const name = localStorage.getItem("bluff-username");
    if (!name) {
      router.replace("/");
    } else {
      setUsername(name);
      setChecking(false);
    }
  }, [router]);


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
    setShowSelector(true);
  }, [selected.size]);

  const handleConfirmPlay = useCallback(() => {
    if (!selectorRank) return;

    const selectedIndices = Array.from(selected);

    if (ws && connected) {
      sounds.play();
      ws.send(
        JSON.stringify({
          action: "play",
          cards: selectedIndices,
          rank: selectorRank,
        })
      );
    }

    setSelected(new Set());
    setShowSelector(false);
    setSelectorRank(null);
  }, [selectorRank, selected, ws, connected, sounds]);

  const handleCancelPlay = useCallback(() => {
    setShowSelector(false);
    setSelectorRank(null);
  }, []);

  const handleCallBluff = useCallback(() => {
    if (ws && connected) {
      sounds.callBluff();
      ws.send(JSON.stringify({ action: "call_bluff" }));
    }
    setSelected(new Set());
    setShowSelector(false);
    setSelectorRank(null);
  }, [ws, connected, sounds]);

  const handlePass = useCallback(() => {
    if (ws && connected) {
      sounds.pass();
      ws.send(JSON.stringify({ action: "pass" }));
    }
    setSelected(new Set());
    setShowSelector(false);
    setSelectorRank(null);
  }, [ws, connected, sounds]);


  const handleLogout = useCallback(() => {
    localStorage.removeItem("bluff-username");
    router.push("/");
  }, [router]);

  const handleBotSelect = useCallback((botId: string) => {
    setSelectedBot(botId);
    setBotSelected(true);
  }, []);

  const handlePlayAgain = useCallback(() => {
    // Close the overlay and close the current socket — re-selecting bot triggers a fresh connection
    setGameOver(null);
    if (ws) ws.close();
    setWs(null);
    setConnected(false);
    setBotSelected(false);
    setSelectedBot(null);
    setHand([]);
    setSelected(new Set());
    setLogs([]);
    setAdaptation(null);
    setPileCount(0);
    setDeckCount(28);
    setRound(1);
    setBotCards(14);
    setCanCallBluff(false);
    setCanPass(false);
  }, [ws]);

  // ── Connect to WebSocket after bot selected ──
  useEffect(() => {
    if (!botSelected || !selectedBot) return;

    let socket: WebSocket | null = null;
    let isMounted = true;

    const getStableUserId = () => {
      if (clerkUserId) return clerkUserId;
      if (typeof window === "undefined") return "";
      let localId = localStorage.getItem("bluff_device_id");
      if (!localId) {
        localId = typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `anon_${Date.now()}_${Math.random()}`;
        localStorage.setItem("bluff_device_id", localId);
      }
      return localId;
    };

    async function initWs() {
      try {
        const stableId = getStableUserId();
        const queryParam = stableId ? `&user_id=${encodeURIComponent(stableId)}` : "";
        const wsParam = stableId ? `?user_id=${encodeURIComponent(stableId)}` : "";
        const res = await fetch(`${API_URL}/rooms?bot_name=${selectedBot}${queryParam}`, {
          method: "POST",
        });
        if (!res.ok) return;
        const data = await res.json();
        const roomId = data.room_id;
        if (!roomId) return;

        socket = new WebSocket(`${WS_URL}/ws/${roomId}${wsParam}`);

        socket.onopen = () => {
          if (!isMounted) return;
          setConnected(true);
          setWs(socket);
          const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          setLogs((prev) => [
            { time: now, text: `Connected — playing ${selectedBot}`, kind: "normal" },
            ...prev,
          ]);
        };

        socket.onmessage = (event) => {
          if (!isMounted) return;
          try {
            const payload = typeof event.data === "string" ? JSON.parse(event.data) : event.data;
            const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            if (payload.type === "game_state") {
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
              if (payload.turn !== undefined) {
                setRound(payload.turn);
              }
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
                setPlayedCount(la.cards?.length || 1);
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
                  text: payload.message || (isDraw ? "Draw — 100-turn limit reached." : humanWon ? "You won!" : "Bot wins!"),
                  kind: logKind,
                },
                ...prev,
              ]);
              setGameOver({
                humanWon,
                isDraw,
                message: payload.message || (isDraw ? "100-turn draw — no cards eliminated." : humanWon ? "You emptied your hand first!" : "Bot emptied its hand first."),
                adaptation,
              });

            } else if (payload.type === "error") {
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
      } catch {
        // Backend offline
      }
    }

    initWs();

    return () => {
      isMounted = false;
      if (socket) socket.close();
    };
  }, [botSelected, selectedBot]);

  // ── loading ──
  if (checking) {
    return (
      // HOTFIX by Tess 2026-09-10: two sibling JSX roots (Suspense + div) made
      // the module unparseable ("Expected ',', got 'ident'" @1246) — wrapped in
      // a fragment. Owner (Muse/Antigravity) review requested on AGENT_CHAT.
      <>
        <Suspense fallback={null}>
          <ClerkIdentity onId={setClerkUserId} />
        </Suspense>
        <div className="flex h-screen items-center justify-center bg-ctp-base">
          <motion.p
            className="text-[14px] text-ctp-subtext0"
            initial={{ opacity: 0 }}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.6, repeat: Infinity }}
          >
            Loading…
          </motion.p>
        </div>
      </>
    );
  }

  // ── bot selection screen ──
  if (!botSelected) {
    return (
      <>
        <Suspense fallback={null}>
          <ClerkIdentity onId={setClerkUserId} />
        </Suspense>
        <GameLayout username={username || ""} onLogout={handleLogout} onOpenRules={() => setRulesOpen(true)} showNav={true}>
          <div className="relative z-10 flex flex-1 items-center justify-center">
            <BotSelector onSelect={handleBotSelect} />
          </div>
          <RulesModal isOpen={rulesOpen} onClose={() => setRulesOpen(false)} />
        </GameLayout>
      </>
    );
  }

  return (
    <>
      <Suspense fallback={null}>
        <ClerkIdentity onId={setClerkUserId} />
      </Suspense>
      <GameLayout username={username || ""} onLogout={handleLogout} onOpenRules={() => setRulesOpen(true)}>

      <div className="flex flex-1 min-h-0 w-full overflow-hidden">
        {/* ── main game area ── */}
        <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
          {/* opponent — top */}
          <div className="shrink-0">
            <OpponentArea cardCount={botCards} botName={BOT_OPTIONS.find(b => b.id === selectedBot)?.bot || "Bot"} lastBy={claimedBy} lastCount={playedCount} lastRank={claimedRank} />
          </div>

          {/* center table — middle */}
          <CenterTable
            round={round}
            playerCount={hand.length}
            botCount={botCards}
            deckCount={deckCount}
            claimedRank={claimedRank}
            claimedBy={claimedBy}
            playedCount={playedCount}
            discardCount={pileCount}
          />

          {/* player hand — bottom */}
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
              canPass={canPass}
              canCallBluff={canCallBluff}
            />
          </div>
        </div>

        {/* ── game log sidebar ── */}
        <GameLogSidebar
          open={logOpen}
          collapsed={logCollapsed}
          onToggleCollapse={() => setLogCollapsed(!logCollapsed)}
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
        className="fixed bottom-4 right-4 z-40 hidden md:flex h-10 w-10 items-center justify-center rounded-full border border-ctp-surface1/60 bg-ctp-crust/80 text-ctp-overlay1 shadow-lg backdrop-blur-sm transition-colors hover:border-ctp-overlay1 hover:text-ctp-text"
        aria-label="Toggle game log"
      >
        <ScrollText className="h-4 w-4" />
      </button>
      {/* Mobile: open overlay */}
      <button
        onClick={() => setLogOpen(true)}
        className="fixed bottom-4 right-4 z-40 flex md:hidden h-10 w-10 items-center justify-center rounded-full border border-ctp-surface1/60 bg-ctp-crust/80 text-ctp-overlay1 shadow-lg backdrop-blur-sm transition-colors hover:border-ctp-overlay1 hover:text-ctp-text"
        aria-label="Open game log"
      >
        <ScrollText className="h-4 w-4" />
      </button>
      {/* ── Game Over Overlay ── */}
      <AnimatePresence>
        {gameOver && (
          <GameOverOverlay result={gameOver} onPlayAgain={handlePlayAgain} />
        )}
      </AnimatePresence>
    </GameLayout>
    </>
  );
}
