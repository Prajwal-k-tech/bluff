"use client";

import { useState } from "react";
import Link from "next/link";
import { MotionConfig, motion } from "motion/react";
import Balatro from "@/components/Balatro";
import RulesModal from "@/components/RulesModal";

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function LoginPage() {
  const [rulesOpen, setRulesOpen] = useState(false);

  return (
    <MotionConfig reducedMotion="user">
    <div className="relative flex min-h-dvh flex-col overflow-x-hidden bg-ctp-crust text-ctp-text">
      <div className="pointer-events-none absolute inset-0 opacity-25" aria-hidden="true">
        <Balatro
          color1="#c6a4e4"
          color2="#1c1722"
          color3="#0d0a12"
          spinSpeed={0.18}
          contrast={1.7}
          lighting={0.24}
          spinAmount={0.16}
          mouseInteraction={false}
        />
      </div>
      <div className="pointer-events-none ambient-scrim absolute inset-0" aria-hidden="true" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_18%_20%,rgba(198,164,228,0.10),transparent_38%),radial-gradient(ellipse_at_85%_75%,rgba(223,193,132,0.08),transparent_42%)]" aria-hidden="true" />

      <main className="relative z-10 mx-auto grid w-full max-w-7xl flex-1 items-center gap-12 px-6 pb-12 pt-4 sm:px-10 lg:grid-cols-[1.1fr_0.9fr] lg:gap-20 lg:px-14">
        <section className="mx-auto w-full max-w-2xl text-center lg:mx-0 lg:text-left">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: "easeOut" }}
          >
            <h1 className="editorial-display-tight max-w-2xl text-6xl leading-[0.96] text-ctp-peach sm:text-7xl lg:text-[82px]">
              Bluff
            </h1>
          </motion.div>

        </section>

        <motion.section
          aria-labelledby="join-title"
          className="marble-panel mx-auto w-full max-w-md rounded-md p-6 sm:p-8 lg:ml-auto"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.12, ease: "easeOut" }}
        >
          <div className="mb-7">
            <h2 id="join-title" className="editorial-display text-4xl leading-tight text-ctp-text sm:text-[44px]">Play</h2>
          </div>

          <p className="text-sm leading-6 text-ctp-subtext0">Choose a bot. Use Data to manage returning-player memory and gameplay logging.</p>
          <Link href="/game" className="button-primary mt-6 flex min-h-12 w-full items-center justify-center rounded-md px-5 text-sm font-bold tracking-wide">
            Choose an opponent
          </Link>

          <div className="my-6 editorial-rule opacity-50" />
          <nav className="flex flex-wrap items-center justify-center gap-2" aria-label="Project links">
            <a href="https://github.com/oGhostyyy/Bluff" target="_blank" rel="noopener noreferrer" className="flex min-h-11 items-center gap-2 rounded-md border border-ctp-surface1/80 px-4 text-xs font-medium text-ctp-subtext0 transition-colors hover:border-ctp-peach/40 hover:text-ctp-peach">
              GitHub
            </a>
            <button type="button" onClick={() => setRulesOpen(true)} className="flex min-h-11 items-center gap-2 rounded-md border border-ctp-surface1/80 px-4 text-xs font-medium text-ctp-subtext0 transition-colors hover:border-ctp-peach/40 hover:text-ctp-peach">
              Rules
            </button>
            <a href="https://linktr.ee/oGhostyyy" target="_blank" rel="noopener noreferrer" className="flex min-h-11 items-center gap-2 rounded-md border border-ctp-surface1/80 px-4 text-xs font-medium text-ctp-subtext0 transition-colors hover:border-ctp-peach/40 hover:text-ctp-peach">
              Linktree
            </a>
          </nav>
        </motion.section>
      </main>

      <footer className="relative z-10 mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-6 pb-5 text-[11px] text-ctp-overlay1 sm:px-10 lg:px-14">
        <span>
          Made by{" "}
          <a href="https://linktr.ee/oGhostyyy" target="_blank" rel="noopener noreferrer" className="text-ctp-subtext1 transition-colors hover:text-ctp-peach">
            oGhostty
          </a>
        </span>
      </footer>
      <RulesModal isOpen={rulesOpen} onClose={() => setRulesOpen(false)} />
    </div>
    </MotionConfig>
  );
}
