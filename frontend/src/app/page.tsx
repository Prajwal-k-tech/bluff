"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  SignInButton,
  SignUpButton,
  UserButton,
  useAuth,
} from "@clerk/nextjs";
import { motion } from "motion/react";
import { BookOpen, ExternalLink } from "lucide-react";
import Waves from "@/components/Waves";
import SplitText from "@/components/SplitText";
import GridDistortion from "@/components/GridDistortion";

function GithubIcon({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24">
      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [mounted, setMounted] = useState(false);
  const router = useRouter();
  const { isSignedIn } = useAuth();

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const name = username.trim();
    if (!name) return;
    localStorage.setItem("bluff-username", name);
    router.push("/game");
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-ctp-crust">
      {/* ---- Clerk auth controls (top-right) ---- */}
      {mounted && (
        isSignedIn ? (
          <div className="absolute right-5 top-5 z-20">
            <UserButton
              appearance={{
                elements: {
                  avatarBox: "h-9 w-9 rounded-full",
                  userButtonPopoverCard: "bg-ctp-surface0 border-ctp-surface1",
                  userButtonPopoverText: "text-ctp-text",
                },
              }}
            />
          </div>
        ) : (
          <div className="absolute right-5 top-5 z-20 flex items-center gap-2">
            <SignInButton mode="modal">
              <button className="rounded-full border border-ctp-surface1/60 bg-ctp-surface0/30 px-3.5 py-2 text-[12px] text-ctp-subtext0 backdrop-blur-sm transition-all duration-200 hover:border-ctp-peach/40 hover:text-ctp-peach hover:bg-ctp-surface0/50">
                Sign In
              </button>
            </SignInButton>
            <SignUpButton mode="modal">
              <button className="rounded-full border border-ctp-peach/30 bg-ctp-peach/10 px-3.5 py-2 text-[12px] font-medium text-ctp-peach backdrop-blur-sm transition-all duration-200 hover:border-ctp-peach/50 hover:bg-ctp-peach/15">
                Sign Up
              </button>
            </SignUpButton>
          </div>
        )
      )}

      {/* ---- Grid distortion background ---- */}
      {mounted && <GridDistortion strength={0.3} className="opacity-40" />}

      {/* ---- Waves background ---- */}
      {mounted && (
        <Waves
          lineColor="rgba(250, 179, 135, 0.08)"
          waveSpeedX={0.012}
          waveSpeedY={0.004}
          waveAmpX={40}
          waveAmpY={20}
          xGap={14}
          yGap={36}
        />
      )}

      {/* ---- Ambient glow layers ---- */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-1/2 top-1/3 h-[700px] w-[700px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-ctp-peach/[0.04] blur-[160px]" />
        <div className="absolute right-1/4 bottom-1/4 h-[400px] w-[400px] rounded-full bg-ctp-mauve/[0.03] blur-[120px]" />
      </div>

      {/* ---- Main content ---- */}
      <div className="relative z-10 w-full max-w-[440px] px-6">
        {/* Title block */}
        <motion.div
          className="mb-8 text-center"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: "easeOut" }}
        >
          <h1
            className="font-[family-name:var(--font-petit-formal-script)] text-[48px] font-normal tracking-[6px] text-ctp-peach sm:text-[64px]"
            style={{
              textShadow:
                "0 0 60px rgba(250, 179, 135, 0.35), 0 0 120px rgba(250, 179, 135, 0.1)",
            }}
          >
            <SplitText text="Bluff" stagger={0.08} delay={0.15} />
          </h1>

          {/* Decorative line */}
          <motion.div
            className="mx-auto mt-4 h-px w-[140px]"
            initial={{ opacity: 0, scaleX: 0 }}
            animate={{ opacity: 1, scaleX: 1 }}
            transition={{ duration: 0.7, delay: 0.25 }}
            style={{
              background:
                "linear-gradient(90deg, transparent, var(--ctp-peach), transparent)",
            }}
          />
        </motion.div>

        {/* ---- Login card — glassmorphism ---- */}
        <motion.div
          className="rounded-2xl border border-white/[0.08] p-8 shadow-2xl backdrop-blur-xl"
          style={{
            background:
              "linear-gradient(135deg, rgba(49, 50, 68, 0.55) 0%, rgba(30, 30, 46, 0.75) 100%)",
            boxShadow:
              "0 8px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.06), 0 0 0 1px rgba(250,179,135,0.04)",
          }}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.35 }}
        >
          <form onSubmit={handleSubmit}>
            <div className="mb-5 text-center">
              <h2 className="text-[15px] font-semibold text-ctp-text">
                Take a seat at the table
              </h2>
              <p className="mt-1 text-[12px] text-ctp-subtext0">
                Pick an alias to challenge the adaptive AI
              </p>
            </div>

            <label
              htmlFor="alias"
              className="mb-2 block text-[11px] font-semibold uppercase tracking-[0.18em] text-ctp-overlay1"
            >
              Your Alias
            </label>
            <input
              id="alias"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your name"
              autoFocus
              className="mb-6 w-full rounded-xl border border-ctp-surface1/80 bg-ctp-crust/60 px-4 py-3.5 text-[14px] text-ctp-text placeholder:text-ctp-overlay0 transition-all duration-200 focus:border-ctp-peach/50 focus:outline-none focus:ring-2 focus:ring-ctp-peach/15 focus:shadow-[0_0_20px_rgba(250,179,135,0.08)]"
            />

            {/* Submit button — clean, no spinning border */}
            <button
              type="submit"
              disabled={!username.trim()}
              className="w-full rounded-full bg-ctp-peach py-3.5 text-[15px] font-bold tracking-wide text-ctp-crust transition-all duration-200 hover:-translate-y-0.5 hover:brightness-105 hover:shadow-[0_8px_28px_rgba(250,179,135,0.45)] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0 disabled:hover:shadow-none"
            >
              Enter the Game
            </button>
          </form>
        </motion.div>

        {/* ---- Quick links ---- */}
        <motion.div
          className="mt-8 flex items-center justify-center gap-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.55 }}
        >
          <a
            href="https://github.com/oGhostyyy/Bluff"
            target="_blank"
            rel="noopener noreferrer"
            className="group flex items-center gap-1.5 rounded-full border border-ctp-surface1/60 bg-ctp-surface0/30 px-3.5 py-2 text-[12px] text-ctp-subtext0 backdrop-blur-sm transition-all duration-200 hover:border-ctp-peach/40 hover:text-ctp-peach hover:bg-ctp-surface0/50"
          >
            <GithubIcon className="h-3.5 w-3.5" />
            GitHub
          </a>
          <a
            href="https://github.com/oGhostyyy/Bluff/blob/main/docs/game-rules.md"
            target="_blank"
            rel="noopener noreferrer"
            className="group flex items-center gap-1.5 rounded-full border border-ctp-surface1/60 bg-ctp-surface0/30 px-3.5 py-2 text-[12px] text-ctp-subtext0 backdrop-blur-sm transition-all duration-200 hover:border-ctp-peach/40 hover:text-ctp-peach hover:bg-ctp-surface0/50"
          >
            <BookOpen className="h-3.5 w-3.5" />
            Rules
          </a>
          <a
            href="https://linktr.ee/oGhostyyy"
            target="_blank"
            rel="noopener noreferrer"
            className="group flex items-center gap-1.5 rounded-full border border-ctp-surface1/60 bg-ctp-surface0/30 px-3.5 py-2 text-[12px] text-ctp-subtext0 backdrop-blur-sm transition-all duration-200 hover:border-ctp-peach/40 hover:text-ctp-peach hover:bg-ctp-surface0/50"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            Linktree
          </a>
        </motion.div>

        {/* ---- Footer ---- */}
        <motion.div
          className="mt-8 flex flex-col items-center gap-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.65 }}
        >
          <div
            className="h-px w-[60px]"
            style={{
              background:
                "linear-gradient(90deg, transparent, var(--ctp-surface2), transparent)",
            }}
          />
          <p className="text-[11px] text-ctp-overlay0">
            Made with ♠ by{" "}
            <a
              href="https://linktr.ee/oGhostyyy"
              target="_blank"
              rel="noopener noreferrer"
              className="text-ctp-peach/80 transition-colors hover:text-ctp-peach"
            >
              @oGhostyyy
            </a>
          </p>
        </motion.div>
      </div>
    </div>
  );
}
