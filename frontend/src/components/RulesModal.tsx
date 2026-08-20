"use client";

import { motion, AnimatePresence } from "motion/react";
import { X, BookOpen, ShieldAlert, Award, Zap, HelpCircle } from "lucide-react";

interface RulesModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function RulesModal({ isOpen, onClose }: RulesModalProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0 bg-ctp-crust/80 backdrop-blur-md"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Modal Container */}
          <motion.div
            className="relative z-10 w-full max-w-xl max-h-[85vh] flex flex-col rounded-2xl border border-ctp-surface1 bg-ctp-base p-6 shadow-2xl overflow-hidden"
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-ctp-surface1 pb-4">
              <div className="flex items-center gap-2.5">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-ctp-peach/10 border border-ctp-peach/30 text-ctp-peach">
                  <BookOpen className="h-5 w-5" />
                </div>
                <div>
                  <h2 className="text-[18px] font-bold text-ctp-text leading-tight">
                    How to Play Bluff
                  </h2>
                  <p className="text-[12px] text-ctp-subtext0">
                    Master the art of deception and strategy
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-ctp-overlay1 transition-colors hover:bg-ctp-surface0 hover:text-ctp-text"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="flex-1 overflow-y-auto py-4 space-y-4 pr-1 text-[13px] text-ctp-subtext1 leading-relaxed">
              {/* Objective */}
              <div className="rounded-xl border border-ctp-surface1/60 bg-ctp-surface0/40 p-4">
                <div className="flex items-center gap-2 font-semibold text-ctp-peach mb-1">
                  <Award className="h-4 w-4" />
                  <span>Objective</span>
                </div>
                <p>
                  Be the first player to eliminate all cards from your hand. You play face-down cards to the center pile each turn.
                </p>
              </div>

              {/* Basic Gameplay */}
              <div className="space-y-3">
                <h3 className="font-semibold text-ctp-text flex items-center gap-2">
                  <Zap className="h-4 w-4 text-ctp-mauve" />
                  <span>Turn Mechanics</span>
                </h3>
                <ul className="list-disc list-inside space-y-2 text-ctp-subtext0 pl-1">
                  <li>
                    Select <span className="text-ctp-peach font-medium">1 to 4 cards</span> from your hand to play onto the center pile.
                  </li>
                  <li>
                    Declare the <span className="text-ctp-peach font-medium">claimed rank</span> of your played cards (e.g. "2 Kings").
                  </li>
                  <li>
                    You <strong className="text-ctp-text">can lie!</strong> The cards you place down do not actually have to match the declared rank.
                  </li>
                </ul>
              </div>

              {/* Calling Bluff */}
              <div className="space-y-3">
                <h3 className="font-semibold text-ctp-text flex items-center gap-2">
                  <ShieldAlert className="h-4 w-4 text-ctp-red" />
                  <span>Calling "Bluff!"</span>
                </h3>
                <p className="text-ctp-subtext0">
                  On an opponent&apos;s play, you can call <strong className="text-ctp-red">Bluff!</strong> if you suspect they lied about their cards.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-[12px]">
                  <div className="rounded-xl border border-ctp-green/20 bg-ctp-green/5 p-3">
                    <p className="font-semibold text-ctp-green mb-1">If Bluffer is Caught</p>
                    <p className="text-ctp-subtext0">The bluffer must pick up the entire center pile into their hand.</p>
                  </div>
                  <div className="rounded-xl border border-ctp-red/20 bg-ctp-red/5 p-3">
                    <p className="font-semibold text-ctp-red mb-1">If Accuser is Wrong</p>
                    <p className="text-ctp-subtext0">The accuser (caller) must pick up the entire center pile!</p>
                  </div>
                </div>
              </div>

              {/* Bot Info */}
              <div className="rounded-xl border border-ctp-mauve/20 bg-ctp-mauve/5 p-3 text-[12px]">
                <p className="font-semibold text-ctp-mauve mb-1 flex items-center gap-1.5">
                  <HelpCircle className="h-3.5 w-3.5" />
                  <span>Adaptive AI Opponent</span>
                </p>
                <p className="text-ctp-subtext0">
                  The bot uses Bayesian opponent modeling to track your bluff frequency over time and adapt its challenges accordingly!
                </p>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="border-t border-ctp-surface1 pt-4 flex justify-end">
              <button
                onClick={onClose}
                className="rounded-full bg-ctp-peach px-6 py-2 text-[13px] font-semibold text-ctp-base transition-all hover:brightness-110"
              >
                Got it, Let&apos;s Play
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
