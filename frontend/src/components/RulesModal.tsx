"use client";

import { useEffect, useRef } from "react";

interface RulesModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function RulesModal({ isOpen, onClose }: RulesModalProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (isOpen && !dialog.open) dialog.showModal();
    if (!isOpen && dialog.open) dialog.close();
  }, [isOpen]);

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="rules-title"
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      className="m-auto max-h-[90dvh] w-[calc(100%-2rem)] max-w-xl overflow-visible rounded-none bg-transparent p-0 text-ctp-text backdrop:bg-ctp-crust/80 backdrop:backdrop-blur-md"
      style={{ borderRadius: 0 }}
    >
      <div
        className="marble-panel relative z-10 flex max-h-[85dvh] w-full flex-col overflow-hidden p-6 shadow-2xl"
        style={{ borderRadius: 0 }}
      >
        <header className="flex items-center justify-between border-b border-ctp-surface1 pb-4">
          <div>
            <h2 id="rules-title" className="editorial-modal-title text-ctp-text">
              Rules
            </h2>
          </div>
        </header>

        <div
          tabIndex={0}
          className="flex-1 space-y-4 overflow-y-auto py-4 pr-1 text-[13px] leading-relaxed text-ctp-subtext1"
        >
          <section className="space-y-1 border-b border-ctp-surface1 pb-4">
            <h3 className="font-semibold text-ctp-text">Objective and deal</h3>
            <p>
              Be the first to empty your hand. The 52-card deck deals 14 cards
              to each player, leaving 24 in the draw supply.
            </p>
          </section>

          <section className="space-y-1 border-b border-ctp-surface1 pb-4">
            <h3 className="font-semibold text-ctp-text">Play a round</h3>
            <p>
              The starter chooses any rank and plays 1 to 4 cards face down,
              claiming they are all that rank. The rank stays fixed for the
              round, and you may bluff. A packet is false if even one card
              differs from the claim.
            </p>
            <p>
              After a play, the other player may play 1 to 4 cards at the same
              rank, pass, or challenge the latest packet. A new play closes the
              previous packet&apos;s challenge opportunity. The starter cannot
              pass before playing.
            </p>
          </section>

          <section className="space-y-1 border-b border-ctp-surface1 pb-4">
            <h3 className="font-semibold text-ctp-text">Call bluff</h3>
            <p>
              A challenge reveals only the latest packet. If any card is the
              wrong rank, its player takes the whole center pile; otherwise,
              the challenger takes it. The challenge winner starts the next
              round and chooses any rank.
            </p>
          </section>

          <section className="space-y-1 border-b border-ctp-surface1 pb-4">
            <h3 className="font-semibold text-ctp-text">Pass</h3>
            <p>
              Passing ends the round. The center pile is returned face down to
              the draw supply and shuffled, then the passing player draws one card. The
              other player starts the next round. Passing is unavailable when
              the draw supply is empty.
            </p>
          </section>

          <section className="space-y-1">
            <h3 className="font-semibold text-ctp-text">Winning</h3>
            <p>
              Your final play is automatically challenged. A truthful packet
              wins; if it is false, you take the whole pile. There is no
              gameplay turn limit.
            </p>
          </section>
        </div>

        <footer className="flex justify-end border-t border-ctp-surface1 pt-4">
          <button
            onClick={onClose}
            className="min-h-11 rounded-none border border-ctp-peach px-6 py-2 text-[13px] font-semibold text-ctp-peach transition-colors hover:bg-ctp-peach/10"
            style={{ borderRadius: 0 }}
          >
            Close
          </button>
        </footer>
      </div>
    </dialog>
  );
}
