"use client";

import { motion, Variants } from "motion/react";
import { ElementType } from "react";

interface SplitTextProps {
  text: string;
  className?: string;
  charClassName?: string;
  delay?: number;
  stagger?: number;
  duration?: number;
  as?: ElementType;
  repeat?: boolean;
}

export default function SplitText({
  text,
  className = "",
  charClassName = "",
  delay = 0,
  stagger = 0.05,
  duration = 0.7,
  as: Tag = "span",
  repeat = false,
}: SplitTextProps) {
  const chars = Array.from(text);

  const container: Variants = {
    hidden: {},
    visible: {
      transition: { staggerChildren: stagger, delayChildren: delay },
    },
  };

  const child: Variants = {
    hidden: { y: "0.7em", opacity: 0, filter: "blur(8px)" },
    visible: {
      y: "0em",
      opacity: 1,
      filter: "blur(0px)",
      transition: { duration, ease: [0.22, 1, 0.36, 1] },
    },
  };

  return (
    <Tag className={className} aria-label={text}>
      <motion.span
        style={{ display: "inline-block" }}
        variants={container}
        initial="hidden"
        animate={repeat ? undefined : "visible"}
        whileInView={repeat ? "visible" : undefined}
        aria-hidden
      >
        {chars.map((c, i) => (
          <motion.span
            key={i}
            variants={child}
            style={{ display: "inline-block", whiteSpace: "pre" }}
            className={charClassName}
          >
            {c}
          </motion.span>
        ))}
      </motion.span>
    </Tag>
  );
}
