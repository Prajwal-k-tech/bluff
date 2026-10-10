import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const cormorantItalic = localFont({
  src: "../../public/fonts/cormorant-garamond-italic-latin.woff2",
  variable: "--font-cormorant-italic",
  style: "italic",
  weight: "400",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Bluff",
  description: "A two-player bluffing game about reading claims, risk, and hidden cards.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${cormorantItalic.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-ctp-base text-ctp-text">
        {children}
      </body>
    </html>
  );
}
