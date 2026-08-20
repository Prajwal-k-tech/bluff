import { ClerkProvider } from "@clerk/nextjs";
import type { Metadata } from "next";
import { Poppins } from "next/font/google";
import localFont from "next/font/local";
import "./globals.css";

const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const petitFormalScript = localFont({
  src: "../../public/fonts/PetitFormalScript-Regular.ttf",
  variable: "--font-petit-formal-script",
  weight: "400",
});

export const metadata: Metadata = {
  title: "Bluff — AI Card Game",
  description: "A card game where AI meets bluff. Can you outsmart the bot?",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${poppins.variable} ${petitFormalScript.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-ctp-base text-ctp-text">
        <ClerkProvider>
          {children}
        </ClerkProvider>
      </body>
    </html>
  );
}