import type { Metadata } from "next";
import { Space_Grotesk, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-ibm-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "\uD0C0\uBBF8\uC628 \uC778\uACF5\uC9C0\uB2A5 ASR Server",
  description: "Real-time speech recognition powered by MLX Qwen3-ASR",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body className={`${spaceGrotesk.variable} ${ibmPlexMono.variable}`} suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
