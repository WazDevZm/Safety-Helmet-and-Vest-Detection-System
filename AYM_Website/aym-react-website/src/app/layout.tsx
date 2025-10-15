import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Adventist Youth Ministries (AYM) - Seventh-day Adventist Church",
  description: "Join the Adventist Youth Ministries community. Discover faith, fellowship, and service opportunities for young people in the Seventh-day Adventist Church.",
  keywords: "Adventist Youth, AYM, Seventh-day Adventist, Youth Ministry, Pathfinder, Adventurer, Ambassador",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased`}>
        <Navigation />
        <main>{children}</main>
      </body>
    </html>
  );
}
