import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "CallPilot",
  description: "Agentic voice scheduling demo"
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
