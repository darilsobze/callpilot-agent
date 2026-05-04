import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(222.2 84% 4.9%)",
        foreground: "hsl(210 40% 98%)",
        muted: "hsl(217.2 32.6% 17.5%)",
        border: "hsl(217.2 32.6% 17.5%)"
      }
    }
  },
  plugins: []
};

export default config;
