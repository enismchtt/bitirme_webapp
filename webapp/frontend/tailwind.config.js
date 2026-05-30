/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          900: '#0a0e1a',
          800: '#0f1424',
          700: '#161b2e',
          600: '#1d2238',
        },
        accent: {
          400: '#7c3aed',
          500: '#6d28d9',
          600: '#5b21b6',
        },
        cyan: {
          glow: '#22d3ee',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
