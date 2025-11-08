/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bee-yellow': '#FDB813',
        'bee-black': '#1A1A1A',
      }
    },
  },
  plugins: [],
}

