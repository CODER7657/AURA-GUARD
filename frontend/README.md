# AeroGuard - NASA Air Quality Forecaster

A React-based air quality monitoring and forecasting application built with Framer components integration, TypeScript, and modern web technologies.

## 🚀 Features

- **Framer Integration**: Seamlessly integrate your Framer template components
- **Air Quality Monitoring**: Real-time air quality data visualization
- **TypeScript Support**: Full type safety and IntelliSense
- **API Integration**: Ready-to-use API layer with error handling
- **Responsive Design**: Mobile-first responsive components
- **Animation Support**: Built-in Framer Motion animations

## 📋 Prerequisites

- Node.js (v20.11.1 or higher)
- npm or yarn
- Framer account (for exporting components)

## 🛠️ Installation

1. **Clone and install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env.local
   ```
   Edit `.env.local` with your API keys and configuration.

3. **Start the development server:**
   ```bash
   npm run dev
   ```

## 🎨 Framer Integration Workflow

### Step 1: Export Components from Framer

1. Install the **React Export** plugin in your Framer project
2. Select the components you want to export
3. Use the plugin to generate React code
4. Download the generated components

### Step 2: Install Unframer CLI

```bash
npm install -g unframer
```

### Step 3: Import Components

1. Run unframer to download components:
   ```bash
   unframer --project-url https://creative-assumptions-648427.framer.app/
   ```

2. Move exported components to `src/components/framer/`

3. Enhance components with API integration:
   ```tsx
   import { withApiIntegration } from '@/components/FramerHelpers';
   
   const EnhancedComponent = withApiIntegration(YourFramerComponent);
   ```

### Step 4: Use Enhanced Components

```tsx
import { EnhancedComponent } from '@/components/framer/YourComponent';

function App() {
  return (
    <EnhancedComponent
      onSubmit={(data) => console.log('Form submitted:', data)}
      isLoading={false}
      data={yourApiData}
    />
  );
}
```

## 📁 Project Structure

```
src/
├── api/                    # API configuration and services
│   ├── config.ts          # Axios configuration
│   └── service.ts         # API service methods
├── components/            # React components
│   ├── framer/           # Exported Framer components
│   ├── FramerHelpers.tsx # Framer integration utilities
│   └── AirQualityDashboard.tsx
├── hooks/                # Custom React hooks
│   └── useApi.ts         # API data fetching hooks
├── types/                # TypeScript type definitions
│   └── index.ts          # Application types
├── utils/                # Utility functions
│   └── helpers.ts        # Helper functions
└── App.tsx               # Main application component
```

## 🔌 API Integration

The project includes a complete API integration layer with custom hooks for data fetching, form submissions, and authentication.

## 🎯 Environment Variables

Create a `.env.local` file with:

```env
VITE_API_BASE_URL=http://localhost:3001/api
VITE_NASA_API_KEY=your_nasa_api_key
VITE_OPENWEATHER_API_KEY=your_openweather_api_key
VITE_DEBUG_MODE=true
```

## 🚀 Development Scripts

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
