# Neural Canvas - AI & Art Literacy Platform

A Deep Learning interface that classifies artwork, explains it with LLMs, and generates new art while teaching AI Literacy.

## Features

- **CNN Analyzer**: Classify images using Convolutional Neural Networks
- **Model Arena**: Compare different AI models side-by-side
- **Diffusion Lab**: Generate images using diffusion models
- **ESRGAN Lab**: Enhance image resolution with ESRGAN
- **Neural Style Transfer Lab**: Apply artistic styles to your images

## Tech Stack

This project is built with:

- **Vite** - Fast build tool and dev server
- **React** - UI framework
- **TypeScript** - Type-safe JavaScript
- **shadcn-ui** - Beautiful UI components
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing
- **Framer Motion** - Animation library

## Prerequisites

- Node.js (v16 or higher recommended)
- npm or yarn package manager

## Local Development

### Installation

1. Clone the repository:
```sh
git clone <YOUR_GIT_URL>
cd neural-canvas
```

2. Install dependencies:
```sh
npm install
```

3. Start the development server:
```sh
npm run dev
```

The application will be available at `http://localhost:8080`

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run build:dev` - Build in development mode
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint to check code quality

## Project Structure

```
src/
├── components/          # React components
│   ├── analyzer/       # CNN Analyzer components
│   ├── arena/          # Model Arena components
│   ├── layout/         # Layout components (Sidebar, TopBar)
│   ├── sections/       # Landing page sections
│   └── ui/             # shadcn-ui components
├── hooks/              # Custom React hooks
├── lib/                # Utility functions
├── pages/              # Page components
│   ├── Index.tsx       # Landing page
│   ├── CNNArena.tsx    # CNN comparison page
│   ├── ModelArena.tsx  # Model comparison page
│   ├── DiffusionLab.tsx # Image generation page
│   ├── ESRGANLab.tsx   # Image enhancement page
│   └── NSTLab.tsx      # Neural Style Transfer page
└── main.tsx            # Application entry point
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Run linting: `npm run lint`
4. Test your changes
5. Submit a pull request

## License

This project is part of UCL COMP0220 coursework.
