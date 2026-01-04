# webRL

Web-based Reinforcement Learning experiments

## Overview

This repository provides a browser-based environment for experimenting with reinforcement learning algorithms. The current implementation features a bipedal walker controlled by GRPO (Group Relative Policy Optimization).

## Features

- **Pure Web-based**: Runs entirely in the browser, no server required
- **Real-time Visualization**: Watch the agent learn in real-time
- **Interactive Controls**: Adjust hyperparameters on the fly
- **Worker-based Training**: Parallel training using Web Workers

## Getting Started

Simply open `index.html` in a modern web browser. No build step or dependencies installation required.

**Important**: The application requires TensorFlow.js and Matter.js to be loaded from CDN. If you see errors about CDN loading:
- Check your ad blocker or content blocker settings
- Ensure network/firewall allows `cdn.jsdelivr.net`
- Try a different browser or incognito/private mode
- Check browser console for specific error messages

### Development

For development with type checking and linting:

```bash
# Install development dependencies (optional)
npm install

# Run static analysis
npm run lint        # Check code with ESLint
npm run typecheck   # Check types with TypeScript

# Fix linting issues automatically
npm run lint:fix
```

The codebase uses **JSDoc type annotations** for type safety without requiring a build step. This provides:
- Type checking via TypeScript in check-only mode
- IDE autocomplete and inline documentation
- Static analysis without compilation

## Architecture

The codebase is organized into modular components for maintainability. See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed information about the project structure and design principles.

- `index.html` - Main entry point with UI
- `js/environment.js` - Environment simulation (Biped walker)
- `js/grpo.js` - GRPO algorithm implementation
- `js/worker.js` - Web Worker for parallel training
- `js/app.js` - Application logic and UI coordination
- `css/styles.css` - UI styling

## Technologies

- [TensorFlow.js](https://www.tensorflow.org/js) - Neural network and tensor operations
- [Matter.js](https://brm.io/matter-js/) - 2D physics engine
- Vanilla JavaScript - No framework dependencies

## License

MIT
