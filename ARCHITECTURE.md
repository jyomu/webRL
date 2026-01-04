# webRL Project Structure

This document explains the architecture and file organization of the webRL project.

## Directory Structure

```
webRL/
├── index.html          # Main entry point
├── css/
│   └── styles.css      # UI styling
├── js/
│   ├── environment.js  # Biped walker physics simulation
│   ├── grpo.js         # GRPO algorithm implementation
│   ├── worker.js       # Web Worker for parallel training
│   └── app.js          # Main application logic and UI handlers
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

## Architecture Principles

### Separation of Concerns

The code is organized into modular files, each with a single responsibility:

1. **environment.js** - Contains the `Biped` class that simulates the bipedal walker using Matter.js physics engine
2. **grpo.js** - Implements the GRPO (Group Relative Policy Optimization) algorithm including policy network creation and action selection
3. **worker.js** - Runs training episodes in a Web Worker for parallel execution without blocking the UI
4. **app.js** - Coordinates all components, handles UI interactions, and manages the rendering loop
5. **styles.css** - All styling in one place for easy maintenance

### Type Safety Without Build Steps

The project uses **JSDoc annotations with TypeScript** for type safety:

```javascript
/**
 * @param {Sequential} policy - Neural network model
 * @param {number[]} state - Environment state (18 values)
 * @param {boolean} [explore=true] - Add exploration noise
 * @returns {number[]} Actions (6 values in [-1, 1])
 */
function getAction(policy, state, explore = true) {
  // Implementation
}
```

This approach provides:
- ✅ Full type checking via TypeScript compiler
- ✅ IDE autocomplete and inline docs
- ✅ No compilation or build step
- ✅ Pure JavaScript that runs in browser
- ✅ Static analysis with ESLint

### Error Handling and Logging

All functions include proper error handling:
- Null/undefined checks before operations
- Clear error messages with emoji indicators
- Graceful degradation when dependencies fail
- Informative logging to UI panel

### Benefits of This Structure

- **Maintainability**: Each file has a clear, single responsibility
- **Testability**: Components can be tested independently
- **Extensibility**: New environments or algorithms can be added by creating new files
- **Readability**: Related code is grouped together
- **Performance**: Worker-based training keeps the UI responsive

## External Dependencies

The project uses CDN-hosted libraries:
- **TensorFlow.js** (v4.17.0) - Neural network operations
- **Matter.js** (v0.19.0) - 2D physics simulation

These are loaded directly in the browser, no build step required.

## Running the Project

Simply open `index.html` in a modern web browser. No installation or build process needed.

## Future Improvements

Potential areas for enhancement:
- Add more RL environments (cart-pole, mountain car, etc.)
- Implement additional RL algorithms (PPO, A3C, etc.)
- Add model save/load functionality (using IndexedDB)
- Create a configuration file for hyperparameters
- Add automated tests with headless browser
- Implement visualization of neural network activations
- Add performance profiling tools

## Type Safety and Quality Assurance

### JSDoc + TypeScript Approach

The codebase uses JSDoc annotations for type safety without requiring compilation:

```bash
# Check types (no compilation, just validation)
npm run typecheck

# Check code quality
npm run lint

# Auto-fix linting issues  
npm run lint:fix
```

### Why JSDoc Instead of Full TypeScript?

1. **No Build Required** - Keep the "open and run" philosophy
2. **Easier Contribution** - No toolchain setup for users
3. **Full Type Safety** - TypeScript checks JSDoc annotations
4. **Better DX** - IDE autocomplete and inline docs
5. **Minimal Dependencies** - TypeScript only for dev

### Development Workflow

```bash
# 1. Install dev dependencies (optional)
npm install

# 2. Make changes to .js files with JSDoc comments

# 3. Check types
npm run typecheck

# 4. Fix any issues
npm run lint:fix

# 5. Test in browser
# Open index.html

# 6. Commit changes
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.
