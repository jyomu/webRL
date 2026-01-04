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
- Add model save/load functionality
- Create a configuration file for hyperparameters
- Add automated tests
