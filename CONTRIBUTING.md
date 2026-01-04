# Contributing to webRL

Thank you for your interest in contributing to webRL! This guide will help you understand our development practices and type safety approach.

## Development Setup

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, or Edge)
- Node.js (optional, for type checking and linting)

### Quick Start

1. **For Users**: Simply open `index.html` in your browser. No installation needed!

2. **For Developers**: Optionally install dev tools for static analysis:
   ```bash
   npm install
   ```

## Type Safety Approach

We use **JSDoc with TypeScript** for type safety without requiring a build step. This provides:

- ✅ Full type checking
- ✅ IDE autocomplete
- ✅ Inline documentation
- ✅ No compilation needed
- ✅ Browser-ready code

### Example

```javascript
/**
 * Creates a neural network policy
 * @param {number} inputSize - Size of input layer
 * @returns {Sequential} TensorFlow.js model
 * @throws {Error} If TensorFlow.js not loaded
 */
function createPolicy(inputSize) {
  if (typeof tf === 'undefined') {
    throw new Error('TensorFlow.js is required');
  }
  return tf.sequential();
}
```

## Code Quality

### Running Static Analysis

```bash
# Check types
npm run typecheck

# Check code quality
npm run lint

# Auto-fix linting issues
npm run lint:fix
```

### Before Committing

1. Ensure no TypeScript errors: `npm run typecheck`
2. Fix all linting issues: `npm run lint:fix`
3. Test in browser manually
4. Check console for errors

## Architecture Principles

### 1. Minimal Responsibility
- No build system required
- CDN dependencies only
- Pure JavaScript (with JSDoc types)
- Single-page application

### 2. Separation of Concerns
- `environment.js` - Physics simulation
- `grpo.js` - RL algorithm
- `worker.js` - Parallel training
- `app.js` - UI and coordination

### 3. Error Handling
Always check for null/undefined:

```javascript
if (!worker) {
  log('❌ Worker not initialized');
  return;
}
```

### 4. Logging
Use emoji indicators for log messages:
- ✅ Success
- ❌ Error
- ℹ️ Info
- 🚀 Action started
- ⏸️ Action paused

## Adding New Features

### 1. New RL Environment

Create a new file in `js/` following the `Biped` pattern:

```javascript
/**
 * @typedef {Object} StepResult
 * @property {number[]} state
 * @property {number} reward
 * @property {boolean} done
 */

class MyEnvironment {
  /**
   * @returns {number[]} Current state
   */
  getState() { }
  
  /**
   * @param {number[]} actions
   * @returns {StepResult}
   */
  step(actions) { }
  
  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} width
   * @param {number} height
   */
  render(ctx, width, height) { }
}
```

### 2. New RL Algorithm

Follow the pattern in `grpo.js`:

```javascript
/**
 * @param {Sequential} policy - Neural network
 * @param {number[]} state - Environment state
 * @returns {number[]} Actions
 */
function getAction(policy, state) {
  // Implementation
}
```

## Testing

### Manual Testing Checklist

- [ ] Application loads without errors
- [ ] Logs appear in UI
- [ ] All buttons work
- [ ] Sliders update values
- [ ] Canvas renders properly
- [ ] Worker initializes
- [ ] Training can start
- [ ] No console errors

### Browser Compatibility

Test in:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Common Issues

### CDN Loading
If TensorFlow.js or Matter.js fail to load:
- Check browser console
- Verify internet connection
- Check for ad blockers or content blockers

### Worker Issues
If the Web Worker fails:
- Check console for errors
- Verify `worker.js` is accessible
- Check CORS if serving remotely

## Pull Request Guidelines

1. **Keep changes minimal** - Focus on one feature/fix
2. **Add JSDoc comments** - Document all new functions
3. **Test manually** - Verify in browser
4. **Run type check** - No TypeScript errors
5. **Run linter** - Clean ESLint output
6. **Update docs** - If adding features

## Questions?

Feel free to open an issue for:
- Feature requests
- Bug reports
- Architecture questions
- Documentation improvements

Thank you for contributing! 🎉
