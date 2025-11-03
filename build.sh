#!/bin/bash

# Ensure we use rustup's toolchain
export PATH="$HOME/.cargo/bin:$PATH"

# Build the WebAssembly module
wasm-pack build --target web --out-dir pkg

echo "Build complete! Open index.html in a browser to run the app."
echo ""
echo "To serve locally, you can run:"
echo "  python3 -m http.server 8000"
echo "  or"
echo "  npx serve ."
