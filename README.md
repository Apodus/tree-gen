# tree-gen

Procedural tree generator. Produces multi-LOD GLB meshes with bark, leaves, and wind weight data.

## Species

Four species with distinct branching, leaf geometry, and textures:
- **Oak** — oblate spheroid canopy, lobed leaves, BentCrossCluster geometry
- **Pine** — conical canopy, branch-aligned needle strips (BranchStrip)
- **Birch** — oblate spheroid canopy, serrated leaves, BentCrossCluster geometry
- **Maple** — oblate spheroid canopy, star-shaped leaves, BentCrossCluster geometry

## Usage

```
rynx-treegen --species oak --seed=42 --out=oak_42.glb
rynx-treegen --scenario=scenarios/c3_pine.json --seed=123 --out=pine_123.glb
```

## Architecture

Space-colonization skeleton growth (Runions 2007) → branch mesh (cylinder extrusion + Hermite fork-blend collar rings) → leaf placement (BranchWalk along skeleton segments) → leaf geometry (BentCrossCluster / BranchStrip) → multi-LOD GLB emission.

All procedural: bark textures (simplex FBM), leaf atlas (polygon rasterizer + vein patterns), needle-strip textures. No external asset dependencies.

## Data root

`rynx-treegen` and the test executables find `scenarios/` (and the rest of this repository) from their own
location, never from the cwd: the build compiles in the *relative* path from the exe's directory to the repository
root (`TREEGEN_ROOT_FROM_EXE`; CMake and rynx's Sharpmake define it), and the root must hold the committed
`.treegen-root` marker (`treegen_root.hpp`). Any copy of the tree that keeps the exe-to-root layout resolves to
itself. `rynx-treegen --print-root` prints the root an exe found.

## Standalone tests

```
cmake -S . -B build -G Ninja && cmake --build build
build/treegen-tests
```
