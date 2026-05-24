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
