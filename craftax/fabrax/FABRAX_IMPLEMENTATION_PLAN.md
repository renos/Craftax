# Fabrax Implementation Plan

## Overview
Fabrax extends Craftax Classic with 32 new achievements (IDs 22-53) across three main branches:
1. **Metallurgy** (steel/bronze tools and advanced smithing)
2. **Glass & Ceramics** (optics and masonry)
3. **Chemistry & Alchemy** (brewing and farming enhancements)

## Implementation Strategy

### Phase 1: Core Infrastructure Updates

#### 1.1 Constants & Enums Updates (`constants.py`)
- **New BlockTypes**: Add 6 new block types
  - `COPPER = 17` (mines with stone pickaxe)
  - `TIN = 18` (mines with stone pickaxe)
  - `SAND = 19` (mines with bare hands, already exists but extend)
  - `CLAY = 20` (mines with bare hands)
  - `LIMESTONE = 21` (mines with wood pickaxe)
  - `ANVIL = 22`, `KILN = 23`, `COMPOSTER = 24`, `ALCHEMY_BENCH = 25`

- **New Actions**: Add 32 new action types (IDs 63-94)
  - Placing: `PLACE_ANVIL`, `PLACE_KILN`, `PLACE_COMPOSTER`, `PLACE_ALCHEMY_BENCH`
  - Making: `MAKE_BELLOWS`, `MAKE_GLASS_BOTTLE`, `MAKE_LENS`, etc.
  - Forging: `FORGE_STEEL_PICKAXE`, `FORGE_BRONZE_PICKAXE`, etc.
  - Smelting: `SMELT_IRON_BAR`, `MAKE_STEEL_BAR`, `SMELT_GLASS`, etc.
  - Brewing: `BREW_TONIC`, `BREW_STONE_SKIN`
  - Special: `HARDEN_EDGE`, `PLACE_WINDOW`, `PLACE_WALL_MASONRY`

- **New Achievements**: Add 32 achievement IDs (22-53)

#### 1.2 Inventory Extension (`envs/craftax_state.py`)
- **New items in Inventory dataclass**:
  ```python
  # Raw materials
  copper: int = 0
  tin: int = 0
  sand: int = 0
  clay: int = 0
  limestone: int = 0
  leather: int = 0  # From cows

  # Intermediate materials
  iron_bar: int = 0
  steel_bar: int = 0
  bronze_bar: int = 0
  glass: int = 0
  brick: int = 0
  lime: int = 0
  tar: int = 0

  # Crafted items
  bottle: int = 0
  lens: int = 0
  telescope: int = 0
  mortar: int = 0
  fertilizer: int = 0
  flux: int = 0

  # Tools (new pickaxes)
  steel_pickaxe: int = 0
  bronze_pickaxe: int = 0

  # Potions/Tonics
  tonic_basic: int = 0
  tonic_stoneskin: int = 0
  ```

#### 1.3 State Extensions
- **Station tracking**: Track placed anvils, kilns, composters, alchemy benches
- **Buff system**: Add player buffs for stoneskin, tool hardening
- **Speed modifiers**: Track furnace speed bonuses from bellows/flux

### Phase 2: World Generation (`world_gen.py`)

#### 2.1 New Block Generation
- **Copper ore**: Generate near stone (similar distribution)
- **Tin ore**: Generate near stone (rarer than copper)
- **Clay**: Generate near water (riverbank replacement logic)
- **Limestone**: Generate in stone-rich areas
- **Sand**: Already exists, ensure proper distribution

### Phase 3: Game Logic Extensions (`game_logic.py`)

#### 3.1 Mining Updates
- Extend `do_action()` to handle new mineable blocks:
  - Copper/tin (require stone pickaxe)
  - Clay (bare hands)
  - Limestone (wood pickaxe)
  - Add leather drops from cows (when killed)

#### 3.2 Smelting System
- **New smelting logic**:
  - Iron → Iron Bar (furnace + coal)
  - Iron Bar + Coal → Steel Bar (furnace + bellows adjacent)
  - Copper + Tin + Coal → Bronze Bar (furnace)
  - Sand + Coal → Glass (kiln)
  - Clay + Coal → Brick (kiln)
  - Limestone + Coal → Lime (kiln)
  - Wood → Tar (kiln)

#### 3.3 Advanced Crafting
- **Forge system** (anvil + table required):
  - Steel/Bronze pickaxes and swords
- **Glass crafting** (kiln required):
  - Bottles, lenses
- **Chemistry** (alchemy bench):
  - Tonics and potions

#### 3.4 Station Placement
- Add placement logic for 4 new station types
- **Adjacent bonuses**:
  - Bellows next to furnace: -30% smelt time
  - Flux next to furnace: -20% smelt time (stacks)

#### 3.5 Special Mechanics
- **Tool hardening**: Quench steel/bronze tools in water for +20% durability/speed
- **Building system**: Windows and masonry walls for safety bonuses
- **Buff system**: Stoneskin potion reduces damage

### Phase 4: Rendering & Assets (`renderer.py` + `assets/`)

#### 4.1 New Textures Needed
- **Blocks**: copper.png, tin.png, clay.png, limestone.png, anvil.png, kiln.png, composter.png, alchemy_bench.png
- **Items**: All new inventory items (30+ new textures)
- **Tools**: steel_pickaxe.png, bronze_pickaxe.png, steel_sword.png, bronze_sword.png

#### 4.2 Rendering Updates
- Update texture loading to handle new block types
- Add inventory rendering for new items
- Visual effects for buffs/debuffs

### Phase 5: Environment Integration

#### 5.1 Environment Class Updates
- Update craftax_symbolic_env.py to handle new action space
- Extend observation space for new inventory items
- Update reward calculation for new achievements

#### 5.2 Flow Environment Integration
- Update craftax_flow_symbolic_env.py for Flow-RL compatibility
- Ensure new achievements work with existing skill progression system

## Implementation Order

### Priority 1 (Core Systems)
1. ✅ Copy base files
2. Update `constants.py` with all new enums
3. Extend `Inventory` dataclass in `craftax_state.py`
4. Update `game_logic.py` basic mining for new blocks

### Priority 2 (Crafting Systems)
5. Implement smelting system in `game_logic.py`
6. Add station placement logic
7. Implement forge crafting mechanics
8. Add chemistry/alchemy brewing

### Priority 3 (World & Rendering)
9. Update `world_gen.py` for new block generation
10. Create placeholder textures in `assets/`
11. Update `renderer.py` for new blocks/items

### Priority 4 (Polish & Integration)
12. Add buff system and special mechanics
13. Update environment classes
14. Testing and balancing

## Key Technical Considerations

### Backwards Compatibility
- All original 22 achievements (IDs 0-21) remain unchanged
- Existing block types and mechanics preserved
- New systems additive, not replacing existing ones

### Performance
- New inventory items use same int-based system
- Station adjacency checks use existing `is_near_block()` pattern
- Minimal impact on core game loop

### Extensibility
- Achievement system easily expandable beyond 53
- Station system designed for future additions
- Buff system framework for more complex effects

### Balance Considerations
- Steel/bronze tools provide meaningful upgrades without being overpowered
- Resource costs scale appropriately with benefit
- Multiple viable progression paths (metallurgy vs ceramics vs alchemy)

## File Modification Summary

| File | Changes |
|------|---------|
| `constants.py` | +6 block types, +32 actions, +32 achievements |
| `craftax_state.py` | +20 inventory items, +buff tracking |
| `game_logic.py` | +smelting, +forging, +brewing, +new mining |
| `world_gen.py` | +5 new block generation algorithms |
| `renderer.py` | +texture loading for 30+ new items |
| `assets/` | +30+ new texture files |
| Environment files | +action space, +observation space updates |

This implementation preserves all existing Craftax Classic functionality while adding the three deep progression branches specified in the Fabrax design document.