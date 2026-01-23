Got it. Here’s a **complete, paste-ready schema** that keeps every **Craftax Classic** achievement (IDs **0–21 unchanged**) and appends all **new branch nodes** (IDs **22+**) for the three deep branches you asked for:

* **Branch 1: Metallurgy** (steel/bronze fork)
* **Branch 2: Glass & Ceramics** (optics/masonry fork)
* **Branch 3: Chemistry & Alchemy** (synergizes with glass)
* **Branch 4: Classic Core** (already present; left intact)

I’ve used a compact JSON-ish format with consistent fields so you can load/validate easily.

---

# Achievements (complete list)

```json
[
  /* =======================
     EXISTING (UNCHANGED)
     ======================= */
  {"id":0,"code":"COLLECT_WOOD","action":"DO","requires":{"near":"tree"},"cost":{},"prereqs":[],"stations":[],"tools":[],"gains":{"wood":1},"notes":"Essential first step"},
  {"id":1,"code":"PLACE_TABLE","action":"PLACE_TABLE","requires":{"inv":{"wood":2},"near":"valid"},"cost":{"wood":2},"prereqs":[],"stations":[],"tools":[],"notes":"Enables all MAKE_*"},
  {"id":2,"code":"EAT_COW","action":"DO","requires":{"near":"cow"},"cost":{},"prereqs":[],"stations":[],"tools":[],"combat":{"target_hp":3,"player_dmg_bare":1},"gains":{"food":6},"notes":"Passive mob"},
  {"id":3,"code":"COLLECT_SAPLING","action":"DO","requires":{"near":"grass"},"cost":{},"prereqs":[],"stations":[],"tools":[],"chance":0.10,"gains":{"sapling":1}},
  {"id":4,"code":"COLLECT_DRINK","action":"DO","requires":{"near":"water"},"cost":{},"prereqs":[],"stations":[],"tools":[],"gains":{"drink":1},"notes":"Resets thirst"},
  {"id":5,"code":"MAKE_WOOD_PICKAXE","action":"MAKE_WOOD_PICKAXE","requires":{"inv":{"wood":1},"near_stations":["table"]},"cost":{"wood":1},"prereqs":[1],"stations":["table"],"tools":[],"notes":"Unlocks stone/coal mining"},
  {"id":6,"code":"MAKE_WOOD_SWORD","action":"MAKE_WOOD_SWORD","requires":{"inv":{"wood":1},"near_stations":["table"]},"cost":{"wood":1},"prereqs":[1],"stations":["table"],"weapon":{"dmg":2}},
  {"id":7,"code":"PLACE_PLANT","action":"PLACE_PLANT","requires":{"inv":{"sapling":1},"near":"grass"},"cost":{"sapling":1},"prereqs":[3],"notes":"Grows ~600 ticks"},
  {"id":8,"code":"DEFEAT_ZOMBIE","action":"DO","requires":{"near":"zombie"},"cost":{},"prereqs":[],"combat":{"target_hp":5,"target_dmg":2,"sleep_dmg":7,"cd":5}},
  {"id":9,"code":"COLLECT_STONE","action":"DO","requires":{"near":"stone"},"cost":{},"prereqs":[5],"tools":["wood_pick"],"gains":{"stone":1}},
  {"id":10,"code":"PLACE_STONE","action":"PLACE_STONE","requires":{"inv":{"stone":1},"near":"valid"},"cost":{"stone":1},"prereqs":[9]},
  {"id":11,"code":"EAT_PLANT","action":"DO","requires":{"near":"ripe_plant"},"cost":{},"prereqs":[7],"gains":{"food":4},"notes":"Resets ripe→plant"},
  {"id":12,"code":"DEFEAT_SKELETON","action":"DO","requires":{"near":"skeleton"},"cost":{},"prereqs":[],"combat":{"target_hp":3,"target_dmg":2,"cd":4,"ranged":true}},
  {"id":13,"code":"MAKE_STONE_PICKAXE","action":"MAKE_STONE_PICKAXE","requires":{"inv":{"wood":1,"stone":1},"near_stations":["table"]},"cost":{"wood":1,"stone":1},"prereqs":[5,9],"stations":["table"],"notes":"Unlocks iron"},
  {"id":14,"code":"MAKE_STONE_SWORD","action":"MAKE_STONE_SWORD","requires":{"inv":{"wood":1,"stone":1},"near_stations":["table"]},"cost":{"wood":1,"stone":1},"prereqs":[9],"stations":["table"],"weapon":{"dmg":3}},
  {"id":15,"code":"WAKE_UP","action":"AUTO","requires":{"state":"sleeping","energy_gte":9},"cost":{},"prereqs":[],"notes":"Also wakes on attack"},
  {"id":16,"code":"PLACE_FURNACE","action":"PLACE_FURNACE","requires":{"inv":{"stone":1},"near":"valid"},"cost":{"stone":1},"prereqs":[9],"notes":"Req for iron-tier"},
  {"id":17,"code":"COLLECT_COAL","action":"DO","requires":{"near":"coal"},"cost":{},"prereqs":[5],"tools":["wood_pick"],"gains":{"coal":1}},
  {"id":18,"code":"COLLECT_IRON","action":"DO","requires":{"near":"iron"},"cost":{},"prereqs":[13],"tools":["stone_pick"],"gains":{"iron":1}},
  {"id":19,"code":"COLLECT_DIAMOND","action":"DO","requires":{"near":"diamond"},"cost":{},"prereqs":[20],"tools":["iron_pick"],"gains":{"diamond":1}},
  {"id":20,"code":"MAKE_IRON_PICKAXE","action":"MAKE_IRON_PICKAXE","requires":{"inv":{"wood":1,"stone":1,"iron":1,"coal":1},"near_stations":["table","furnace"]},"cost":{"wood":1,"stone":1,"iron":1,"coal":1},"prereqs":[13,16,17,18],"stations":["table","furnace"],"notes":"Unlocks diamond"},
  {"id":21,"code":"MAKE_IRON_SWORD","action":"MAKE_IRON_SWORD","requires":{"inv":{"wood":1,"stone":1,"iron":1,"coal":1},"near_stations":["table","furnace"]},"cost":{"wood":1,"stone":1,"iron":1,"coal":1},"prereqs":[14,16,17,18],"stations":["table","furnace"],"weapon":{"dmg":5}},

  /* =======================
     NEW — BRANCH 1: METALLURGY
     ======================= */
  {"id":22,"code":"PLACE_ANVIL","action":"PLACE_ANVIL","requires":{"inv":{"stone":2},"near":"valid"},"cost":{"stone":2},"prereqs":[9],"stations_created":["anvil"],"notes":"Enables FORGE_*"},
  {"id":23,"code":"MAKE_BELLOWS","action":"MAKE_BELLOWS","requires":{"inv":{"wood":1,"leather":1},"near_stations":["table"]},"cost":{"wood":1,"leather":1},"prereqs":[1,2],"stations":["table"],"notes":"Furnace smelt time −30% if adjacent"},
  {"id":24,"code":"SMELT_IRON_BAR","action":"DO","requires":{"near_stations":["furnace"],"inv":{"iron":1,"coal":1}},"cost":{"iron":1,"coal":1},"prereqs":[16,17,18],"gains":{"iron_bar":1},"notes":"Intermediate for steel tools"},
  {"id":25,"code":"COLLECT_COPPER","action":"DO","requires":{"near":"copper"},"cost":{},"prereqs":[13],"tools":["stone_pick"],"gains":{"copper":1}},
  {"id":26,"code":"COLLECT_TIN","action":"DO","requires":{"near":"tin"},"cost":{},"prereqs":[13],"tools":["stone_pick"],"gains":{"tin":1}},
  {"id":27,"code":"MAKE_STEEL_BAR","action":"DO","requires":{"near_stations":["furnace"],"adjacent":["bellows_optional"],"inv":{"iron_bar":1,"coal":1}},"cost":{"iron_bar":1,"coal":1},"prereqs":[23,24],"gains":{"steel_bar":1}},
  {"id":28,"code":"MAKE_BRONZE_BAR","action":"DO","requires":{"near_stations":["furnace"],"inv":{"copper":1,"tin":1,"coal":1}},"cost":{"copper":1,"tin":1,"coal":1},"prereqs":[16,17,25,26],"gains":{"bronze_bar":1}},
  {"id":29,"code":"FORGE_STEEL_PICKAXE","action":"FORGE_STEEL_PICKAXE","requires":{"near_stations":["anvil","table"],"inv":{"wood":1,"stone":1,"steel_bar":1}},"cost":{"wood":1,"stone":1,"steel_bar":1},"prereqs":[22,27],"stations":["anvil","table"],"tools_created":["steel_pick"],"notes":"+1 mining speed tier vs iron"},
  {"id":30,"code":"FORGE_BRONZE_PICKAXE","action":"FORGE_BRONZE_PICKAXE","requires":{"near_stations":["anvil","table"],"inv":{"wood":1,"stone":1,"bronze_bar":1}},"cost":{"wood":1,"stone":1,"bronze_bar":1},"prereqs":[22,28],"stations":["anvil","table"],"tools_created":["bronze_pick"],"notes":"Between stone and iron speed"},
  {"id":31,"code":"HARDEN_EDGE","action":"DO","requires":{"near":"water","holding_any":["steel_pick","bronze_pick"]},"cost":{},"prereqs":[29,30],"effects":{"tool_buff":{"durability_mult":1.2,"speed_mult":1.2}},"notes":"Quench while holding tool"},
  {"id":32,"code":"FORGE_STEEL_SWORD","action":"FORGE_STEEL_SWORD","requires":{"near_stations":["anvil","table"],"inv":{"wood":1,"steel_bar":1}},"cost":{"wood":1,"steel_bar":1},"prereqs":[27,22],"weapon":{"dmg":6}},
  {"id":33,"code":"FORGE_BRONZE_SWORD","action":"FORGE_BRONZE_SWORD","requires":{"near_stations":["anvil","table"],"inv":{"wood":1,"bronze_bar":1}},"cost":{"wood":1,"bronze_bar":1},"prereqs":[28,22],"weapon":{"dmg":4}},

  /* =======================
     NEW — BRANCH 2: GLASS & CERAMICS
     ======================= */
  {"id":34,"code":"COLLECT_SAND","action":"DO","requires":{"near":"sand"},"cost":{},"prereqs":[],"gains":{"sand":1}},
  {"id":35,"code":"PLACE_KILN","action":"PLACE_KILN","requires":{"inv":{"stone":1},"near":"valid"},"cost":{"stone":1},"prereqs":[9],"stations_created":["kiln"],"notes":"Smelts glass/ceramics"},
  {"id":36,"code":"SMELT_GLASS","action":"DO","requires":{"near_stations":["kiln"],"inv":{"sand":1,"coal":1}},"cost":{"sand":1,"coal":1},"prereqs":[34,35,17],"gains":{"glass":1}},
  {"id":37,"code":"COLLECT_CLAY","action":"DO","requires":{"near":"clay"},"cost":{},"prereqs":[],"gains":{"clay":1},"notes":"Riverbank block"},
  {"id":38,"code":"FIRE_BRICK","action":"DO","requires":{"near_stations":["kiln"],"inv":{"clay":1,"coal":1}},"cost":{"clay":1,"coal":1},"prereqs":[35,37,17],"gains":{"brick":1}},
  {"id":39,"code":"MAKE_GLASS_BOTTLE","action":"MAKE_GLASS_BOTTLE","requires":{"near_stations":["kiln"],"inv":{"glass":1}},"cost":{"glass":1},"prereqs":[36],"gains":{"bottle":1},"notes":"+2 drink container"},
  {"id":40,"code":"MAKE_LENS","action":"MAKE_LENS","requires":{"near_stations":["kiln","table"],"inv":{"glass":1}},"cost":{"glass":1},"prereqs":[36,1,35],"gains":{"lens":1}},
  {"id":41,"code":"COLLECT_LIMESTONE","action":"DO","requires":{"near":"limestone"},"cost":{},"prereqs":[5],"tools":["wood_pick"],"gains":{"limestone":1}},
  {"id":42,"code":"MAKE_LIME","action":"DO","requires":{"near_stations":["kiln"],"inv":{"limestone":1,"coal":1}},"cost":{"limestone":1,"coal":1},"prereqs":[41,35,17],"gains":{"lime":1}},
  {"id":43,"code":"MAKE_TELESCOPE","action":"MAKE_TELESCOPE","requires":{"near_stations":["table"],"inv":{"lens":1,"wood":1}},"cost":{"lens":1,"wood":1},"prereqs":[40,1],"gains":{"telescope":1},"notes":"+3 reveal radius"},
  {"id":44,"code":"MAKE_MORTAR","action":"MAKE_MORTAR","requires":{"near_stations":["table"],"inv":{"lime":1,"sand":1}},"cost":{"lime":1,"sand":1},"prereqs":[42,34,1],"gains":{"mortar":1}},
  {"id":45,"code":"PLACE_WINDOW","action":"PLACE_WINDOW","requires":{"inv":{"glass":1},"near":"valid"},"cost":{"glass":1},"prereqs":[36],"notes":"Indoor spawn −1"},
  {"id":46,"code":"PLACE_WALL_MASONRY","action":"PLACE_WALL_MASONRY","requires":{"inv":{"mortar":1},"near":"valid"},"cost":{"mortar":1},"prereqs":[44],"notes":"Safer sleep: −1 dmg while sleeping indoors"},

  /* =======================
     NEW — BRANCH 3: CHEMISTRY & ALCHEMY
     ======================= */
  {"id":47,"code":"PLACE_COMPOSTER","action":"PLACE_COMPOSTER","requires":{"inv":{"wood":1},"near":"valid"},"cost":{"wood":1},"prereqs":[1],"stations_created":["composter"]},
  {"id":48,"code":"MAKE_TAR","action":"DO","requires":{"near_stations":["kiln"],"inv":{"wood":1}},"cost":{"wood":1},"prereqs":[35],"gains":{"tar":1}},
  {"id":49,"code":"MAKE_FERTILIZER","action":"MAKE_FERTILIZER","requires":{"near_stations":["composter"],"inv":{"plant":1,"tar":1}},"cost":{"plant":1,"tar":1},"prereqs":[47,48],"gains":{"fertilizer":1},"notes":"Apply to plot/plant → growth −30%"},
  {"id":50,"code":"PLACE_ALCHEMY_BENCH","action":"PLACE_ALCHEMY_BENCH","requires":{"inv":{"stone":1},"near":"valid"},"cost":{"stone":1},"prereqs":[9],"stations_created":["alchemy_bench"]},
  {"id":51,"code":"BREW_TONIC","action":"DO","requires":{"near_stations":["alchemy_bench"],"inv":{"bottle":1,"plant":1}},"cost":{"bottle":1,"plant":1},"prereqs":[39,50],"gains":{"tonic_basic":1},"notes":"Minor regen or thirst buffer (config)"},
  {"id":52,"code":"BREW_STONE_SKIN","action":"DO","requires":{"near_stations":["alchemy_bench"],"inv":{"bottle":1,"tar":1}},"cost":{"bottle":1,"tar":1},"prereqs":[50,48,39],"gains":{"tonic_stoneskin":1},"effects":{"buff":{"dmg_reduction":1,"duration_turns":N}}},
  {"id":53,"code":"MAKE_FLUX","action":"MAKE_FLUX","requires":{"near_stations":["alchemy_bench"],"inv":{"lime":1,"tar":1}},"cost":{"lime":1,"tar":1},"prereqs":[42,48,50],"gains":{"flux":1},"notes":"Place beside furnace: smelt −20% (stacks with bellows)"}
]
```

---

## Stations (reference)

```json
{
  "table":"crafting table",
  "furnace":"metal smelter (existing)",
  "anvil":"forging station (new)",
  "kiln":"glass/ceramic kiln (new)",
  "composter":"organic processor (new)",
  "alchemy_bench":"brewing/chemistry bench (new)"
}
```

---

## Blocks & Mining Tool Requirements (extended)

```json
[
  {"block":"tree","tool_required":"none"},
  {"block":"stone","tool_required":"wood_pick"},
  {"block":"coal","tool_required":"wood_pick"},
  {"block":"iron","tool_required":"stone_pick"},
  {"block":"diamond","tool_required":"iron_pick"},
  {"block":"sand","tool_required":"none"},
  {"block":"clay","tool_required":"none"},
  {"block":"limestone","tool_required":"wood_pick"},
  {"block":"copper","tool_required":"stone_pick"},
  {"block":"tin","tool_required":"stone_pick"}
]
```

---

## Items (new craftables/intermediates)

```json
[
  "iron_bar","steel_bar","bronze_bar",
  "glass","bottle","lens","telescope",
  "brick","lime","mortar",
  "tar","fertilizer","flux"
]
```

---

## Quick Resource Summary (delta over Classic)

| New Achievement Code   | Wood | Stone | Coal | Iron | Copper | Tin | Sand | Clay | Limestone | Leather | Stations Needed        |
| ---------------------- | ---: | ----: | ---: | ---: | -----: | --: | ---: | ---: | --------: | ------: | ---------------------- |
| PLACE\_ANVIL           |    0 |     2 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | —                      |
| MAKE\_BELLOWS          |    1 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       1 | Table                  |
| SMELT\_IRON\_BAR       |    0 |     0 |    1 |    1 |      0 |   0 |    0 |    0 |         0 |       0 | Furnace                |
| MAKE\_STEEL\_BAR       |    0 |     0 |    1 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Furnace (+Bellows adj) |
| MAKE\_BRONZE\_BAR      |    0 |     0 |    1 |    0 |      1 |   1 |    0 |    0 |         0 |       0 | Furnace                |
| FORGE\_STEEL\_PICKAXE  |    1 |     1 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Anvil + Table          |
| FORGE\_BRONZE\_PICKAXE |    1 |     1 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Anvil + Table          |
| PLACE\_KILN            |    0 |     1 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | —                      |
| SMELT\_GLASS           |    0 |     0 |    1 |    0 |      0 |   0 |    1 |    0 |         0 |       0 | Kiln                   |
| MAKE\_GLASS\_BOTTLE    |    0 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Kiln                   |
| FIRE\_BRICK            |    0 |     0 |    1 |    0 |      0 |   0 |    0 |    1 |         0 |       0 | Kiln                   |
| MAKE\_LENS             |    0 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Kiln + Table           |
| MAKE\_LIME             |    0 |     0 |    1 |    0 |      0 |   0 |    0 |    0 |         1 |       0 | Kiln                   |
| MAKE\_TELESCOPE        |    1 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Table                  |
| MAKE\_MORTAR           |    0 |     0 |    0 |    0 |      0 |   0 |    1 |    0 |         0 |       0 | Table                  |
| PLACE\_WINDOW          |    0 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | —                      |
| PLACE\_WALL\_MASONRY   |    0 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | —                      |
| PLACE\_COMPOSTER       |    1 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | —                      |
| MAKE\_TAR              |    1 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Kiln                   |
| MAKE\_FERTILIZER       |    0 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Composter              |
| PLACE\_ALCHEMY\_BENCH  |    0 |     1 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | —                      |
| BREW\_TONIC            |    0 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Alchemy Bench          |
| BREW\_STONE\_SKIN      |    0 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Alchemy Bench          |
| MAKE\_FLUX             |    0 |     0 |    0 |    0 |      0 |   0 |    0 |    0 |         0 |       0 | Alchemy Bench          |

---

## Notes for implementation

* **Stacking smelt speed buffs**:

  * Bellows (adjacent to furnace): **−30%** time.
  * Flux (placed beside furnace): **−20%** time.
  * Multiplicative stacking keeps returns sane.

* **Fork visibility**: surface both **steel** and **bronze** as separate achievements so players see the choice.

* **Bottles**: decide if bottles are consumed on use or returned empty; schema supports either.

