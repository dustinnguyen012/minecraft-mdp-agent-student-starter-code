# Action Reference

Complete list of the 186 bot actions (indices 0-185) available in the Minecraft MDP. The bridge defines all actions server-side; `MinecraftMDPEnv` fetches them at startup so `env.num_actions` and `env.ACTION_NAMES` are always in sync.

## EMA Speed Groups

Actions are grouped by how long they typically take. The bridge measures actual execution time per action and maintains an exponential moving average (EMA) per group.

| Group | Typical | What's in it |
|-------|---------|--------------|
| fast | ~350 ms | movement, noop, equip_armor, toggle_door, drop_\*, mount, dismount |
| medium | ~1 s | craft_\*, attack_nearest, eat, feed, place, chop (axe), smelt_\*, shear/milk, shoot_arrow, shield, potion, climb_up, pick_up_\* |
| slow | ~3 s | mine_\*, mine_stone, get_wood, sleep, hunt_\* |
| very_slow | up to 45 s | fish only |

## All Actions (0-185)

| # | Name | Group | Requirements / Notes |
|---|------|-------|----------------------|
| 0 | move_north | fast | Always available. Walk +Z one block. |
| 1 | move_south | fast | Always available. Walk -Z one block. |
| 2 | move_east | fast | Always available. Walk +X one block. |
| 3 | move_west | fast | Always available. Walk -X one block. |
| 4 | mine_below | slow | Non-air block below + matching tool (shovel for dirt-family, pickaxe for stone-family, axe for wood). Equips the correct tool before digging. Blocks consecutive mine_below to prevent bottomless pits. |
| 5 | mine_forward | slow | **Pickaxe-only.** Pickaxe-tier block at foot-level (y=0) in front AND a matching pickaxe in inventory. For shovel-family blocks use `dig_forward` (69). For other vertical positions see actions 162-167. |
| 6 | place_forward | medium | Item in inventory. Places the first stack on the block in front. |
| 7 | noop | fast | Always available. No-op; advances the clock only. |
| 8 | craft_planks | medium | Any log in inventory. Any log variant accepted (oak, spruce, birch, etc.). |
| 9 | craft_sticks | medium | 2+ planks. |
| 10 | craft_crafting_table | medium | 4+ planks. Crafts, then auto-places the block. |
| 11 | craft_wooden_pickaxe | medium | 3 planks + 2 sticks + crafting table nearby. |
| 12 | craft_stone_pickaxe | medium | 3 cobblestone (or blackstone / cobbled_deepslate, mixable) + 2 sticks + crafting table nearby. |
| 13 | craft_wooden_sword | medium | 2 planks + 1 stick + crafting table nearby. |
| 14 | attack_nearest | medium | Hostile mob within 5 blocks. Equips sword if present. |
| 15 | eat | medium | Food in inventory + hunger not full. |
| 16 | feed_animal | medium | Animal within 4 blocks + matching food (seeds for chicken, wheat for cow, carrot/potato for pig). |
| 17 | get_wood | slow | Log block within 5 blocks. Bare-hand (slow). |
| 18 | chop_wood | medium | Log block within 5 blocks + axe in inventory (fast). |
| 19 | craft_wooden_axe | medium | 3 planks + 2 sticks + crafting table nearby. |
| 20 | craft_wooden_shovel | medium | 1 plank + 2 sticks + crafting table nearby. |
| 21 | craft_furnace | medium | 8 cobblestone (or blackstone / cobbled_deepslate, mixable) + crafting table nearby. Unlocks smelting. |
| 22 | craft_stone_sword | medium | 2 cobblestone + 1 stick + crafting table nearby. |
| 23 | mine_coal | slow | Scans 6-block radius. Requires wooden+ pickaxe. Matches `coal_ore` and deepslate variant. |
| 24 | mine_iron | slow | Scans 6-block radius. Requires **stone+** pickaxe. |
| 25 | mine_copper | slow | Scans 6-block radius. Requires stone+ pickaxe. |
| 26 | mine_gold | slow | Scans 6-block radius. Requires **iron+** pickaxe. Matches overworld + nether gold ore. |
| 27 | mine_redstone | slow | Scans 6-block radius. Requires iron+ pickaxe. |
| 28 | mine_lapis | slow | Scans 6-block radius. Requires stone+ pickaxe. |
| 29 | mine_diamond | slow | Scans 6-block radius. Requires iron+ pickaxe. |
| 30 | mine_emerald | slow | Scans 6-block radius. Requires iron+ pickaxe. |
| 31 | farm_nearest | medium | Harvests nearest fully-mature crop (wheat age=7, carrots age=7, potatoes age=7, beetroots age=3) within 6 blocks. |
| 32 | craft_torch | medium | 1 coal or charcoal + 1 stick = 4 torches. 2x2 recipe, no crafting table needed. |
| 33 | place_torch | medium | Torch in inventory. Prefers wall-in-front, falls back to floor. |
| 34 | equip_armor | fast | Armor piece in inventory that beats what's currently worn. Loops all four slots (head/torso/legs/feet), picks highest tier. |
| 35 | sleep | slow | Night time + bed within 8 blocks + no hostile mob within 8 blocks. |
| 36 | smelt_start | medium | **Minor smelt channel.** Furnace within 4 + fuel + one of: raw_copper, raw_gold, copper_ore, gold_ore (or deepslate variants), sand, red_sand, clay_ball. Opens furnace, deposits input + fuel, closes. For iron use action 100; for cooking use action 98; for logs/cobblestone/stone use actions 106-108. |
| 37 | collect_cooked | medium | **Session-gated.** Furnace session must be open (via peek_furnace, action 102) AND the output slot has an item. Pulls whatever finished product is there (cooked food, ingots, charcoal, stone, smooth_stone, etc.) and closes the session. |
| 38 | fish | very_slow | Fishing rod in inventory + water within 3 blocks. Can take 5-45 seconds. |
| 39 | shear_sheep | medium | Shears in inventory + sheep within 4 blocks. |
| 40 | milk_cow | medium | Bucket in inventory + cow within 4 blocks. Bucket becomes milk_bucket. |
| 41 | toggle_door | fast | Door / trapdoor / fence_gate within 4 blocks. |
| 42 | drop_cobblestone | fast | >=64 cobblestone in inventory AND inventory nearly full (>=30 slots used). Drops up to 32. |
| 43 | drop_gravel | fast | >=64 gravel + inventory nearly full. Drops up to 32. |
| 44 | drop_dirt | fast | >=64 dirt + inventory nearly full. Drops up to 32. |
| 45 | shoot_arrow | medium | Bow + arrow in inventory + mob within 10 blocks. Full-draw charge. |
| 46 | raise_shield | medium | Shield in inventory + hostile mob within 4 blocks. Holds block for 1 second. |
| 47 | drink_potion | medium | Healing/regen potion in inventory + health < 20. |
| 48 | mount | fast | Rideable entity (tamed horse, saddled pig, boat, minecart) within 3 blocks + not already mounted. |
| 49 | dismount | fast | Currently mounted on something. |
| 50 | climb_up | medium | Pillar-jump. Needs dirt, cobblestone, gravel, or stone in inventory + 2 blocks of air above + standing on solid ground. |
| 51 | hunt_cow | slow | Cow within 10 blocks + weapon. Pursues up to 10 seconds. Drops: raw_beef, leather. |
| 52 | hunt_pig | slow | Pig within 10 blocks + weapon. Drops: raw_porkchop. |
| 53 | hunt_chicken | slow | Chicken within 10 blocks + weapon. Drops: raw_chicken, feather. |
| 54 | hunt_sheep | slow | Sheep within 10 blocks + weapon. Drops: raw_mutton, wool. |
| 55 | hunt_rabbit | slow | Rabbit within 10 blocks + weapon. Drops: raw_rabbit, rabbit_hide. |
| 56 | attack_villager | slow | **DISABLED by default.** |
| 57 | attack_player | slow | **DISABLED by default** (requires `PVP_ENABLED=true` on the server). |
| 58 | pick_up_meat | medium | Walks to nearest dropped meat item (raw + cooked) within 8 blocks. |
| 59 | pick_up_ore | medium | Walks to nearest dropped ore/ingot/gem within 8 blocks (coal, iron, copper, gold, redstone, lapis, diamond, emerald, quartz). |
| 60 | pick_up_material | medium | Walks to nearest dropped building material within 8 blocks (logs, planks, stone, cobblestone, dirt, sticks, crops, etc.). |
| 61 | place_crafting_table | medium | Crafting table in inventory. Scans for a valid placement face. |
| 62 | place_furnace | medium | Furnace in inventory. |
| 63 | place_cobblestone | medium | Cobblestone in inventory. |
| 64 | place_stone | medium | Stone (smelted cobblestone) in inventory. |
| 65 | place_plank | medium | Any planks in inventory. |
| 66 | place_log | medium | Any log in inventory. |
| 67 | place_redstone | medium | Redstone in inventory. |
| 68 | place_dirt | medium | Dirt in inventory. |
| 69 | dig_forward | medium | **Shovel-family only.** Dirt / grass / sand / gravel / clay / soul_sand at foot-level (y=0) in front. Equips shovel if available; otherwise bare-hand. For other vertical positions see actions 161-167. |
| 70 | mine_stone | slow | Targeted mine for `stone` and `cobblestone` within 6-block radius (granite/diorite/andesite excluded). Requires any pickaxe. Primary cobblestone source. |
| 71 | craft_wooden_hoe | medium | 2 planks + 2 sticks + crafting table nearby. Unlocks `till_soil` (73). |
| 72 | craft_stone_hoe | medium | 2 cobblestone + 2 sticks + crafting table nearby. |
| 73 | till_soil | medium | Hoe in inventory + dirt / grass_block / coarse_dirt / dirt_path nearby with air above. Converts the block to farmland. Confirms the transition actually happened (silent failures are detected). |
| 74 | plant_seed | medium | Seed in inventory (wheat_seeds / beetroot_seeds / carrot / potato / melon_seeds / pumpkin_seeds) + farmland within 4 blocks. Places seed on nearest farmland. |
| 75 | craft_stone_axe | medium | 3 cobblestone + 2 sticks + crafting table nearby. |
| 76 | craft_stone_shovel | medium | 1 cobblestone + 2 sticks + crafting table nearby. |
| 77 | craft_iron_pickaxe | medium | 3 iron_ingot + 2 sticks + crafting table nearby. **Unlocks mine_gold / mine_redstone / mine_diamond / mine_emerald.** |
| 78 | craft_iron_axe | medium | 3 iron_ingot + 2 sticks + crafting table nearby. |
| 79 | craft_iron_sword | medium | 2 iron_ingot + 1 stick + crafting table nearby. |
| 80 | craft_iron_shovel | medium | 1 iron_ingot + 2 sticks + crafting table nearby. |
| 81 | craft_iron_hoe | medium | 2 iron_ingot + 2 sticks + crafting table nearby. |
| 82 | craft_diamond_pickaxe | medium | 3 diamond + 2 sticks + crafting table nearby. Endgame pickaxe. |
| 83 | craft_diamond_axe | medium | 3 diamond + 2 sticks + crafting table nearby. |
| 84 | craft_diamond_sword | medium | 2 diamond + 1 stick + crafting table nearby. |
| 85 | craft_diamond_shovel | medium | 1 diamond + 2 sticks + crafting table nearby. |
| 86 | craft_diamond_hoe | medium | 2 diamond + 2 sticks + crafting table nearby. |
| 87 | craft_bread | medium | 3 wheat + crafting table nearby. Reliable food source from the farming chain. |
| 88 | craft_chest | medium | 8 planks (any variant) + crafting table nearby. Enables deposit/withdraw actions (111-160). |
| 89 | craft_bed | medium | 3 wool (any colors, mixable) + 3 planks (any variants, mixable) + crafting table nearby. Output: white_bed. Enables `sleep` (35). |
| 90 | craft_shield | medium | 6 planks + 1 iron_ingot + crafting table nearby. |
| 91 | craft_bow | medium | 3 sticks + 3 string + crafting table nearby. Enables `shoot_arrow` (45). |
| 92 | craft_flint_and_steel | medium | 1 iron_ingot + 1 flint + crafting table nearby. |
| 93 | craft_bucket | medium | 3 iron_ingot + crafting table nearby. Enables `milk_cow` (40). |
| 94 | pick_up_planks | medium | Any `*_planks` block within 3 blocks. **Barehand-safe escape action.** Uses axe if available, otherwise bare hand. |
| 95 | hit_vegetation | medium | Leaves, vines, cobweb, dead_bush, glow_lichen, or mangrove_roots within 3 blocks. **Barehand-safe escape action.** Excludes grass/fern (seed source), seagrass/kelp (food source), saplings, and flowers. |
| 96 | break_nodrop | slow | Any non-air/bedrock/water/lava block in front or below. **Universal escape action.** Uses whatever is currently in hand. Works on stone without a pickaxe (~7.5s bare-hand dig). |
| 97 | place_chest | medium | Chest in inventory + valid placement face + no other chest within 4 blocks. |
| 98 | cook_start | medium | **Cook channel.** Furnace within 4 + fuel + at least one raw food (beef / porkchop / chicken / mutton / rabbit / cod / salmon / potato / kelp). Opens furnace, deposits food + fuel, closes. |
| 99 | *(reserved)* | medium | Index retained for compatibility; not currently active. |
| 100 | smelt_iron_start | medium | **Iron channel.** Furnace within 4 + fuel + raw_iron or iron_ore or deepslate_iron_ore. |
| 101 | *(reserved)* | medium | Index retained for compatibility; not currently active. |
| 102 | peek_furnace | medium | Furnace within 4 + no existing furnace session. Opens the furnace and **holds it open**. While the session is active, the bot's available actions are restricted to `{close_furnace, collect_cooked}` depending on what output is ready. Session auto-closes after 60 seconds. |
| 103 | close_furnace | fast | Always available when a furnace session is active. Closes the session and restores the normal action set. |
| 104 | craft_smoker | medium | 4 logs (any variants, mixable) + 1 furnace + crafting table nearby. Smokers cook food 2x faster than regular furnaces. |
| 105 | place_smoker | medium | Smoker in inventory + valid placement face + no other smoker within 4 blocks. |
| 106 | smoke_wood | medium | **Charcoal channel.** Furnace within 4 + fuel + any log variant. Smelts log into charcoal. Collect via `collect_cooked` (37). |
| 107 | smelt_cobblestone | medium | **Stone channel.** Furnace within 4 + fuel + cobblestone. Smelts cobblestone into stone. Collect via `collect_cooked` (37). |
| 108 | smelt_stone | medium | **Smooth stone channel.** Furnace within 4 + fuel + stone. Smelts stone into smooth_stone. Collect via `collect_cooked` (37). Required for `craft_blast_furnace` (109). |
| 109 | craft_blast_furnace | medium | 5 iron_ingot + 1 furnace + 3 smooth_stone + crafting table nearby. Blast furnaces smelt ores 2x faster. |
| 110 | place_blast_furnace | medium | Blast furnace in inventory + valid placement face + no other blast_furnace within 4 blocks. |

### Chest deposit/withdraw family (111-160)

50 actions covering bulk materials, food, and a full 4x5 tool matrix. Every deposit has a matching withdraw. All gates are server-side: deposits check bot inventory, withdrawals check chest contents.

The bridge keeps a per-bot **chest session** so consecutive deposit/withdraw actions batch efficiently. The first chest action opens the nearest chest (~800 ms); subsequent chest actions reuse the open window (~80 ms each). The session closes automatically when the bot fires any non-chest action.

| # | Name | Category |
|---|------|----------|
| 111-112 | deposit_cobblestone / withdraw_cobblestone | Bulk material |
| 113-114 | deposit_gravel / withdraw_gravel | Bulk material |
| 115-116 | deposit_dirt / withdraw_dirt | Bulk material |
| 117-118 | deposit_food_raw / withdraw_food_raw | First matching raw food (beef, porkchop, chicken, mutton, rabbit, cod, salmon, potato, kelp) |
| 119-120 | deposit_food_cooked / withdraw_food_cooked | First matching cooked food (cooked meats + bread, baked_potato, dried_kelp, apple) |
| 121-160 | deposit/withdraw_\<material\>_\<type\> | Tool matrix: material in {wooden, stone, iron, diamond} x type in {pickaxe, sword, axe, shovel, hoe}. 40 actions total. Example: 121-122 = deposit/withdraw_wooden_pickaxe; 159-160 = deposit/withdraw_diamond_hoe. |

### Mine/dig vertical-scope expansion (161-167)

`mine_forward` (5) and `dig_forward` (69) scan foot-level (y=0) only. These seven actions cover the remaining vertical positions, each with pickaxe (mine) and shovel (dig) variants.

| # | Name | Group | Position |
|---|------|-------|----------|
| 161 | dig_below | medium | Shovel-family block directly beneath the bot (y-1). |
| 162 | mine_above | slow | Pickaxe-family block directly above the bot's head (ceiling). |
| 163 | dig_above | medium | Shovel-family ceiling block. |
| 164 | mine_overhang | slow | Pickaxe-family block at head-level in front. Opens a full-height tunnel when combined with `mine_forward`. |
| 165 | dig_overhang | medium | Shovel-family head-level-forward block. |
| 166 | mine_diagonal | slow | Pickaxe-family block one forward and one down. Staircase-descent primitive. |
| 167 | dig_diagonal | medium | Shovel-family forward-and-down block. |

### Armor crafting (168-183)

All armor crafts require a crafting table nearby. Material counts follow standard Minecraft recipes (helmet=5, chestplate=8, leggings=7, boots=4).

| # | Name | Material |
|---|------|----------|
| 168-171 | craft_leather_helmet / chestplate / leggings / boots | Leather |
| 172-175 | craft_iron_helmet / chestplate / leggings / boots | Iron ingot |
| 176-179 | craft_golden_helmet / chestplate / leggings / boots | Gold ingot |
| 180-183 | craft_diamond_helmet / chestplate / leggings / boots | Diamond |

### Farming and building (184-185)

| # | Name | Group | Requirements / Notes |
|---|------|-------|----------------------|
| 184 | gather_seeds | medium | Breaks nearest short_grass / tall_grass / fern / large_fern within 4 blocks. Drops wheat_seeds. Completes the farming chain: craft_hoe -> till_soil -> gather_seeds -> plant_seed -> farm_nearest. |
| 185 | craft_stone_bricks | medium | 4 stone (smelted cobblestone) -> 4 stone_bricks. 2x2 recipe, no crafting table needed. Decorative building block. |

## Tool-tier requirements

The bot must own a pickaxe of at least the listed tier for mining actions to be available:

| Block | Minimum pickaxe tier |
|-------|---------------------|
| coal_ore, stone family | Wooden |
| iron_ore, copper_ore, lapis_ore | Stone |
| diamond_ore, emerald_ore, gold_ore, redstone_ore | Iron |
| obsidian, ancient_debris | Diamond |

A bot carrying multiple pickaxes automatically uses the highest-tier one.

## Hunting actions (51-55)

All hunting actions share the same pattern: find the target species within 10 blocks, equip the best weapon (sword > axe > bow+arrow), pursue the animal for up to 10 seconds. The action confirms success when the target entity disappears. If the animal outruns the bot, the action gives up and reports the actual elapsed time.
