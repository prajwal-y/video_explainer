"""Scene generator - creates Remotion scene components from scripts."""

import json
import re
from pathlib import Path
from typing import Any

from ..config import Config, LLMConfig, load_config
from ..understanding.llm_provider import ClaudeCodeLLMProvider, get_llm_provider
from .syntax_verifier import SyntaxVerifier, VerificationResult as SyntaxVerificationResult
from .validator import SceneValidator, ValidationResult


# System prompt for scene generation
SCENE_SYSTEM_PROMPT = """You are an expert React/Remotion developer creating animated scene components for technical explainer videos.

## Your Primary Goal

Create visuals that illuminate the specific concept being explained. Every animation should help viewers understand the concept—not just decorate the screen.

## What Good Technical Visuals Look Like

Here are examples from successful technical videos:

**Example 1 - Attention Matrix Visualization:**
"Massive attention matrix visualization with 197x197 glowing connection points. Show multiple attention heads working in parallel, with early layers highlighting local patch-to-patch connections and deeper layers showing broad, global attention patterns. The CLS token's attention map highlights semantically important image regions."

This works because it shows the MECHANISM of attention—not just "boxes connected by lines."

**Example 2 - Network Stack:**
"Central visualization: your message as a core, with layers wrapping around it. TCP layer (show ports, sequence numbers), IP layer (show addresses, TTL), Ethernet frame. Each layer is a different color with labeled fields. Show the complete packet structure with byte counts."

This shows the actual STRUCTURE being explained, with specific details.

**Example 3 - GPU Architecture:**
"NVIDIA H100 die diagram showing CUDA cores grid, HBM3 memory stacks on sides. Power consumption meter climbing. Eight GPUs in a server node with NVLink connections. Liquid cooling visualization—water flowing through cold plates."

This visualizes the actual hardware components, not generic shapes.

**Example 4 - Transistor Physics:**
"MOSFET cross-section: source, drain, gate, oxide layer, channel region. Animate switching—voltage applied, field lines forming, channel appearing, current flowing. CMOS inverter with NMOS and PMOS paired. Show complementary switching."

This shows HOW the mechanism works step by step.

## Core Principles

### Match Visuals to Narration

If the narration says "slice the image into 16×16 pixel patches", show:
- The image appearing
- A grid overlay dividing it into patches
- One patch being extracted and flattened into a vector
- The vector being projected through a linear layer

Don't just show "an image becomes tokens"—show the actual process.

### Use Specific Numbers from Narration

When narration mentions "196 patches" or "768 dimensions" or "16,896 CUDA cores", these numbers should appear in the visualization.

### Sync Timing to Speech

Visual elements should appear slightly BEFORE the narration mentions them (10-15 frames early), not after. The visual anticipates what's being explained.

### Show Mechanisms, Not Just Icons

For any process, show:
1. The input state
2. The transformation happening
3. The output state
4. How this connects to the next step

## Your Technical Expertise

You create visually stunning, educational animations using:
- **Remotion**: useCurrentFrame, useVideoConfig, interpolate, spring, Sequence, AbsoluteFill
- **React**: Functional components with TypeScript
- **CSS-in-JS**: Inline styles for all styling

## CRITICAL REQUIREMENTS

### 1. Reference/Citation Component (OPTIONAL - Recommended for Technical Scenes)

For scenes with technical citations, use the Reference component (import from "./components/Reference"):
```typescript
import {{ Reference }} from "./components/Reference";

// At the bottom of your scene JSX:
<Reference
  sources={{[
    "Source 1 description",
    "Source 2 description",
  ]}}
  startFrame={{startFrame}}
  delay={{90}}
/>
```

Note: Not all scenes need references. Hook scenes, transitions, and conclusion scenes may skip this.

### 2. DYNAMIC LAYOUT System (MANDATORY - PREVENTS OVERLAPS)

The layout system dynamically calculates all positions from base constraints. NEVER use hardcoded pixel values.

**Import the layout system:**
```typescript
import {{ LAYOUT, getCenteredPosition, getTwoColumnLayout, getThreeColumnLayout, getTwoRowLayout, getFlexibleGrid, getCenteredStyle }} from "./styles";
```

**Base constraints (defined in styles.ts):**
- Canvas: 1920x1080
- Left margin: 60px, Right margin: 60px (symmetric)
- Title area: 120px from top
- Bottom margin: 160px (for references)
- Full-width layout by default (no sidebar reservation)

**Usable content area (calculated automatically):**
```typescript
LAYOUT.content.startX   // 60px - left edge of content
LAYOUT.content.endX     // 1860px - right edge (full width minus right margin)
LAYOUT.content.width    // 1800px - full usable width
LAYOUT.content.startY   // 120px - top of content area
LAYOUT.content.endY     // 920px - bottom of content area
LAYOUT.content.height   // 800px - full usable height
```

Note: Some projects may have a sidebar reserved (SIDEBAR.width > 0), which reduces content width.

**Layout Helpers - Choose the right one for your scene:**

1. **4 QUADRANTS** (for scenes with 4 main elements):
```typescript
const {{ quadrants }} = LAYOUT;
// Use: quadrants.topLeft, topRight, bottomLeft, bottomRight
// Each has: {{ cx: centerX, cy: centerY }}

<div style={{{{
  position: "absolute",
  left: quadrants.topLeft.cx * scale,
  top: quadrants.topLeft.cy * scale,
  transform: "translate(-50%, -50%)",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
}}}}>
  {{/* Element centered in top-left quadrant */}}
</div>
```

2. **2 COLUMNS** (for left/right split layouts):
```typescript
const layout = getTwoColumnLayout();
// layout.left and layout.right have: {{ cx, cy, width, height }}

<div style={{{{
  position: "absolute",
  left: layout.left.cx * scale,
  top: layout.left.cy * scale,
  transform: "translate(-50%, -50%)",
}}}}>
  {{/* Left side content */}}
</div>
```

3. **3 COLUMNS** (for left/center/right layouts):
```typescript
const layout = getThreeColumnLayout();
// layout.left, layout.center, layout.right have: {{ cx, cy, width, height }}
```

4. **2 ROWS** (for top/bottom split layouts):
```typescript
const layout = getTwoRowLayout();
// layout.top and layout.bottom have: {{ cx, cy, width, height }}
```

5. **CENTERED** (for single main element):
```typescript
const center = getCenteredPosition();
// center has: {{ cx, cy, width, height }}
```

6. **FLEXIBLE GRID** (for any N×M grid):
```typescript
const cells = getFlexibleGrid(3, 2);  // 3 columns, 2 rows = 6 cells
// Each cell has: {{ cx, cy, width, height }}
```

**PROPORTIONAL Positioning (for complex layouts):**
Use percentages of the usable area instead of absolute pixels:
```typescript
const leftX = LAYOUT.content.startX + LAYOUT.content.width * 0.12;   // 12% from left
const rightX = LAYOUT.content.startX + LAYOUT.content.width * 0.88;  // 88% from left
const centerY = LAYOUT.content.startY + LAYOUT.content.height * 0.5; // centered vertically
```

**CRITICAL**:
- ALWAYS use LAYOUT constants or helper functions, NEVER hardcode pixel values
- Use transform: "translate(-50%, -50%)" to center elements at their position
- Each element should be self-contained with title + visualization + caption
- Size elements relative to their container: `width: LAYOUT.content.width * 0.8`

### 2.1 Layout Requirements (MANDATORY)

- **No overflow**: ALL elements must stay within 1920x1080 bounds at ALL frames
- **No overlapping**: Elements must NEVER overlap unless intentionally layered
  - Calculate exact positions for all elements before placing them
  - When showing new content, either: (a) position it in empty space, or (b) fade out/remove previous elements first
  - Stack elements vertically or horizontally with proper gaps (16-20px scaled, NOT 24+)
- **Fill the space**: Main content should use at least 60-70% of canvas - AVOID empty/wasted space
- **Consistent margins**: Use 60-80px scaled margins from edges
- **Component sizing**: Make elements LARGE and readable
  - Boxes, diagrams, images should be substantial (at least 200-400px scaled)
  - Don't make elements tiny with lots of whitespace around them
- **Container overflow prevention**: Content inside boxes must fit within the box bounds
  - Calculate content size before setting container size
  - Add padding inside containers (12-16px scaled, NOT 24+)
  - Use overflow: "hidden" if needed, but prefer proper sizing

### 2.2 CRITICAL: Content Positioning to Avoid Header Overlap

**ALWAYS add offset from the header/subtitle area:**
```typescript
// Main content container - ALWAYS use this pattern:
<div style={{{{
  position: "absolute",
  left: LAYOUT.content.startX * scale,
  top: (LAYOUT.content.startY + 30) * scale,  // +30 offset from header!
  width: LAYOUT.content.width * scale,
  height: (LAYOUT.content.height - 60) * scale,  // Reduce height to compensate
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gridTemplateRows: "1fr 0.85fr",  // Use uneven rows to prevent bottom overflow
  gap: 16 * scale,  // 16, not 24
}}}}>
```

### 2.3 CRITICAL: Preventing Bottom Overflow

When using CSS Grid layouts:
1. **Reduce content height**: Use `(LAYOUT.content.height - 60) * scale` instead of full height
2. **Use uneven grid rows**: `"1fr 0.85fr"` or `"1.3fr 0.7fr"` instead of `"1fr 1fr"`
3. **Keep gaps small**: Use `gap: 16 * scale` not `gap: 24 * scale`
4. **Reduce padding**: Use `padding: 12-16 * scale` not `padding: 24 * scale`
5. **Compact SVG viewBoxes**: Size SVG viewBox to fit content, not oversized

**Example of proper 2x2 grid that doesn't overflow:**
```typescript
<div style={{{{
  position: "absolute",
  left: LAYOUT.content.startX * scale,
  top: (LAYOUT.content.startY + 30) * scale,
  width: LAYOUT.content.width * scale,
  height: (LAYOUT.content.height - 60) * scale,  // Account for offset
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gridTemplateRows: "1fr 0.85fr",  // Bottom row slightly smaller
  gap: 16 * scale,
}}}}>
  {{/* Top-left panel */}}
  <div style={{{{ padding: 16 * scale, borderRadius: 12 * scale }}}}>
    <svg viewBox="0 0 400 250" preserveAspectRatio="xMidYMid meet">
      {{/* Compact SVG content */}}
    </svg>
  </div>
  {{/* ... other panels */}}
</div>
```

### 2.4 Scene Indicator (OPTIONAL)

Scene indicators showing scene numbers are optional and often not needed:
- Skip scene indicators for cleaner visuals
- If used, keep them small and unobtrusive

### 3. Animation Requirements (MANDATORY)

- **No chaotic motion**: No shaking, trembling, or erratic movements
- **Smooth springs**: Use damping: 12-20, stiffness: 80-120 for natural movement
- **Proportional timing**: Phase durations scale with durationInFrames
- **Stagger delays**: 10-20 frames between sequential elements
- **Bounded motion**: Ensure animated elements stay within canvas
- **Complete animations**: If animating a sequence (e.g., flattening pixels), ensure it completes for ALL items
- **Narration sync**: Visual phases MUST align with voiceover timing
  - When narration mentions something, it should be visible on screen at that moment
  - Don't show visuals too early or too late relative to narration
  - Calculate phase timings based on when concepts are mentioned in the voiceover

### 3.1 Arrows and Connections (MANDATORY)

- **Complete paths**: Arrows must connect from source to destination without breaks
- **Proper endpoints**: Arrow heads should touch their target elements
- **Visibility**: Arrows should be visible (2-3px stroke, contrasting color)
- **Animation**: Animate arrows drawing from source to destination using strokeDasharray/strokeDashoffset

### 3.2 Dynamic Background System (MANDATORY - NO STATIC SCENES)

Every scene MUST have continuous visual interest. NEVER have a static background.

**Required Background Elements:**
```typescript
// 1. Animated gradient background (subtle hue shifts)
const bgHue1 = interpolate(localFrame, [0, durationInFrames], [140, 180]);
const bgHue2 = interpolate(localFrame, [0, durationInFrames], [200, 240]);

<AbsoluteFill style={{{{
  background: `linear-gradient(135deg,
    hsl(${{bgHue1}}, 12%, 97%) 0%,
    hsl(${{bgHue2}}, 15%, 95%) 50%,
    hsl(${{bgHue1}}, 10%, 98%) 100%)`,
}}}}>

// 2. SVG Grid pattern with floating particles
<svg style={{{{ position: "absolute", width: "100%", height: "100%" }}}}>
  <defs>
    <pattern id="grid" width={{40 * scale}} height={{40 * scale}} patternUnits="userSpaceOnUse">
      <path d={{`M ${{40 * scale}} 0 L 0 0 0 ${{40 * scale}}`}} fill="none" stroke={{COLORS.border}} strokeWidth={{0.5}} opacity={{0.3}} />
    </pattern>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
    </filter>
  </defs>
  <rect width="100%" height="100%" fill="url(#grid)" />
</svg>

// 3. Floating background particles (continuous motion)
const bgParticles = Array.from({{ length: 20-30 }}).map((_, i) => {{
  const seed = i * 137.5;
  const baseX = (seed * 7.3) % 100;
  const baseY = (seed * 11.7) % 100;
  const speed = 0.3 + (i % 5) * 0.15;
  return {{ baseX, baseY, speed, phase: i * 0.5 }};
}});

// Render particles with continuous animation
{{bgParticles.map((p, i) => {{
  const x = (p.baseX + (localFrame * p.speed * 0.5) % 100);
  const y = p.baseY + Math.sin((localFrame + p.phase) * 0.03) * 5;
  const opacity = 0.2 + Math.sin((localFrame + p.phase) * 0.05) * 0.15;
  return <circle key={{i}} cx={{`${{x}}%`}} cy={{`${{y}}%`}} r={{3 * scale}} fill={{COLORS.primary}} opacity={{opacity}} />;
}})}}

// 4. Glow pulse for emphasis (use throughout)
const glowPulse = 0.7 + 0.3 * Math.sin(localFrame * 0.1);
// Apply: opacity={{0.3 * glowPulse}}, boxShadow={{`0 0 ${{15 * glowPulse}}px ${{color}}60`}}

// 5. Pulse rings emanating from center (optional but recommended)
const pulseRings = Array.from({{ length: 4 }}).map((_, i) => ({{ delay: i * 45, duration: 180 }}));
{{pulseRings.map((ring, i) => {{
  const ringFrame = (localFrame + ring.delay) % ring.duration;
  const ringProgress = ringFrame / ring.duration;
  const ringRadius = ringProgress * 600;
  const ringOpacity = interpolate(ringProgress, [0, 0.2, 0.8, 1], [0, 0.15, 0.05, 0]);
  return <circle key={{i}} cx="50%" cy="50%" r={{ringRadius * scale}} fill="none" stroke={{COLORS.primary}} strokeWidth={{1 * scale}} opacity={{ringOpacity}} />;
}})}}
```

### 3.3 Continuous Data Flow Visualization (RECOMMENDED)

Show data flowing between components for visual interest:
```typescript
// Particles flowing between elements
{{Array.from({{ length: 8 }}).map((_, i) => {{
  const progress = ((localFrame * 0.015 + i * 0.125) % 1);
  const x = startX + (endX - startX) * progress;
  const y = startY + (endY - startY) * progress + Math.sin(progress * Math.PI * 3) * 15;
  const opacity = interpolate(progress, [0, 0.1, 0.9, 1], [0, 0.8, 0.8, 0]);
  return (
    <g key={{i}}>
      <circle cx={{x * scale}} cy={{y * scale}} r={{4 * scale}} fill={{COLORS.primary}} opacity={{opacity * 0.5}} filter="url(#glow)" />
      <circle cx={{x * scale}} cy={{y * scale}} r={{2 * scale}} fill={{COLORS.primary}} opacity={{opacity}} />
    </g>
  );
}})}}
```

### 3.4 Activity Indicators (RECOMMENDED)

Show continuous activity at the bottom of scenes:
```typescript
<div style={{{{ position: "absolute", left: 80 * scale, bottom: 60 * scale, display: "flex", gap: 24 * scale }}}}>
  <div style={{{{ display: "flex", alignItems: "center", gap: 8 * scale }}}}>
    <div style={{{{
      width: 8 * scale, height: 8 * scale, borderRadius: "50%",
      backgroundColor: COLORS.primary,
      boxShadow: `0 0 ${{8 + Math.sin(localFrame * 0.15) * 4}}px ${{COLORS.primary}}`,
      opacity: 0.7 + Math.sin(localFrame * 0.15) * 0.3,
    }}}} />
    <span style={{{{ fontSize: 10 * scale, color: COLORS.textMuted, fontFamily: FONTS.mono }}}}>ACTIVE</span>
  </div>
</div>
```

### 4. Typography Requirements (MANDATORY)

- **Font weight**: Always use fontWeight: 400 for handwritten fonts (FONTS.handwritten)
- **Line height**: Use 1.5 for body text
- **Font sizes**:
  - Titles: 42-48px scaled
  - Subtitles: 20-26px scaled
  - Body: 18-22px scaled
  - Labels: 14-18px scaled
  - Citations: 14-16px scaled

### 5. Visual Content Requirements (MANDATORY)

- **Use real visuals, not placeholders**:
  - Instead of "[CAT]" or "cat picture" text, use actual image elements or colored rectangles representing images
  - Use emoji or unicode symbols where appropriate (🐱, 🚗, 🏥) instead of text labels
  - Create visual representations (colored boxes, icons, shapes) rather than text descriptions
- **No brand names**: Don't use specific company names (Tesla, Google, etc.) - use generic descriptions
- **Representational images**: When showing "an image of X", render a stylized visual representation, not just text

## Animation Principles

1. **Frame-based timing**: Everything is based on `useCurrentFrame()`. Calculate local frames relative to scene start.
2. **Smooth interpolation**: Use `interpolate()` for all transitions with proper extrapolation clamping.
3. **Spring animations**: Use `spring()` for natural movements (NOT bouncy or chaotic).
4. **Staggered reveals**: Animate elements sequentially with calculated delays.
5. **Scale-responsive**: Always use a scale factor based on `width/1920` for responsive sizing.

## Code Patterns

```typescript
// Standard scene structure with dynamic layout
import React from "react";
import {{ AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, spring }} from "remotion";
import {{
  COLORS, FONTS, LAYOUT,
  getSceneIndicatorStyle, getSceneIndicatorTextStyle,
  getTwoColumnLayout, getCenteredPosition, getCenteredStyle
}} from "./styles";

interface SceneNameProps {{
  startFrame?: number;
}}

export const SceneName: React.FC<SceneNameProps> = ({{ startFrame = 0 }}) => {{
  const frame = useCurrentFrame();
  const {{ fps, width, height, durationInFrames }} = useVideoConfig();
  const localFrame = frame - startFrame;
  const scale = Math.min(width / 1920, height / 1080);

  // Phase timings as percentages of total duration
  const phase1End = Math.round(durationInFrames * 0.25);
  const phase2End = Math.round(durationInFrames * 0.50);
  // ...

  // Get layout positions (choose one based on your scene)
  const {{ quadrants }} = LAYOUT;  // For 4-element scenes
  // OR: const layout = getTwoColumnLayout();  // For left/right split
  // OR: const center = getCenteredPosition(); // For single centered element

  // Animations using interpolate
  const titleOpacity = interpolate(localFrame, [0, 15], [0, 1], {{
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  }});

  return (
    <AbsoluteFill style={{{{ backgroundColor: COLORS.background, fontFamily: FONTS.primary }}}}>
      {{/* Scene indicator */}}
      <div style={{{{ ...getSceneIndicatorStyle(scale), opacity: titleOpacity }}}}>
        <span style={{getSceneIndicatorTextStyle(scale)}}>1</span>
      </div>

      {{/* Main content - positioned using LAYOUT quadrants */}}
      <div style={{{{
        position: "absolute",
        left: quadrants.topLeft.cx * scale,
        top: quadrants.topLeft.cy * scale,
        transform: "translate(-50%, -50%)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}}}>
        {{/* Top-left quadrant content */}}
      </div>

      <div style={{{{
        position: "absolute",
        left: quadrants.topRight.cx * scale,
        top: quadrants.topRight.cy * scale,
        transform: "translate(-50%, -50%)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}}}>
        {{/* Top-right quadrant content */}}
      </div>

      {{/* Citation - bottom right */}}
      <div style={{{{
        position: "absolute",
        bottom: LAYOUT.margin.bottom * scale,
        right: LAYOUT.margin.right * scale,
        fontSize: 14 * scale,
        color: COLORS.textMuted,
        fontStyle: "italic",
      }}}}>
        "Paper Title" — Authors et al., Year
      </div>
    </AbsoluteFill>
  );
}};
```

## Reusable Components (USE THESE)

Import and use these pre-built components for consistency:

### 1. Reference Component (Citations/Sources)
```typescript
import {{ Reference }} from "./components/Reference";

// Usage (bottom-right, auto-positioned)
<Reference
  sources={{[
    "Source description 1",
    "Author et al. paper reference",
    "Technical specification name",
  ]}}
  startFrame={{startFrame}}
  delay={{90}}  // frames before appearing
/>
```
- Automatically positioned in bottom-right
- Fades in after specified delay
- Consistent styling across all scenes

## Visual Elements to Use

- **Text reveals**: Fade in with slight upward movement
- **Diagrams**: Build up progressively, highlighting active parts
- **Token grids**: Show data as colored blocks that animate
- **Progress bars**: Show comparisons and changes over time
- **Arrows/connections**: Animate to show data flow
- **Subtle highlights**: Use box-shadow sparingly for emphasis
- **SVG for complex shapes**: Use SVG for circuit diagrams, waveforms, neural networks
- **Emoji icons**: Use for quick visual recognition (💻, 🔀, 🌐, 📡, 🏢)

## Advanced Visual Patterns (HIGHLY RECOMMENDED)

### 1. Phase-Based Narration Sync (CRITICAL)
Analyze the voiceover to identify key moments, then create phase timings:
```typescript
// Phase timings based on narration flow (~30 second scene example)
const phase1 = Math.round(durationInFrames * 0.08);  // First concept mentioned
const phase2 = Math.round(durationInFrames * 0.25);  // Second concept
const phase3 = Math.round(durationInFrames * 0.42);  // Main point
const phase4 = Math.round(durationInFrames * 0.60);  // Key insight
const phase5 = Math.round(durationInFrames * 0.80);  // Conclusion builds
const phase6 = Math.round(durationInFrames * 0.92);  // Final message
```

### 2. Dynamic Pulsing Effects
Create living, breathing animations:
```typescript
const pulse = Math.sin(localFrame * 0.1) * 0.15 + 0.85;
const cellPulse = (index: number) => Math.sin(localFrame * 0.12 + index * 0.4) * 0.3 + 0.7;
// Use: opacity: pulse, boxShadow: pulse > 0.8 ? `0 0 ${15 * scale}px ${color}60` : "none"
```

### 3. Flowing Particle Animations
Show data flow with moving particles:
```typescript
const flowOffset = (localFrame * 2) % 200;
const renderFlowingParticles = (count: number, startX: number, endX: number) => {{
  return Array.from({{ length: count }}, (_, i) => {{
    const progress = ((flowOffset / 200) + (i / count)) % 1;
    const x = startX + (endX - startX) * progress;
    const opacity = Math.sin(progress * Math.PI);
    return <div key={{i}} style={{{{ left: x * scale, opacity }}}} />;
  }});
}};
```

### 4. SVG-Based Visualizations
Use SVG for complex shapes like brain diagrams, waves, connections:
```typescript
<svg width={{400 * scale}} height={{300 * scale}} viewBox="0 0 400 300">
  <path
    d={{`M 50 150 Q ${{100 + Math.sin(localFrame * 0.1) * 20}} 100 200 150`}}
    stroke={{COLORS.primary}}
    strokeWidth={{3}}
    fill="none"
  />
</svg>
```

### 5. Comparison Layouts (Problem vs Solution)
Side-by-side layouts with animated transitions:
```typescript
// LEFT: Old/Problem | CENTER: Arrow/Transform | RIGHT: New/Solution
<div style={{{{ display: "flex", gap: 80 * scale }}}}>
  <div style={{{{ opacity: oldOpacity, filter: oldOpacity < 0.5 ? "grayscale(100%)" : "none" }}}}>
    {{/* Old state */}}
  </div>
  <svg>{{/* Animated arrow */}}</svg>
  <div style={{{{ opacity: newOpacity, transform: `scale(${{newScale}})` }}}}>
    {{/* New state with glow effect */}}
  </div>
</div>
```

### 6. Animated Wave Frequencies
Show multi-frequency concepts (brain waves, signals):
```typescript
const wavePoints = Array.from({{ length: 50 }}, (_, i) => {{
  const x = i * 8;
  const y = 50 + Math.sin(i * 0.3 + localFrame * 0.15) * amplitude;
  return `${{i === 0 ? "M" : "L"}} ${{x}} ${{y}}`;
}}).join(" ");
```

### 7. Scene Layout Structure (RECOMMENDED)
Consistent structure for all scenes - NOTE: Scene indicators are OPTIONAL and often not needed:
```typescript
return (
  <AbsoluteFill style={{{{ backgroundColor: COLORS.background, fontFamily: FONTS.primary }}}}>
    {{/* Title - left aligned at top */}}
    <div style={{{{ position: "absolute", top: LAYOUT.title.y * scale, left: LAYOUT.title.x * scale }}}}>
      <div style={{{{ fontSize: 52 * scale, fontWeight: 600, color: COLORS.text }}}}>{{title}}</div>
      <div style={{{{ fontSize: 22 * scale, color: COLORS.textMuted, marginTop: 8 * scale }}}}>{{subtitle}}</div>
    </div>

    {{/* Main content - USE GRID LAYOUT with proper offset */}}
    <div style={{{{
      position: "absolute",
      left: LAYOUT.content.startX * scale,
      top: (LAYOUT.content.startY + 30) * scale,  // CRITICAL: +30 offset from header!
      width: LAYOUT.content.width * scale,
      height: (LAYOUT.content.height - 60) * scale,  // CRITICAL: Reduce height to prevent overflow
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gridTemplateRows: "1fr 0.85fr",  // CRITICAL: Uneven rows prevent bottom overflow
      gap: 16 * scale,  // CRITICAL: 16, not 24
    }}}}>
      {{/* Grid panels here */}}
    </div>

    {{/* Reference component for citations */}}
    <Reference sources={{sources}} startFrame={{startFrame}} delay={{90}} />
  </AbsoluteFill>
);
```

### 8. COMMON PITFALLS TO AVOID (CRITICAL)

Based on real production issues, NEVER make these mistakes:

**1. Content Overlapping Header/Subtitle:**
```typescript
// BAD - content starts at LAYOUT.content.startY directly
top: LAYOUT.content.startY * scale,

// GOOD - always add 30px offset
top: (LAYOUT.content.startY + 30) * scale,
```

**2. Bottom Overflow with Equal Grid Rows:**
```typescript
// BAD - equal rows often cause bottom overflow
gridTemplateRows: "1fr 1fr",
height: LAYOUT.content.height * scale,

// GOOD - uneven rows and reduced height
gridTemplateRows: "1fr 0.85fr",  // or "1.3fr 0.7fr"
height: (LAYOUT.content.height - 60) * scale,
```

**3. Excessive Padding and Gaps:**
```typescript
// BAD - too much padding causes overflow
padding: 24 * scale,
gap: 24 * scale,
borderRadius: 16 * scale,

// GOOD - compact values
padding: 12 * scale,  // or 16 max
gap: 16 * scale,
borderRadius: 12 * scale,
```

**4. Oversized SVG ViewBoxes:**
```typescript
// BAD - viewBox too large for container
<svg viewBox="0 0 400 300">

// GOOD - size viewBox to actual content needs
<svg viewBox="0 0 400 250">  // Reduced height
```

**5. Multiple Sections Vertically Without Height Constraints:**
```typescript
// BAD - flex column with no height management
display: "flex",
flexDirection: "column",
gap: 20 * scale,

// GOOD - use minHeight: 0 on flex children and careful gap sizing
display: "flex",
flexDirection: "column",
gap: 14 * scale,
// Children should have: flex: 1, minHeight: 0
```

**6. Font Sizes Too Large:**
```typescript
// BAD - oversized fonts
fontSize: 28 * scale,  // for body text

// GOOD - appropriate sizes
// Titles: 48-52px, Section headers: 16-18px, Body: 12-14px, Labels: 10-12px
fontSize: 14 * scale,
```

**7. Stats/Metrics Too Large:**
```typescript
// BAD
fontSize: 36 * scale,
gap: 50 * scale,

// GOOD
fontSize: 28 * scale,  // or smaller
gap: 40 * scale,
```

**8. Not Using minHeight: 0 in Flex Containers:**
```typescript
// BAD - flex child can overflow
<div style={{{{ flex: 1 }}}}>

// GOOD - prevents overflow
<div style={{{{ flex: 1, minHeight: 0 }}}}>
```

## Scene Type Archetypes

### Problem/Challenge Scenes
- Show broken/failing state with red highlights
- Use dissolution/fading effects for "forgetting" concepts
- Comparison grids showing before/after degradation

### Solution/Introduction Scenes
- Build up progressively from simple to complex
- Use green/success colors for revelations
- Spring animations for "aha moment" appearances

### Technical Deep-Dive Scenes
- Side-by-side comparison views
- Animated arrows showing data/concept flow
- Memory cell grids with pulsing effects

### Results/Performance Scenes
- Animated bar charts that grow
- Large numerical callouts with emphasis
- Before/after comparisons with metrics

### Conclusion/Vision Scenes
- Timeline visualizations
- Glowing final message with box-shadow
- Old vs New comparison fading

## Color Scheme (import from ./styles)

- primary: "#00d9ff" (cyan - main headings, key elements, emphasis)
- secondary: "#ff6b35" (orange - supporting elements, contrasts)
- success: "#00ff88" (green - positive outcomes, solutions, checkmarks)
- error: "#ff4757" (red - problems, warnings, alerts)
- textDim: "#888888" (secondary text, less important info)
- textMuted: "#666666" (tertiary text, citations, captions)
"""


SCENE_GENERATION_PROMPT = """Generate a Remotion scene component for the following scene.

## Scene Information

**Scene Number**: {scene_number}
**Title**: {title}
**Type**: {scene_type}
**Duration**: {duration} seconds at 30fps = {total_frames} frames

**Voiceover/Narration**:
"{voiceover}"

**Visual Description**:
{visual_description}

**Key Elements to Animate**:
{elements}

{word_timestamps_section}

## STEP 0: Think About What Would Make This Concept Click (CRITICAL - DO THIS FIRST)

Before writing any code, think carefully:

1. **What concept is this scene explaining?**
   Read the narration. What specific idea or mechanism is being taught?

2. **What visualization would create genuine understanding?**
   Not "what looks cool" but "what would make a viewer actually get it"?

   - If explaining an algorithm: show each step, show the transformation
   - If explaining a formula: show what each term means, show the computation
   - If explaining a problem: show why it's hard, show what breaks
   - If comparing approaches: show the actual difference in behavior

3. **How can I show the MECHANISM, not just represent the idea?**
   - BAD: "Show a box labeled 'attention'"
   - GOOD: "Show Query and Key vectors, animate the dot product, show scores appearing"

4. **What from the visual description is concept-specific vs generic?**
   Keep the concept-specific parts. Replace generic parts with something that actually illustrates the concept.

5. **How does the visual sync with the narration?**
   When does each concept get mentioned? That's when its visual should appear.

Now proceed with the technical implementation:

## STEP 1: Sync Animations to Narration (CRITICAL)

Before writing code, analyze the voiceover and word timestamps above:
1. Identify key visual concepts mentioned in the narration
2. Find the EXACT frame when each concept is spoken (from Word Timestamps section)
3. Set animation triggers to those frame numbers (or 10-15 frames earlier for anticipation)

**Example of CORRECT timing approach**:
```typescript
// GOOD: Using exact frame from word timestamps
// "silicon" spoken at frame 394 (13.14s)
const siliconAppears = 380;  // Start 14 frames early for anticipation

// BAD: Using percentage-based timing (NEVER DO THIS)
const siliconAppears = Math.floor(durationInFrames * 0.15);  // DON'T DO THIS
```

The emotional arc should follow the narration's natural timing, NOT arbitrary percentages.

## STEP 2: Choose Visual Patterns Based on Scene Type

For "{scene_type}" scenes, use these patterns:

**If problem/challenge**:
- Red/error colors for broken states
- Dissolution/fading effects
- Comparison showing degradation

**If solution/introduction**:
- Green/success colors for revelations
- Build-up animations
- Spring effects for "aha moments"

**If technical/deep-dive**:
- Side-by-side comparisons
- Animated data flow arrows
- Pulsing memory/node visualizations

**If results/performance**:
- Animated bar charts
- Large numerical callouts
- Before/after metrics

**If conclusion/vision**:
- Timeline with milestones
- Glowing final message
- Transition from old to new

## Reference: Example Scene Structure

```typescript
{example_scene}
```

## MANDATORY Requirements

1. Create a complete, working React/Remotion component
2. Name the component `{component_name}`
3. Export it as a named export
4. Include proper TypeScript interface for props
5. Use frame-based animations that match the narration timing
6. Scene indicators are OPTIONAL - skip them for cleaner visuals
7. Make all sizes responsive using the scale factor
8. Import styles from "./styles" (COLORS, FONTS, LAYOUT)
9. Phase timings should be based on word timestamps (NOT percentages of durationInFrames)
10. Add a detailed comment block at the top explaining the visual flow and the narration text

## CRITICAL Layout & Style Requirements (PREVENTING OVERFLOW)

11. **CONTENT OFFSET**: ALWAYS use `top: (LAYOUT.content.startY + 30) * scale` to avoid header overlap
12. **CONTENT HEIGHT**: ALWAYS use `height: (LAYOUT.content.height - 60) * scale` to prevent bottom overflow
13. **GRID ROWS**: Use UNEVEN row ratios like "1fr 0.85fr" or "1.3fr 0.7fr", NOT "1fr 1fr"
14. **COMPACT GAPS**: Use `gap: 16 * scale`, NOT 24 or higher
15. **COMPACT PADDING**: Use `padding: 12-16 * scale`, NOT 24 or higher
16. **COMPACT BORDER RADIUS**: Use `borderRadius: 12 * scale`, NOT 16 or higher
17. **SVG VIEWBOX**: Size SVG viewBox to fit content compactly (e.g., "0 0 400 250" not "0 0 400 300")
18. **FLEX CHILDREN**: Always add `minHeight: 0` to flex children to prevent overflow
19. **NO OVERFLOW**: All elements MUST stay within 1920x1080 bounds at ALL animation keyframes
20. **NO CHAOTIC MOTION**: No shaking, trembling, or erratic animations

## Typography & Sizing Requirements

21. **TITLE**: 48-52px scaled, fontWeight: 600
22. **SUBTITLE**: 20-22px scaled, color: COLORS.textMuted
23. **SECTION HEADERS**: 16-18px scaled, fontWeight: 600
24. **BODY TEXT**: 12-14px scaled
25. **LABELS**: 10-12px scaled
26. **STATS/METRICS**: 28px scaled max (not 36+)
27. **Citations**: Use Reference component, not manual positioning

## CRITICAL Visual Quality Requirements

28. **DYNAMIC EFFECTS**: Use pulsing (Math.sin), flowing particles, or wave animations for living visuals
29. **SVG FOR COMPLEXITY**: Use SVG for brain diagrams, wave patterns, connection arrows
30. **CONSISTENT LAYOUT**: Title at top, subtitle below, main visualization in center
31. **LARGE VISUALIZATIONS**: Main visual elements should be substantial (200-400px scaled), not tiny with whitespace
32. **VISUAL METAPHORS**: Translate abstract concepts into concrete visuals (e.g., "memory" → pulsing grid cells)

## STEP 0: Layout Planning (DO THIS FIRST - BEFORE WRITING CODE)

Before writing any code, mentally plan your layout to prevent overflow:

1. **Identify your elements**: List all visual elements (title, subtitle, main visual, labels, etc.)
2. **Choose a layout type**:
   - `single_centered` - One main element, centered (best for simple concepts)
   - `title_with_visual` - Title at top, large visual below (most common)
   - `side_by_side` - Two columns for comparison (use 55%/45% split, not 50/50)
   - `grid_2x2` - Four quadrants (use uneven rows: 1.2fr/0.8fr)
   - `stacked_vertical` - Multiple items stacked (limit to 3-4 items max)

3. **Calculate available space**:
   - Total canvas: 1920 x 1080
   - Header reserved: top 120px (LAYOUT.header.height)
   - Content area: starts at y=150, height ~880px
   - Safe margins: 60px on each side

4. **Size your elements** (scaled values):
   - Title: 50px height
   - Subtitle: 30px height
   - Main visual: remaining height minus 100px buffer
   - For grid layouts: divide available height by rows, subtract gaps

5. **Verify before coding**:
   - Does total height of elements + gaps + padding fit in 880px?
   - Do widths fit in 1800px (1920 - 2*60 margins)?
   - Is there breathing room (at least 20px between elements)?

**Common overflow causes to AVOID**:
- Using equal row heights in grids (use 1.2fr/0.8fr instead of 1fr/1fr)
- Large gaps (24px+) - use 12-16px instead
- Thick borders/padding (24px+) - use 12-16px instead
- Multiple large SVGs without size limits
- Forgetting to account for animation expansion (elements that scale up)

## Output

Return ONLY the TypeScript/React code. No markdown code blocks, no explanation - just the code.
The component should be saved to: {output_path}
"""


SYNC_SCENE_PROMPT = """Update the timing in an existing scene to match new voiceover timestamps.

## CRITICAL: TIMING-ONLY UPDATE

You are syncing an existing scene to new voiceover timing. This is NOT a regeneration.

**DO NOT CHANGE**:
- Visual structure or layout
- Animation types or effects
- Color schemes or styling
- Component hierarchy
- Import statements
- Props interface

**ONLY UPDATE**:
- Frame numbers for animation triggers
- Phase timing constants (e.g., phase1End, phase2End)
- Animation delays and durations that are tied to narration
- Comments about timing (update to reflect new timestamps)

## Existing Scene Code

```typescript
{existing_code}
```

## New Word Timestamps

**Scene Duration**: {duration}s = {total_frames} frames at 30fps

{word_timestamps_section}

## Instructions

1. Read through the existing code and identify all timing-related values:
   - Phase end frames (e.g., `const phase1End = 90`)
   - Animation trigger frames (e.g., `const titleAppears = 30`)
   - Delays and durations tied to narration timing

2. For each timing value, find the corresponding word/phrase in the new timestamps

3. Update ONLY the numeric values to match the new timestamps:
   - If "solution" was at frame 150 and is now at frame 180, update that constant
   - Keep animation durations the same (e.g., if fade-in was 30 frames, keep it 30 frames)
   - Maintain the relative timing between elements within a phase

4. Update any timing-related comments to reflect the new timestamps

## Example

If the existing code has:
```typescript
// "memory" spoken at frame 120 (4.0s)
const memoryAppears = 105;  // 15 frames early for anticipation
```

And the new timestamps show "memory" at frame 150 (5.0s), update to:
```typescript
// "memory" spoken at frame 150 (5.0s)
const memoryAppears = 135;  // 15 frames early for anticipation
```

## Output

Return the COMPLETE updated component code with ONLY timing values changed.
The component should be saved to: {output_path}
"""


STYLES_TEMPLATE = '''/**
 * Shared Style Constants for {project_title}
 *
 * Light theme with glow effects and dynamic layout system.
 * Uses Outfit font - modern geometric sans-serif for tech content.
 */

import React from "react";

// Outfit font family (loaded via @remotion/google-fonts in Root.tsx)
const outfitFont = '"Outfit", -apple-system, BlinkMacSystemFont, sans-serif';

// ===== COLOR PALETTE - LIGHT THEME WITH GLOW =====
export const COLORS = {{
  // Background colors
  background: "#FAFAFA",
  surface: "#FFFFFF",
  surfaceAlt: "#F5F5F7",

  // Text colors
  text: "#1A1A1A",
  textDim: "#555555",
  textMuted: "#888888",

  // Accent colors (optimized for glow effects)
  primary: "#0066FF",
  primaryGlow: "#0088FF",
  secondary: "#FF6600",
  secondaryGlow: "#FF8800",
  success: "#00AA55",
  successGlow: "#00DD77",
  warning: "#F5A623",
  warningGlow: "#FFB840",
  error: "#E53935",
  errorGlow: "#FF5555",
  purple: "#8844FF",
  purpleGlow: "#AA66FF",
  cyan: "#00BCD4",
  cyanGlow: "#00E5FF",
  pink: "#E91E63",
  pinkGlow: "#FF4081",
  lime: "#76B900",
  limeGlow: "#9BE000",

  // Layer visualization
  layerActive: "#0066FF",
  layerCompleted: "#00AA55",
  layerPending: "#E0E0E5",

  // Borders and shadows
  border: "#E0E0E5",
  borderLight: "#EEEEEE",
  shadow: "rgba(0, 0, 0, 0.08)",

  // Glow-specific
  glowSubtle: "rgba(0, 102, 255, 0.15)",
  glowMedium: "rgba(0, 102, 255, 0.3)",
  glowStrong: "rgba(0, 102, 255, 0.5)",
}};

// ===== FONTS =====
export const FONTS = {{
  primary: outfitFont,
  heading: outfitFont,
  mono: "SF Mono, Monaco, Consolas, monospace",
  system: outfitFont,
}};

// ===== SCENE INDICATOR =====
export const SCENE_INDICATOR = {{
  container: {{
    top: 24,
    left: 24,
    width: 44,
    height: 44,
    borderRadius: 10,
  }},
  text: {{
    fontSize: 16,
    fontWeight: 600 as const,
  }},
}};

// ===== SIDEBAR AREA =====
// Reserved space on the right for optional project-specific sidebars
// Set to 0 for full-width layouts (default), or 260 for layouts with a sidebar
export const SIDEBAR = {{
  width: {sidebar_width},
  padding: 16,
  gap: 4,
  borderRadius: 8,
}};

// ===== LAYOUT GRID SYSTEM =====
// Designed for 1920x1080 canvas
// When sidebar_width is 0, content uses full canvas width
// All values are CALCULATED from base constraints - no hardcoded positions

// Base constraints (these are the only "magic numbers")
const CANVAS_WIDTH = 1920;
const CANVAS_HEIGHT = 1080;
const SIDEBAR_WIDTH = {sidebar_width};  // Width of right sidebar area (0 = full width)
const SIDEBAR_GAP = {sidebar_width} > 0 ? 30 : 0;  // Gap only when sidebar exists
const MARGIN_LEFT = 60;
const MARGIN_RIGHT = 60;  // Symmetric with left margin
const TITLE_HEIGHT = 120;     // Space for title at top
const BOTTOM_MARGIN = 160;    // Space for references at bottom

// Derived values
const USABLE_LEFT = MARGIN_LEFT;
const USABLE_RIGHT = CANVAS_WIDTH - MARGIN_RIGHT - SIDEBAR_WIDTH - SIDEBAR_GAP;
const USABLE_WIDTH = USABLE_RIGHT - USABLE_LEFT;
const USABLE_TOP = TITLE_HEIGHT;
const USABLE_BOTTOM = CANVAS_HEIGHT - BOTTOM_MARGIN;
const USABLE_HEIGHT = USABLE_BOTTOM - USABLE_TOP;

// Quadrant calculations
const QUADRANT_WIDTH = USABLE_WIDTH / 2;
const QUADRANT_HEIGHT = USABLE_HEIGHT / 2;
const LEFT_CENTER_X = USABLE_LEFT + QUADRANT_WIDTH / 2;
const RIGHT_CENTER_X = USABLE_LEFT + QUADRANT_WIDTH + QUADRANT_WIDTH / 2;
const TOP_CENTER_Y = USABLE_TOP + QUADRANT_HEIGHT / 2;
const BOTTOM_CENTER_Y = USABLE_TOP + QUADRANT_HEIGHT + QUADRANT_HEIGHT / 2;

export const LAYOUT = {{
  // Canvas dimensions
  canvas: {{
    width: CANVAS_WIDTH,
    height: CANVAS_HEIGHT,
  }},

  // Margins from edges
  margin: {{
    left: MARGIN_LEFT,
    right: MARGIN_RIGHT,
    top: 40,
    bottom: 60,
  }},

  // Sidebar area (reserved for optional project-specific sidebars)
  sidebar: {{
    width: SIDEBAR_WIDTH,
    gap: SIDEBAR_GAP,
  }},

  // Content area bounds
  content: {{
    startX: USABLE_LEFT,
    endX: USABLE_RIGHT,
    width: USABLE_WIDTH,
    startY: USABLE_TOP,
    endY: USABLE_BOTTOM,
    height: USABLE_HEIGHT,
  }},

  // QUADRANT SYSTEM - dynamically calculated from constraints
  // Elements are CENTERED within their quadrant using transform: translate(-50%, -50%)
  quadrants: {{
    // Usable bounds
    bounds: {{
      left: USABLE_LEFT,
      right: USABLE_RIGHT,
      top: USABLE_TOP,
      bottom: USABLE_BOTTOM,
      width: USABLE_WIDTH,
      height: USABLE_HEIGHT,
    }},
    // Quadrant centers (for centering elements)
    topLeft: {{ cx: LEFT_CENTER_X, cy: TOP_CENTER_Y }},
    topRight: {{ cx: RIGHT_CENTER_X, cy: TOP_CENTER_Y }},
    bottomLeft: {{ cx: LEFT_CENTER_X, cy: BOTTOM_CENTER_Y }},
    bottomRight: {{ cx: RIGHT_CENTER_X, cy: BOTTOM_CENTER_Y }},
    // Quadrant dimensions
    quadrantWidth: QUADRANT_WIDTH,
    quadrantHeight: QUADRANT_HEIGHT,
  }},

  // Title area
  title: {{
    x: 80,
    y: 40,
    subtitleY: 90,
  }},
}};

// ===== ANIMATION =====
export const ANIMATION = {{
  fadeIn: 20,
  stagger: 5,
  spring: {{ damping: 20, stiffness: 120, mass: 1 }},
}};

// ===== FLEXIBLE LAYOUT HELPERS =====
// These functions dynamically calculate positions for any grid configuration

/**
 * Get layout positions for a flexible grid (any number of columns/rows)
 * Returns center positions for each cell, meant to be used with transform: translate(-50%, -50%)
 */
export const getFlexibleGrid = (
  cols: number,
  rows: number
): {{ cx: number; cy: number; width: number; height: number }}[] => {{
  const cellWidth = USABLE_WIDTH / cols;
  const cellHeight = USABLE_HEIGHT / rows;
  const positions: {{ cx: number; cy: number; width: number; height: number }}[] = [];

  for (let row = 0; row < rows; row++) {{
    for (let col = 0; col < cols; col++) {{
      positions.push({{
        cx: USABLE_LEFT + cellWidth * col + cellWidth / 2,
        cy: USABLE_TOP + cellHeight * row + cellHeight / 2,
        width: cellWidth,
        height: cellHeight,
      }});
    }}
  }}
  return positions;
}};

/**
 * Get a single centered position (for scenes with one main element)
 */
export const getCenteredPosition = (): {{ cx: number; cy: number; width: number; height: number }} => ({{
  cx: USABLE_LEFT + USABLE_WIDTH / 2,
  cy: USABLE_TOP + USABLE_HEIGHT / 2,
  width: USABLE_WIDTH,
  height: USABLE_HEIGHT,
}});

/**
 * Get 2-column layout (left and right halves)
 */
export const getTwoColumnLayout = (): {{
  left: {{ cx: number; cy: number; width: number; height: number }};
  right: {{ cx: number; cy: number; width: number; height: number }};
}} => {{
  const colWidth = USABLE_WIDTH / 2;
  return {{
    left: {{
      cx: USABLE_LEFT + colWidth / 2,
      cy: USABLE_TOP + USABLE_HEIGHT / 2,
      width: colWidth,
      height: USABLE_HEIGHT,
    }},
    right: {{
      cx: USABLE_LEFT + colWidth + colWidth / 2,
      cy: USABLE_TOP + USABLE_HEIGHT / 2,
      width: colWidth,
      height: USABLE_HEIGHT,
    }},
  }};
}};

/**
 * Get 3-column layout
 */
export const getThreeColumnLayout = (): {{
  left: {{ cx: number; cy: number; width: number; height: number }};
  center: {{ cx: number; cy: number; width: number; height: number }};
  right: {{ cx: number; cy: number; width: number; height: number }};
}} => {{
  const colWidth = USABLE_WIDTH / 3;
  return {{
    left: {{
      cx: USABLE_LEFT + colWidth / 2,
      cy: USABLE_TOP + USABLE_HEIGHT / 2,
      width: colWidth,
      height: USABLE_HEIGHT,
    }},
    center: {{
      cx: USABLE_LEFT + colWidth + colWidth / 2,
      cy: USABLE_TOP + USABLE_HEIGHT / 2,
      width: colWidth,
      height: USABLE_HEIGHT,
    }},
    right: {{
      cx: USABLE_LEFT + colWidth * 2 + colWidth / 2,
      cy: USABLE_TOP + USABLE_HEIGHT / 2,
      width: colWidth,
      height: USABLE_HEIGHT,
    }},
  }};
}};

/**
 * Get 2-row layout (top and bottom halves)
 */
export const getTwoRowLayout = (): {{
  top: {{ cx: number; cy: number; width: number; height: number }};
  bottom: {{ cx: number; cy: number; width: number; height: number }};
}} => {{
  const rowHeight = USABLE_HEIGHT / 2;
  return {{
    top: {{
      cx: USABLE_LEFT + USABLE_WIDTH / 2,
      cy: USABLE_TOP + rowHeight / 2,
      width: USABLE_WIDTH,
      height: rowHeight,
    }},
    bottom: {{
      cx: USABLE_LEFT + USABLE_WIDTH / 2,
      cy: USABLE_TOP + rowHeight + rowHeight / 2,
      width: USABLE_WIDTH,
      height: rowHeight,
    }},
  }};
}};

/**
 * Get style for centering an element at a position
 * Use with: left: pos.cx * scale, top: pos.cy * scale, transform: "translate(-50%, -50%)"
 */
export const getCenteredStyle = (
  pos: {{ cx: number; cy: number }},
  scale: number
): React.CSSProperties => ({{
  position: 'absolute',
  left: pos.cx * scale,
  top: pos.cy * scale,
  transform: 'translate(-50%, -50%)',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
}});

/**
 * Convert a position to scaled pixel values
 */
export const scalePosition = (
  pos: {{ cx: number; cy: number; width: number; height: number }},
  scale: number
): {{ cx: number; cy: number; width: number; height: number }} => ({{
  cx: pos.cx * scale,
  cy: pos.cy * scale,
  width: pos.width * scale,
  height: pos.height * scale,
}});

// ===== HELPER FUNCTIONS =====
export const getScale = (width: number, height: number): number => {{
  return Math.min(width / 1920, height / 1080);
}};

export const getSceneIndicatorStyle = (scale: number): React.CSSProperties => ({{
  position: "absolute",
  top: SCENE_INDICATOR.container.top * scale,
  left: SCENE_INDICATOR.container.left * scale,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  width: SCENE_INDICATOR.container.width * scale,
  height: SCENE_INDICATOR.container.height * scale,
  borderRadius: SCENE_INDICATOR.container.borderRadius * scale,
  backgroundColor: `${{COLORS.primary}}20`,
  border: `2px solid ${{COLORS.primary}}`,
  boxShadow: `0 2px 12px ${{COLORS.primary}}30`,
}});

export const getSceneIndicatorTextStyle = (scale: number): React.CSSProperties => ({{
  fontSize: SCENE_INDICATOR.text.fontSize * scale,
  fontWeight: SCENE_INDICATOR.text.fontWeight,
  color: COLORS.primary,
  fontFamily: FONTS.mono,
}});

export default {{ COLORS, FONTS, ANIMATION, SIDEBAR }};
'''


INDEX_TEMPLATE = '''/**
 * {project_title} Scene Registry
 *
 * Exports all scene components for the video.
 * Keys match scene_id suffixes in storyboard.json (e.g., "scene1_hook" -> "hook")
 */

import React from "react";

{imports}

export type SceneComponent = React.FC<{{ startFrame?: number }}>;

/**
 * Scene registry mapping storyboard scene types to components.
 * Keys must match the scene_id suffix in storyboard.json
 */
const SCENE_REGISTRY: Record<string, SceneComponent> = {{
{registry_entries}
}};

// Standard export name for the build system (required by remotion/src/scenes/index.ts)
export const PROJECT_SCENES = SCENE_REGISTRY;

{exports}

export function getScene(type: string): SceneComponent | undefined {{
  return SCENE_REGISTRY[type];
}}

export function getAvailableSceneTypes(): string[] {{
  return Object.keys(SCENE_REGISTRY);
}}
'''


REFERENCE_TEMPLATE = '''/**
 * Reference Component
 *
 * Displays source references in the bottom-right corner of scenes.
 * Used to show citations, sources, and technical references.
 */

import React from "react";
import {{ interpolate, useCurrentFrame, useVideoConfig, spring }} from "remotion";
import {{ COLORS, FONTS }} from "../styles";

interface ReferenceProps {{
  sources: string[];
  startFrame?: number;
  delay?: number;
}}

export const Reference: React.FC<ReferenceProps> = ({{
  sources,
  startFrame = 0,
  delay = 60,
}}) => {{
  const frame = useCurrentFrame();
  const {{ fps, width, height }} = useVideoConfig();
  const localFrame = frame - startFrame;
  const scale = Math.min(width / 1920, height / 1080);

  // Fade in animation
  const opacity = spring({{
    frame: localFrame - delay,
    fps,
    config: {{ damping: 20, stiffness: 80 }},
  }});

  if (sources.length === 0) return null;

  return (
    <div
      style={{{{
        position: "absolute",
        right: 24 * scale,
        bottom: 24 * scale,
        maxWidth: 300 * scale,
        opacity,
        fontFamily: FONTS.system,
      }}}}
    >
      <div
        style={{{{
          backgroundColor: "rgba(255, 255, 255, 0.9)",
          borderRadius: 8 * scale,
          padding: `${{8 * scale}}px ${{12 * scale}}px`,
          border: `1px solid ${{COLORS.border}}`,
          boxShadow: `0 2px 8px rgba(0, 0, 0, 0.08)`,
        }}}}
      >
        <div
          style={{{{
            fontSize: 9 * scale,
            fontWeight: 600,
            color: COLORS.textMuted,
            textTransform: "uppercase",
            letterSpacing: 0.5 * scale,
            marginBottom: 4 * scale,
          }}}}
        >
          Sources
        </div>
        {{sources.map((source, index) => (
          <div
            key={{index}}
            style={{{{
              fontSize: 9 * scale,
              color: COLORS.textDim,
              lineHeight: 1.4,
              marginTop: index > 0 ? 2 * scale : 0,
            }}}}
          >
            {{source}}
          </div>
        ))}}
      </div>
    </div>
  );
}};

export default Reference;
'''


class SceneGenerator:
    """Generates Remotion scene components from scripts using Claude Code."""

    MAX_RETRIES = 3  # Maximum attempts to generate a valid scene

    def __init__(
        self,
        config: Config | None = None,
        working_dir: Path | None = None,
        timeout: int = 300,
        skip_validation: bool = False,
    ):
        """Initialize the scene generator.

        Args:
            config: Configuration object
            working_dir: Working directory for Claude Code
            timeout: Timeout for LLM calls in seconds
            skip_validation: Skip validation and auto-correction (for debugging)
        """
        self.config = config or load_config()
        self.working_dir = working_dir or Path.cwd()
        self.timeout = timeout
        self.skip_validation = skip_validation
        self.validator = SceneValidator()

    def generate_all_scenes(
        self,
        project_dir: Path,
        script_path: Path | None = None,
        example_scenes_dir: Path | None = None,
        voiceover_manifest_path: Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Generate all scene components for a project.

        Args:
            project_dir: Path to the project directory
            script_path: Path to script.json (defaults to project_dir/script/script.json)
            example_scenes_dir: Directory with example scenes for reference
            voiceover_manifest_path: Path to voiceover manifest.json with word timestamps
            force: Overwrite existing scenes

        Returns:
            Dict with generation results
        """
        # Resolve paths
        script_path = script_path or project_dir / "script" / "script.json"
        scenes_dir = project_dir / "scenes"

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Check if scenes already exist
        if scenes_dir.exists() and not force:
            existing = list(scenes_dir.glob("*.tsx"))
            if existing:
                raise FileExistsError(
                    f"Scenes already exist in {scenes_dir}. Use --force to overwrite."
                )

        # Load script
        with open(script_path) as f:
            script = json.load(f)

        # Load voiceover manifest for word timestamps (if available)
        word_timestamps_by_scene: dict[str, list[dict]] = {}
        if voiceover_manifest_path and voiceover_manifest_path.exists():
            with open(voiceover_manifest_path) as f:
                manifest = json.load(f)
            for scene_data in manifest.get("scenes", []):
                scene_id = scene_data.get("scene_id", "")
                word_timestamps_by_scene[scene_id] = scene_data.get("word_timestamps", [])

        # Create scenes directory
        scenes_dir.mkdir(parents=True, exist_ok=True)

        # Load example scene for reference
        example_scene = self._load_example_scene(example_scenes_dir)

        # Generate styles.ts first
        self._generate_styles(scenes_dir, script.get("title", "Untitled"))

        # Generate Reference component in components/ subdirectory
        self._generate_reference_component(scenes_dir)

        # Generate each scene
        results = {
            "scenes_dir": str(scenes_dir),
            "scenes": [],
            "errors": [],
        }

        for idx, scene in enumerate(script.get("scenes", [])):
            scene_num = idx + 1
            # Get word timestamps for this scene (if available)
            scene_id = scene.get("scene_id", f"scene{scene_num}")
            word_timestamps = word_timestamps_by_scene.get(scene_id, [])
            try:
                result = self._generate_scene(
                    scene=scene,
                    scene_number=scene_num,
                    scenes_dir=scenes_dir,
                    example_scene=example_scene,
                    word_timestamps=word_timestamps,
                )
                results["scenes"].append(result)
                print(f"  ✓ Generated scene {scene_num}: {result['component_name']}")

            except Exception as e:
                title = scene.get("title", f"Scene {scene_num}")
                error = {"scene_number": scene_num, "title": title, "error": str(e)}
                results["errors"].append(error)
                print(f"  ✗ Failed to generate scene {scene_num}: {e}")

        # Generate index.ts
        self._generate_index(scenes_dir, results["scenes"], script.get("title", "Untitled"))

        # Run final syntax verification on all generated scenes
        if not self.skip_validation:
            print("\nVerifying scene syntax...")
            syntax_result = self._verify_and_fix_syntax(scenes_dir)
            results["syntax_verification"] = {
                "success": syntax_result.success,
                "errors": [str(e) for e in syntax_result.errors],
                "fixed_files": syntax_result.fixed_files,
                "unfixed_files": syntax_result.unfixed_files,
            }
            if syntax_result.fixed_files:
                print(f"  ✓ Auto-fixed syntax in: {', '.join(syntax_result.fixed_files)}")
            if syntax_result.unfixed_files:
                print(f"  ⚠ Syntax errors remain in: {', '.join(syntax_result.unfixed_files)}")
                for error in syntax_result.errors[:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(syntax_result.errors) > 5:
                    print(f"    ... and {len(syntax_result.errors) - 5} more errors")
            if syntax_result.success:
                print("  ✓ All scenes pass syntax verification")

        return results

    def _verify_and_fix_syntax(self, scenes_dir: Path) -> SyntaxVerificationResult:
        """Verify and attempt to fix syntax errors in all scene files.

        Args:
            scenes_dir: Directory containing scene files

        Returns:
            SyntaxVerificationResult with errors and fix status
        """
        verifier = SyntaxVerifier(remotion_dir=self.working_dir / "remotion")
        return verifier.verify_scenes(scenes_dir, auto_fix=True)

    def sync_all_scenes(
        self,
        project_dir: Path,
        voiceover_manifest_path: Path | None = None,
        scene_filter: str | None = None,
    ) -> dict[str, Any]:
        """Sync timing in existing scenes to match updated voiceover timestamps.

        This is a lightweight operation that only updates timing values in scenes,
        preserving all visual structure and animations.

        Args:
            project_dir: Path to the project directory
            voiceover_manifest_path: Path to voiceover manifest.json with word timestamps
            scene_filter: Optional scene filename to sync (e.g., "HookScene.tsx")

        Returns:
            Dict with sync results
        """
        scenes_dir = project_dir / "scenes"
        script_path = project_dir / "script" / "script.json"

        if not scenes_dir.exists():
            raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")

        # Load voiceover manifest for word timestamps
        voiceover_manifest_path = voiceover_manifest_path or project_dir / "voiceover" / "manifest.json"
        if not voiceover_manifest_path.exists():
            raise FileNotFoundError(
                f"Voiceover manifest not found: {voiceover_manifest_path}\n"
                "Run 'voiceover' command first to generate voiceover with timestamps."
            )

        with open(voiceover_manifest_path) as f:
            manifest = json.load(f)

        # Build lookup of timestamps by scene_id
        timestamps_by_scene_id: dict[str, tuple[list[dict], float]] = {}
        for scene_data in manifest.get("scenes", []):
            scene_id = scene_data.get("scene_id", "")
            timestamps_by_scene_id[scene_id] = (
                scene_data.get("word_timestamps", []),
                scene_data.get("duration_seconds", 20.0),
            )

        # Load script to get scene info
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        with open(script_path) as f:
            script = json.load(f)

        # Get existing scene files
        scene_files = list(scenes_dir.glob("*.tsx"))
        scene_files = [f for f in scene_files if f.name not in ("index.tsx", "styles.tsx")]

        if scene_filter:
            scene_files = [f for f in scene_files if f.name == scene_filter]
            if not scene_files:
                raise FileNotFoundError(f"Scene not found: {scene_filter}")

        results = {
            "scenes_dir": str(scenes_dir),
            "synced": [],
            "skipped": [],
            "errors": [],
        }

        for scene_file in scene_files:
            # Find matching scene in script
            component_name = scene_file.stem
            scene_key = self._component_to_registry_key(component_name)

            # Try to find scene_id that matches this scene
            matching_scene_id = None
            matching_scene = None
            for idx, scene in enumerate(script.get("scenes", [])):
                scene_id = scene.get("scene_id", f"scene_{idx + 1}")
                # scene_id is a slug like "the_impossible_leap"
                # For backwards compat, strip "scene{N}_" prefix if present
                if re.match(r'^scene\d+_', str(scene_id)):
                    key_part = re.sub(r'^scene\d+_', '', str(scene_id))
                else:
                    key_part = str(scene_id)

                # Check if this scene matches
                title_key = self._title_to_scene_key(scene.get("title", ""))
                if key_part == scene_key or title_key == scene_key:
                    matching_scene_id = scene_id
                    matching_scene = scene
                    break

            if not matching_scene_id or matching_scene_id not in timestamps_by_scene_id:
                results["skipped"].append({
                    "filename": scene_file.name,
                    "reason": "No matching voiceover timestamps found",
                })
                continue

            word_timestamps, duration = timestamps_by_scene_id[matching_scene_id]
            if not word_timestamps:
                results["skipped"].append({
                    "filename": scene_file.name,
                    "reason": "No word timestamps in voiceover manifest",
                })
                continue

            try:
                self._sync_scene_timing(
                    scene_file=scene_file,
                    word_timestamps=word_timestamps,
                    duration=duration,
                    voiceover=matching_scene.get("voiceover", "") if matching_scene else "",
                )
                results["synced"].append({
                    "filename": scene_file.name,
                    "scene_id": matching_scene_id,
                })
                print(f"  ✓ Synced {scene_file.name}")

            except Exception as e:
                results["errors"].append({
                    "filename": scene_file.name,
                    "error": str(e),
                })
                print(f"  ✗ Failed to sync {scene_file.name}: {e}")

        return results

    def _sync_scene_timing(
        self,
        scene_file: Path,
        word_timestamps: list[dict],
        duration: float,
        voiceover: str = "",
    ) -> None:
        """Sync timing in a single scene file to match new voiceover timestamps.

        Args:
            scene_file: Path to the scene .tsx file
            word_timestamps: Word-level timestamps from voiceover
            duration: Scene duration in seconds
            voiceover: The voiceover text (for context)
        """
        # Read existing scene code
        with open(scene_file) as f:
            existing_code = f.read()

        # Format word timestamps for the prompt
        word_timestamps_section = self._format_word_timestamps_for_prompt(
            word_timestamps, voiceover, duration
        )

        # Build sync prompt
        prompt = SYNC_SCENE_PROMPT.format(
            existing_code=existing_code,
            duration=duration,
            total_frames=int(duration * 30),
            word_timestamps_section=word_timestamps_section,
            output_path=scene_file,
        )

        llm = get_llm_provider(self.config)

        if isinstance(llm, ClaudeCodeLLMProvider):
            full_prompt = f"""{prompt}

Write the complete updated component code to the file: {scene_file}
"""
            result = llm.generate_with_file_access(full_prompt, allow_writes=True)
            if not result.success:
                raise RuntimeError(f"LLM sync failed: {result.error_message}")
            if not scene_file.exists():
                code = self._extract_code(result.response)
                if code:
                    with open(scene_file, "w") as f:
                        f.write(code)
                else:
                    raise RuntimeError(f"Sync failed - no code generated for {scene_file}")
        else:
            full_prompt = f"""{prompt}

Write the complete updated component code as a single TypeScript/TSX code block.
"""
            response = llm.generate(full_prompt)
            code = self._extract_code(response)
            if not code:
                raise RuntimeError(f"Sync failed - no code generated for {scene_file}")
            with open(scene_file, "w") as f:
                f.write(code)

        # Validate the synced scene
        validation = self.validator.validate_single_scene(scene_file)
        if validation.errors:
            # Log errors but don't fail - timing sync shouldn't break validation
            print(f"    ⚠ Validation warnings after sync:")
            for error in validation.errors[:3]:
                print(f"      - {error.message}")

    def _generate_scene(
        self,
        scene: dict,
        scene_number: int,
        scenes_dir: Path,
        example_scene: str,
        word_timestamps: list[dict] | None = None,
    ) -> dict:
        """Generate a single scene component with validation and auto-correction.

        Generates the scene, validates it, and if validation fails, regenerates
        with feedback about the errors. Retries up to MAX_RETRIES times.

        Args:
            scene: Scene data from script
            scene_number: Scene number (1-indexed)
            scenes_dir: Output directory for scenes
            example_scene: Example scene code for reference
            word_timestamps: Word-level timestamps from voiceover for precise animation timing

        Returns:
            Dict with scene generation result

        Raises:
            RuntimeError: If scene generation fails after all retries
        """
        # Extract scene info
        title = scene.get("title", f"Scene {scene_number}")
        scene_type = scene.get("scene_type", "explanation")
        duration = scene.get("duration_seconds", 20)
        voiceover = scene.get("voiceover", "")

        # Derive scene key from title
        scene_key = self._title_to_scene_key(title)

        # Handle both old and new visual formats
        if "visual_cue" in scene:
            visual_desc = scene["visual_cue"].get("description", "")
            elements = scene["visual_cue"].get("elements", [])
        else:
            visual_desc = scene.get("visual_description", "")
            elements = scene.get("key_elements", [])

        # Generate component name from title
        component_name = self._title_to_component_name(title)
        filename = f"{component_name}.tsx"
        output_path = scenes_dir / filename

        # Format elements list
        elements_str = "\n".join(f"- {e}" for e in elements) if elements else "- General scene elements"

        # Format word timestamps for precise animation timing
        word_timestamps_section = self._format_word_timestamps_for_prompt(
            word_timestamps or [], voiceover, duration
        )

        # Build base prompt
        base_prompt = SCENE_GENERATION_PROMPT.format(
            scene_number=scene_number,
            title=title,
            scene_type=scene_type,
            duration=duration,
            total_frames=int(duration * 30),
            voiceover=voiceover,
            visual_description=visual_desc,
            elements=elements_str,
            component_name=component_name,
            example_scene=example_scene[:4000],
            output_path=output_path,
            word_timestamps_section=word_timestamps_section,
        )

        validation_feedback = ""
        last_error = None

        # If skip_validation, just generate once without validation
        if self.skip_validation:
            self._generate_scene_file(
                base_prompt=base_prompt,
                output_path=output_path,
                validation_feedback="",
            )
            print(f"    ⚠ Validation skipped - please check {filename} manually")
            return {
                "scene_number": scene_number,
                "title": title,
                "component_name": component_name,
                "filename": filename,
                "path": str(output_path),
                "scene_type": scene_type,
                "scene_key": scene_key,
            }

        for attempt in range(self.MAX_RETRIES):
            try:
                # Generate the scene
                self._generate_scene_file(
                    base_prompt=base_prompt,
                    output_path=output_path,
                    validation_feedback=validation_feedback,
                )

                # Validate the generated scene
                validation = self.validator.validate_single_scene(output_path)

                if not validation.errors:
                    # Success - scene is valid
                    if validation.warnings:
                        # Log warnings but don't fail
                        for warning in validation.warnings:
                            print(f"    ⚡ Warning: {warning.message}")
                    return {
                        "scene_number": scene_number,
                        "title": title,
                        "component_name": component_name,
                        "filename": filename,
                        "path": str(output_path),
                        "scene_type": scene_type,
                        "scene_key": scene_key,
                    }

                # Validation failed - build feedback for retry
                error_messages = []
                for error in validation.errors:
                    msg = f"- Line {error.line}: {error.message}"
                    if error.suggestion:
                        msg += f" (Fix: {error.suggestion})"
                    error_messages.append(msg)

                validation_feedback = f"""

## IMPORTANT: The previous attempt had validation errors. Fix these issues:

{chr(10).join(error_messages)}

Read the existing file at {output_path} and fix only the issues listed above.
Do not rewrite the entire file - just fix the specific errors.
"""
                last_error = f"Validation errors: {'; '.join(e.message for e in validation.errors)}"
                print(f"    ⚠ Attempt {attempt + 1}/{self.MAX_RETRIES}: {len(validation.errors)} error(s), retrying...")

            except Exception as e:
                last_error = str(e)
                print(f"    ⚠ Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")
                validation_feedback = f"""

## IMPORTANT: The previous attempt failed with an error: {e}

Please fix this issue and try again.
"""

        # All retries exhausted
        raise RuntimeError(f"Failed to generate valid scene after {self.MAX_RETRIES} attempts. Last error: {last_error}")

    def _generate_scene_file(
        self,
        base_prompt: str,
        output_path: Path,
        validation_feedback: str = "",
    ) -> None:
        """Generate a scene file using the configured LLM provider.

        Args:
            base_prompt: The base generation prompt
            output_path: Path to write the scene file
            validation_feedback: Optional feedback from previous validation failure
        """
        llm = get_llm_provider(self.config)

        full_prompt = f"""{SCENE_SYSTEM_PROMPT}

{base_prompt}
{validation_feedback}
Write the complete component code as a single TypeScript/TSX code block.
"""

        if isinstance(llm, ClaudeCodeLLMProvider):
            full_prompt_cc = f"""## FIRST: Load the Remotion skill
Before doing ANYTHING else, invoke the `/remotion` skill by using the Skill tool with skill="remotion".
This loads essential Remotion best practices for animations, timing, and visual patterns.
DO THIS NOW before proceeding.

---

{SCENE_SYSTEM_PROMPT}

{base_prompt}
{validation_feedback}
Write the complete component code to the file: {output_path}
"""
            result = llm.generate_with_file_access(full_prompt_cc, allow_writes=True)
            if not result.success:
                raise RuntimeError(f"LLM generation failed: {result.error_message}")
            if not output_path.exists():
                code = self._extract_code(result.response)
                if code:
                    with open(output_path, "w") as f:
                        f.write(code)
                else:
                    raise RuntimeError(f"Scene file not created: {output_path}")
        else:
            response = llm.generate(full_prompt, system_prompt=SCENE_SYSTEM_PROMPT)
            code = self._extract_code(response)
            if not code:
                raise RuntimeError("LLM returned no extractable code")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(code)

    def _generate_styles(
        self, scenes_dir: Path, project_title: str, sidebar_width: int = 0
    ) -> None:
        """Generate the styles.ts file.

        Args:
            scenes_dir: Directory to write styles.ts to
            project_title: Title for the comment header
            sidebar_width: Width of sidebar in pixels. 0 = full-width layout (default),
                          260 = reserved sidebar space for projects with persistent sidebars
        """
        styles_path = scenes_dir / "styles.ts"
        content = STYLES_TEMPLATE.format(
            project_title=project_title,
            sidebar_width=sidebar_width,
        )
        with open(styles_path, "w") as f:
            f.write(content)

    def _generate_index(
        self,
        scenes_dir: Path,
        scenes: list[dict],
        project_title: str,
    ) -> None:
        """Generate the index.ts file."""
        # Build imports
        imports = []
        exports = []
        registry_entries = []

        for scene in scenes:
            name = scene["component_name"]
            filename = scene["filename"].replace(".tsx", "")
            # Use scene_key as registry key - derived from title to match storyboard
            # (storyboard extracts keys from narration scene_ids, which are based on titles)
            scene_key = scene["scene_key"]

            imports.append(f'import {{ {name} }} from "./{filename}";')
            exports.append(f"export {{ {name} }} from \"./{filename}\";")
            registry_entries.append(f'  {scene_key}: {name},')

        content = INDEX_TEMPLATE.format(
            project_title=project_title,
            imports="\n".join(imports),
            exports="\n".join(exports),
            registry_entries="\n".join(registry_entries),
        )

        index_path = scenes_dir / "index.ts"
        with open(index_path, "w") as f:
            f.write(content)

    def _generate_reference_component(self, scenes_dir: Path) -> None:
        """Generate the Reference component in components/ subdirectory.

        This creates the Reference component that scenes can optionally import
        for displaying source citations.
        """
        components_dir = scenes_dir / "components"
        components_dir.mkdir(parents=True, exist_ok=True)

        reference_path = components_dir / "Reference.tsx"
        with open(reference_path, "w") as f:
            f.write(REFERENCE_TEMPLATE)

    def _load_example_scene(self, example_dir: Path | None = None) -> str:
        """Load an example scene for reference.

        Prioritizes well-designed scenes with rich animations and clear structure.
        If example_dir is provided, only that directory is searched.
        """
        projects_dir = Path(__file__).parent.parent.parent / "projects"

        # If specific directory provided, only use that (don't fall back)
        if example_dir:
            example_dirs = [example_dir]
        else:
            # Try multiple project directories in order of quality
            example_dirs = [
                projects_dir / "continual-learning" / "scenes",  # Best examples
                projects_dir / "llm-inference" / "scenes",
            ]

        # Priority list of example scenes (these have rich animations)
        priority_files = [
            "TheAmnesiaProblemScene.tsx",      # Good problem scene with timeline
            "IntroducingNestedLearningScene.tsx",  # Good solution scene with Russian dolls
            "PerformanceBreakthroughScene.tsx",  # Good results scene with bar charts
            "TheFutureOfLearningScene.tsx",    # Good conclusion scene
            "PhasesScene.tsx",
            "HookScene.tsx",
            "BottleneckScene.tsx",
        ]

        for dir_path in example_dirs:
            if not dir_path.exists():
                continue

            # Try priority files first
            for filename in priority_files:
                path = dir_path / filename
                if path.exists():
                    with open(path) as f:
                        return f.read()

            # Fall back to any .tsx file (not styles or index)
            tsx_files = [
                f for f in dir_path.glob("*.tsx")
                if f.name not in ("styles.ts", "index.ts", "index.tsx")
            ]
            if tsx_files:
                with open(tsx_files[0]) as f:
                    return f.read()

        return "// No example scene available"

    def _format_word_timestamps_for_prompt(
        self, word_timestamps: list[dict], voiceover: str, duration: float
    ) -> str:
        """Format word timestamps into a useful prompt section for animation timing.

        This extracts key phrases from the voiceover and maps them to their timestamps,
        helping the LLM sync animations precisely to the narration.

        Args:
            word_timestamps: List of {word, start_seconds, end_seconds} dicts
            voiceover: The voiceover text
            duration: Scene duration in seconds

        Returns:
            Formatted string for inclusion in the prompt
        """
        if not word_timestamps:
            return """
## Word Timestamps (NOT AVAILABLE)
No voiceover timestamps available. Use percentage-based timing as a fallback:
- phase1: ~10% into scene
- phase2: ~25% into scene
- phase3: ~40% into scene
- phase4: ~60% into scene
- phase5: ~80% into scene
- phase6: ~95% into scene
"""

        # Build a timeline string showing words at their timestamps
        timeline_entries = []
        for i, wt in enumerate(word_timestamps):
            word = wt.get("word", "")
            start = wt.get("start_seconds", 0)
            frame = int(start * 30)  # Convert to frame number
            timeline_entries.append(f'  - "{word}" at {start:.2f}s (frame {frame})')

        # Group timeline entries (show first 20 and last 10 for long narrations)
        if len(timeline_entries) > 35:
            timeline_str = "\n".join(timeline_entries[:20])
            timeline_str += f"\n  ... ({len(timeline_entries) - 30} more words) ...\n"
            timeline_str += "\n".join(timeline_entries[-10:])
        else:
            timeline_str = "\n".join(timeline_entries)

        # Find key transition words and their timestamps
        key_phrases = []
        transition_words = ["but", "however", "the", "this", "that", "so", "now", "finally", "first", "second", "third", "next", "then"]

        for i, wt in enumerate(word_timestamps):
            word = wt.get("word", "").lower().rstrip(",.!?")
            start = wt.get("start_seconds", 0)
            frame = int(start * 30)

            # Collect all words with their timing for context
            if word in transition_words or len(word) > 6:
                # Get surrounding context (2 words before and after)
                context_words = []
                for j in range(max(0, i - 2), min(len(word_timestamps), i + 3)):
                    context_words.append(word_timestamps[j].get("word", ""))
                context = " ".join(context_words)
                key_phrases.append(f'  - "{context}" → frame {frame} ({start:.2f}s)')

        # Limit key phrases shown
        key_phrases_str = "\n".join(key_phrases[:15]) if key_phrases else "  - (analyze the word timeline above to identify key moments)"

        total_frames = int(duration * 30)

        return f"""
## Word Timestamps (USE THESE FOR ANIMATION TIMING)

**CRITICAL**: DO NOT use percentage-based timing. Use the exact timestamps below to sync animations with narration.

**Scene Duration**: {duration:.2f}s = {total_frames} frames at 30fps

### Full Word Timeline:
{timeline_str}

### Key Moments (potential animation triggers):
{key_phrases_str}

### How to Use These Timestamps:
1. Read the voiceover and identify when key visual concepts are mentioned
2. Find that word/phrase in the timeline above
3. Set your animation phase to start AT or SLIGHTLY BEFORE that frame
4. Example: If narration says "the solution" at frame 150, set solutionAppears = 145

**DO NOT**:
- Use percentage-based timing (e.g., durationInFrames * 0.2)
- Guess when concepts are mentioned
- Ignore these timestamps

**DO**:
- Match animations to specific frame numbers from the timeline
- Start visuals 0-15 frames BEFORE the corresponding word is spoken
- Reference these timestamps in comments (e.g., // "solution" spoken at frame 150)
"""

    def _title_to_component_name(self, title: str) -> str:
        """Convert a scene title to a component name."""
        # Remove special characters and convert to PascalCase
        words = re.sub(r"[^a-zA-Z0-9\s]", "", title).split()
        pascal = "".join(word.capitalize() for word in words)
        return f"{pascal}Scene"

    def _component_to_registry_key(self, component_name: str) -> str:
        """Convert component name to registry key."""
        # Remove 'Scene' suffix and convert to snake_case
        name = component_name.replace("Scene", "")
        # Insert underscore before capitals and lowercase
        key = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()
        return key

    def _title_to_scene_key(self, title: str) -> str:
        """Convert a scene title to a scene key for registry.

        This must match how narration generates scene_ids (e.g., "scene1_hook").
        The storyboard extracts keys from scene_ids using split("_", 1)[1].

        Examples:
            "The Pixel Problem" -> "hook" (matches scene_type for hook)
            "The Tokenization Challenge" -> "tokenization_challenge"
            "Cutting Images Into Visual Words" -> "cutting_patches"
        """
        # Remove common prefixes like "The", "A", "An"
        words = title.split()
        if words and words[0].lower() in ("the", "a", "an"):
            words = words[1:]

        # Convert to snake_case: lowercase with underscores
        key = "_".join(word.lower() for word in words)

        # Remove any non-alphanumeric characters except underscores
        key = re.sub(r"[^a-z0-9_]", "", key)

        # Collapse multiple underscores
        key = re.sub(r"_+", "_", key).strip("_")

        return key

    def _title_to_registry_name(self, title: str) -> str:
        """Convert project title to registry constant name."""
        # Convert to UPPER_SNAKE_CASE
        words = re.sub(r"[^a-zA-Z0-9\s]", "", title).split()
        return "_".join(word.upper() for word in words[:3]) + "_SCENES"

    def _extract_code(self, response: str) -> str | None:
        """Extract TypeScript code from response."""
        # Try to find code block
        patterns = [
            r"```(?:typescript|tsx|ts)?\s*([\s\S]*?)```",
            r"```\s*([\s\S]*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()

        # If response looks like code, return as-is
        if "import" in response and "export" in response:
            return response.strip()

        return None


def generate_scenes(
    project_dir: Path,
    script_path: Path | None = None,
    force: bool = False,
    timeout: int = 300,
) -> dict[str, Any]:
    """Convenience function to generate all scenes for a project.

    Args:
        project_dir: Path to project directory
        script_path: Optional custom script path
        force: Overwrite existing scenes
        timeout: Timeout for each scene generation

    Returns:
        Generation results dict
    """
    generator = SceneGenerator(timeout=timeout)
    return generator.generate_all_scenes(
        project_dir=project_dir,
        script_path=script_path,
        force=force,
    )
