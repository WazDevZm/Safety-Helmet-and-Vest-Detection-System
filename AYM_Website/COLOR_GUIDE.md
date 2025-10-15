# SDA AYM Website Color Guide
## Unified Color Palette System

This document outlines the complete color system for the SDA Adventist Youth Ministries website, ensuring visual consistency and spiritual symbolism across all pages and club sections.

---

## 🎨 **Global Color Theme (Main Site Palette)**

### Primary Colors
| Role | Color | Hex Code | Usage |
|------|-------|----------|-------|
| **Primary** | Deep Navy Blue | `#0A2342` | Main headers, navigation bar, buttons |
| **Secondary** | Golden Yellow | `#FFD700` | Highlights, icons, link hover effects |
| **Accent** | Forest Green | `#1B5E20` | Section dividers, call-to-action areas |
| **Background** | Off White | `#FAFAFA` | Page background for clean and soft look |

### Text Colors
| Role | Color | Hex Code | Usage |
|------|-------|----------|-------|
| **Main Text** | Charcoal Gray | `#333333` | Regular text and paragraph content |
| **Light Text** | Soft Gray | `#777777` | Subheadings, descriptions |
| **Link/Highlight** | Royal Blue | `#1565C0` | Links, hover effects, buttons |

### Status Colors
| Role | Color | Hex Code | Usage |
|------|-------|----------|-------|
| **Error/Alert** | Maroon | `#8B0000` | Warnings, alerts, or errors |
| **Success** | Emerald Green | `#2E7D32` | Confirmation messages |

---

## 🏠 **Page-Specific Color Applications**

### 1. Home Page
- **Background:** `#FAFAFA` (Off White)
- **Hero Section:** Gradient from `#0A2342` → `#1B5E20` (Deep Blue → Forest Green)
- **Text:** White on hero, charcoal gray elsewhere
- **Buttons:** Gold (`#FFD700`) with white text
- **Hover:** Button darkens to `#E6C200`

### 2. About Us Page
- **Background:** White (`#FFFFFF`)
- **Header Bar:** Deep Navy Blue (`#0A2342`)
- **Text:** Charcoal Gray (`#333333`)
- **Quote Box:** Light gold background (`#FFF8E1`) with italic text in dark blue

### 3. Events & Programs
- **Background:** Very light gray (`#F4F4F4`)
- **Event Cards:** White with blue borders (`#1565C0`)
- **Buttons:** Gold background, blue text
- **Hover:** Switch to dark blue background, white text

### 4. Spiritual Corner
- **Background:** Very soft cream (`#FFFDF5`)
- **Devotional titles:** Navy Blue (`#0A2342`)
- **Bible verses:** Gold accent (`#FFD700`) underline or border left line
- **Quote boxes:** Light blue background (`#E3F2FD`)

### 5. Gallery
- **Background:** White
- **Image frame hover:** Subtle gold glow or shadow
- **Title text:** Deep Navy Blue (`#0A2342`)

### 6. Contact & Join Page
- **Background:** White
- **Form fields:** Light gray borders (`#CCCCCC`)
- **Buttons:** Gold with blue hover
- **Icons:** Forest Green (`#1B5E20`)

### 7. Footer (Global)
- **Background:** Deep Navy Blue (`#0A2342`)
- **Text:** White
- **Links hover:** Gold underline
- **Social icons:** Gold or white

---

## 🎵 **Club-Specific Accent Colors**

Each club maintains its unique identity while following the unified design system:

| Club | Accent Color | Hex Code | Usage |
|------|-------------|----------|-------|
| **Pathfinder Club** | Maroon Red | `#800000` | Borders, badges, and headers |
| **Adventurer Club** | Royal Blue | `#1565C0` | Headings, button outlines |
| **Ambassador Club** | Deep Purple | `#512DA8` | Section headers, icons |
| **Master Guide Club** | Dark Green | `#1B5E20` | Titles, dividers |
| **Choir / Music Ministry** | Golden Yellow | `#FFD700` | Headings, accents |
| **Media / Drama Team** | Slate Gray | `#455A64` | Background cards, icons |

### Club Color Application Rules:
- Keep **backgrounds neutral** (white/light gray)
- Keep **text consistent** (charcoal gray)
- Only accent **headers, icons, and buttons** with club colors
- Maintain **consistent padding/margin** (1.5–2rem per section)

---

## 🌙 **Dark Mode Support**

For future dark mode implementation:

| Element | Dark Mode Color | Hex Code |
|---------|----------------|----------|
| **Background** | Dark Gray | `#121212` |
| **Text** | Light Gray | `#E0E0E0` |
| **Accent** | Golden Yellow | `#FFD700` |

---

## 🎨 **Color Symbolism**

### Spiritual Meaning:
- **Blue** – Represents faith, loyalty, and heaven
- **Gold** – Symbolizes divine glory and youth energy
- **Green** – Reflects growth, service, and renewal
- **White** – Purity and simplicity, key Christian tone

### Psychological Impact:
- **Navy Blue** – Trust, stability, professionalism
- **Golden Yellow** – Energy, optimism, divine light
- **Forest Green** – Growth, harmony, nature
- **Charcoal Gray** – Readability, sophistication

---

## 🛠️ **Implementation Guidelines**

### CSS Variables Usage:
```css
/* Use these variables throughout the site */
color: var(--primary-navy);
background-color: var(--secondary-gold);
border-color: var(--accent-green);
```

### Design Principles:
1. **Consistency** – Use the same colors for the same purposes
2. **Hierarchy** – Use color intensity to show importance
3. **Accessibility** – Ensure sufficient contrast ratios
4. **Unity** – Maintain visual harmony across all pages

### Button Styles:
- **Primary:** Gold background (`#FFD700`) with navy text
- **Secondary:** Navy background with white text
- **Hover:** Darken to `#E6C200` for gold buttons

### Form Elements:
- **Borders:** Light gray (`#CCCCCC`)
- **Focus:** Royal blue (`#1565C0`)
- **Error:** Maroon (`#8B0000`)
- **Success:** Emerald green (`#2E7D32`)

---

## 📱 **Responsive Considerations**

- Colors remain consistent across all device sizes
- Ensure touch targets have sufficient contrast
- Maintain readability on small screens
- Use color to guide user attention appropriately

---

## 🔧 **Maintenance Notes**

- Always use CSS variables instead of hardcoded hex values
- Test color combinations for accessibility compliance
- Maintain the 3-4 color rule per screen for calmness
- Keep buttons and icons rounded for youth-friendly appearance
- Use consistent fonts: *Poppins* for headings, *Open Sans* for body text

---

**Last Updated:** January 2024  
**Version:** 1.0  
**Maintained by:** AYM Web Development Team
