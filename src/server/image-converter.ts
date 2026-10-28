/**
 * Precision AI Vision & Multimodal Whiteboard Converter Engine
 *
 * Extracts and reconstructs editable native YY Whiteboard elements from:
 * - Microsoft Whiteboard exports & screenshots
 * - Photos of physical whiteboards
 * - Flowcharts, mind maps, architecture diagrams, wireframes & sketches
 * - PDF document pages, slides & vector/raster mixes
 * - Tables, handwritten notes, sticky notes & annotations
 *
 * Capabilities:
 * - Server-side Gemini Vision API (@google/genai)
 * - Emits exact WhiteboardCanvasObject structures matching the native YY Canvas engine:
 *   (id, type, x, y, w, h, rotation, color, fill, width, text, size, font, bold, italic, underline, bg, points, src)
 */

import { GoogleGenAI } from "@google/genai";

/**
 * Default Gemini Vision model.
 */
export const DEFAULT_GEMINI_VISION_MODEL = "gemini-2.5-flash";

export interface ConvertImageRequest {
  imageBase64: string;
  mimeType: string;
  width: number;
  height: number;
}

export type WhiteboardCanvasType =
  | "rect"
  | "roundRect"
  | "circle"
  | "ellipse"
  | "triangle"
  | "diamond"
  | "star"
  | "hexagon"
  | "heart"
  | "text"
  | "sticky"
  | "line"
  | "arrow"
  | "doubleArrow"
  | "dashed"
  | "pen"
  | "highlighter"
  | "image"
  | "emoji";

export interface WhiteboardCanvasObject {
  id: string;
  type: WhiteboardCanvasType;
  x: number;
  y: number;
  w: number;
  h: number;
  rotation: number;
  color?: string | undefined;
  fill?: boolean | undefined;
  width?: number | undefined; // stroke width in px
  text?: string | undefined;
  size?: number | undefined; // font size in px
  font?: string | undefined;
  bold?: boolean | undefined;
  italic?: boolean | undefined;
  underline?: boolean | undefined;
  bg?: string | undefined; // background color for sticky notes or filled shapes
  points?: Array<{ x: number; y: number }> | undefined; // for pen / highlighter
  src?: string | undefined; // image source
  opacity?: number | undefined;
  isPlaceholder?: boolean | undefined;
  penStyle?: number | undefined;
  highlighterStyle?: number | undefined;
  startX?: number | undefined;
  startY?: number | undefined;
  endX?: number | undefined;
  endY?: number | undefined;
}

export type DetectedObject = WhiteboardCanvasObject;

// Exported types for test suite and client consumption
export type DetectedTextObject = {
  type: "text";
  text: string;
  x: number;
  y: number;
  width: number;
  height: number;
  fontSize: number;
  color?: string | undefined;
  bold?: boolean | undefined;
  italic?: boolean | undefined;
  underline?: boolean | undefined;
  rotation?: number | undefined;
};

export type DetectedStickyObject = {
  type: "sticky";
  text?: string | undefined;
  x: number;
  y: number;
  width: number;
  height: number;
  bg?: string | undefined;
  color?: string | undefined;
  fontSize?: number | undefined;
  rotation?: number | undefined;
};

export type DetectedShapeObject = {
  type: "shape";
  shapeType: string;
  x: number;
  y: number;
  width: number;
  height: number;
  color?: string | undefined;
  fill?: boolean | undefined;
  strokeWidth?: number | undefined;
  rotation?: number | undefined;
};

export type DetectedConnectorObject = {
  type: "connector";
  connectorType: string;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  color?: string | undefined;
  strokeWidth?: number | undefined;
};

export type DetectedDrawingObject = {
  type: "drawing";
  points: Array<{ x: number; y: number }>;
  color?: string | undefined;
  strokeWidth?: number | undefined;
  isHighlighter?: boolean | undefined;
};

export type DetectedEmbeddedImageObject = {
  type: "embeddedImage";
  x: number;
  y: number;
  width: number;
  height: number;
  description?: string | undefined;
};

export interface ConversionResponse {
  title: string;
  imageWidth: number;
  imageHeight: number;
  objects: WhiteboardCanvasObject[];
  source: "gemini-vision";
  counts: {
    text: number;
    sticky: number;
    shapes: number;
    connectors: number;
    drawings: number;
    embeddedImages: number;
    total: number;
  };
}

function genId(): string {
  return `el_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Returns clean base64 data string without header prefixes.
 */
function cleanBase64(input: string): string {
  if (input.includes(",")) {
    return input.split(",")[1] || "";
  }
  return input;
}

/**
 * Defensive JSON extractor and parser.
 * Extracts clean JSON or parses the first balanced {...} block if markdown/wrapper exists.
 */
function extractAndParseJson(text: string): Record<string, unknown> | null {
  if (!text || typeof text !== "string") return null;
  let cleaned = text.trim();

  // Strip markdown code fences if present
  if (cleaned.startsWith("```")) {
    cleaned = cleaned
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/i, "")
      .trim();
  }

  // 1. Direct JSON parse
  try {
    const parsed = JSON.parse(cleaned);
    if (parsed && typeof parsed === "object") return parsed as Record<string, unknown>;
  } catch {
    // 2. Extract first {...} block
    const firstBrace = cleaned.indexOf("{");
    const lastBrace = cleaned.lastIndexOf("}");
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      try {
        const sub = cleaned.substring(firstBrace, lastBrace + 1);
        const parsed = JSON.parse(sub);
        if (parsed && typeof parsed === "object") return parsed as Record<string, unknown>;
      } catch (e) {
        console.warn("[ImageConverter] Defensive JSON block extraction failed:", e);
      }
    }
  }

  return null;
}

/**
 * Processes and normalizes raw AI-detected objects from Gemini into native WhiteboardCanvasObjects.
 * Maps exact JSON schema from prompt into internal whiteboard element types.
 */
function processRawAiObjects(
  rawObjects: Array<Record<string, unknown>>,
  width: number,
  height: number,
): WhiteboardCanvasObject[] {
  const convertedObjects: WhiteboardCanvasObject[] = [];

  for (const item of rawObjects) {
    if (!item || typeof item !== "object") continue;

    const rawType = String(item["type"] || "rect")
      .toLowerCase()
      .trim();

    // 1. Text elements -> native "text" element type
    if (rawType === "text") {
      const text = item["text"] != null ? String(item["text"]).trim() : "";
      if (!text) continue;

      const rawX = Number(item["x"]) || 0;
      const rawY = Number(item["y"]) || 0;
      const rawW = Number(item["width"] ?? item["w"]) || 160;
      const rawH = Number(item["height"] ?? item["h"]) || 36;
      const rawFontSize = Number(item["fontSize"] ?? item["size"]) || 18;

      const x = Math.max(0, Math.min(width - 10, Math.round(rawX)));
      const y = Math.max(0, Math.min(height - 10, Math.round(rawY)));
      const w = Math.max(20, Math.min(width, Math.round(rawW)));
      const h = Math.max(16, Math.min(height, Math.round(rawH)));

      const color =
        typeof item["color"] === "string" && item["color"].startsWith("#")
          ? item["color"]
          : "#1E1E1E";

      const bold = item["fontWeight"] === "bold" || Boolean(item["bold"]);
      const italic =
        item["fontStyle"] === "italic" ||
        item["fontWeight"] === "italic" ||
        Boolean(item["italic"]);
      const fontFamily =
        typeof item["fontFamily"] === "string" && item["fontFamily"].trim()
          ? item["fontFamily"].trim()
          : typeof item["font"] === "string" && item["font"].trim()
            ? item["font"].trim()
            : "Segoe UI,Inter,system-ui,sans-serif";

      convertedObjects.push({
        id: genId(),
        type: "text",
        x,
        y,
        w,
        h,
        rotation: 0,
        text,
        color,
        size: Math.max(12, Math.min(140, Math.round(rawFontSize))),
        font: fontFamily,
        bold,
        italic,
        underline: Boolean(item["underline"]),
        isPlaceholder: false,
      });
    }
    // 2. Sticky notes
    else if (rawType === "sticky") {
      const rawX = Number(item["x"]) || 0;
      const rawY = Number(item["y"]) || 0;
      const rawW = Number(item["width"] ?? item["w"]) || 180;
      const rawH = Number(item["height"] ?? item["h"]) || 140;

      const x = Math.max(0, Math.min(width - 10, Math.round(rawX)));
      const y = Math.max(0, Math.min(height - 10, Math.round(rawY)));
      const w = Math.max(60, Math.min(width, Math.round(rawW)));
      const h = Math.max(60, Math.min(height, Math.round(rawH)));

      const text = item["text"] != null ? String(item["text"]).trim() : "";
      const bg =
        typeof item["bg"] === "string" && item["bg"].startsWith("#")
          ? item["bg"]
          : typeof item["fill"] === "string" && item["fill"].startsWith("#")
            ? item["fill"]
            : "#fef08a";

      const color =
        typeof item["color"] === "string" && item["color"].startsWith("#")
          ? item["color"]
          : "#422006";

      const fontSize = Math.max(
        12,
        Math.min(48, Math.round(Number(item["fontSize"] ?? item["size"]) || 16)),
      );

      convertedObjects.push({
        id: genId(),
        type: "sticky",
        x,
        y,
        w,
        h,
        rotation: 0,
        text,
        bg,
        color,
        size: fontSize,
        font: "Segoe UI,Inter,system-ui,sans-serif",
        bold: Boolean(item["bold"]),
        italic: Boolean(item["italic"]),
        underline: false,
        isPlaceholder: !text,
      });
    }
    // 3. Rect / RoundRect / Shapes (including triangle, diamond, star, hexagon, heart, ellipse, circle)
    else if (
      rawType === "rect" ||
      rawType === "roundrect" ||
      rawType === "shape" ||
      rawType === "triangle" ||
      rawType === "diamond" ||
      rawType === "star" ||
      rawType === "hexagon" ||
      rawType === "heart"
    ) {
      const rx = Number(item["rx"]) || 0;
      let targetType: WhiteboardCanvasType = "rect";
      if (
        rawType === "triangle" ||
        rawType === "diamond" ||
        rawType === "star" ||
        rawType === "hexagon" ||
        rawType === "heart"
      ) {
        targetType = rawType;
      } else if (item["shapeType"] && typeof item["shapeType"] === "string") {
        targetType = item["shapeType"] as WhiteboardCanvasType;
      } else if (rawType === "roundrect" || rx > 4) {
        targetType = "roundRect";
      }

      const rawX = Number(item["x"]) || 0;
      const rawY = Number(item["y"]) || 0;
      const rawW = Number(item["width"] ?? item["w"]) || 120;
      const rawH = Number(item["height"] ?? item["h"]) || 80;

      const x = Math.max(0, Math.min(width - 10, Math.round(rawX)));
      const y = Math.max(0, Math.min(height - 10, Math.round(rawY)));
      const w = Math.max(16, Math.min(width, Math.round(rawW)));
      const h = Math.max(16, Math.min(height, Math.round(rawH)));

      const stroke =
        typeof item["stroke"] === "string" && item["stroke"].startsWith("#")
          ? item["stroke"]
          : typeof item["color"] === "string" && item["color"].startsWith("#")
            ? item["color"]
            : "#1E1E1E";

      const fillVal = item["fill"];
      const isFilled = fillVal !== "transparent" && fillVal !== false && Boolean(fillVal);
      const bg =
        typeof item["bg"] === "string" && item["bg"].startsWith("#")
          ? item["bg"]
          : typeof fillVal === "string" && fillVal.startsWith("#")
            ? fillVal
            : isFilled
              ? stroke
              : undefined;

      const strokeW = Math.max(
        1,
        Math.min(16, Math.round(Number(item["strokeWidth"] ?? item["width"]) || 2)),
      );
      const text = item["text"] != null ? String(item["text"]).trim() : "";

      convertedObjects.push({
        id: genId(),
        type: targetType,
        x,
        y,
        w,
        h,
        rotation: 0,
        color: stroke,
        fill: isFilled,
        bg: isFilled ? bg : undefined,
        width: strokeW,
      });

      // If shape container had embedded label text, emit a paired text object inside the shape
      if (text) {
        convertedObjects.push({
          id: genId(),
          type: "text",
          x: x + 8,
          y: y + Math.max(4, Math.round(h / 2 - 12)),
          w: Math.max(40, w - 16),
          h: Math.max(20, Math.round(h / 2)),
          rotation: 0,
          text,
          color: stroke,
          size: Math.max(12, Math.min(28, Math.round(h * 0.25))),
          font: "Segoe UI,Inter,system-ui,sans-serif",
          bold: true,
          italic: false,
          underline: false,
          isPlaceholder: false,
        });
      }
    }
    // 4. Circle / Ellipse elements -> native "circle" or "ellipse" shape element type
    else if (rawType === "circle" || rawType === "ellipse") {
      const radius = Number(item["radius"]) || 0;
      const rawW = radius > 0 ? radius * 2 : Number(item["width"] ?? item["w"]) || 100;
      const rawH = radius > 0 ? radius * 2 : Number(item["height"] ?? item["h"]) || 100;
      const rawX = Number(item["x"]) || 0;
      const rawY = Number(item["y"]) || 0;

      const x = Math.max(0, Math.min(width - 10, Math.round(rawX)));
      const y = Math.max(0, Math.min(height - 10, Math.round(rawY)));
      const w = Math.max(16, Math.min(width, Math.round(rawW)));
      const h = Math.max(16, Math.min(height, Math.round(rawH)));

      const stroke =
        typeof item["stroke"] === "string" && item["stroke"].startsWith("#")
          ? item["stroke"]
          : typeof item["color"] === "string" && item["color"].startsWith("#")
            ? item["color"]
            : "#1E1E1E";

      const fillVal = item["fill"];
      const isFilled = fillVal !== "transparent" && fillVal !== false && Boolean(fillVal);
      const bg =
        typeof item["bg"] === "string" && item["bg"].startsWith("#")
          ? item["bg"]
          : typeof fillVal === "string" && fillVal.startsWith("#")
            ? fillVal
            : isFilled
              ? stroke
              : undefined;

      const strokeW = Math.max(
        1,
        Math.min(16, Math.round(Number(item["strokeWidth"] ?? item["width"]) || 2)),
      );

      convertedObjects.push({
        id: genId(),
        type: rawType === "ellipse" ? "ellipse" : "circle",
        x,
        y,
        w,
        h,
        rotation: 0,
        color: stroke,
        fill: isFilled,
        bg: isFilled ? bg : undefined,
        width: strokeW,
      });

      const text = item["text"] != null ? String(item["text"]).trim() : "";
      if (text) {
        convertedObjects.push({
          id: genId(),
          type: "text",
          x: x + 8,
          y: y + Math.max(4, Math.round(h / 2 - 12)),
          w: Math.max(40, w - 16),
          h: Math.max(20, Math.round(h / 2)),
          rotation: 0,
          text,
          color: stroke,
          size: Math.max(12, Math.min(28, Math.round(h * 0.25))),
          font: "Segoe UI,Inter,system-ui,sans-serif",
          bold: true,
          italic: false,
          underline: false,
          isPlaceholder: false,
        });
      }
    }
    // 5. Line / Arrow / DoubleArrow / Dashed
    else if (
      rawType === "line" ||
      rawType === "arrow" ||
      rawType === "doublearrow" ||
      rawType === "dashed" ||
      rawType === "connector"
    ) {
      let targetType: WhiteboardCanvasType = "arrow";
      if (rawType === "line") targetType = "line";
      else if (rawType === "doublearrow") targetType = "doubleArrow";
      else if (rawType === "dashed") targetType = "dashed";
      else if (item["connectorType"]) {
        const ct = String(item["connectorType"]).toLowerCase();
        if (ct === "doublearrow") targetType = "doubleArrow";
        else if (ct === "dashed") targetType = "dashed";
        else if (ct === "line") targetType = "line";
      }

      const x1 = Number(item["x1"] ?? item["startX"]);
      const y1 = Number(item["y1"] ?? item["startY"]);
      const x2 = Number(item["x2"] ?? item["endX"]);
      const y2 = Number(item["y2"] ?? item["endY"]);

      let sx = 0,
        sy = 0,
        ex = 100,
        ey = 50,
        x = 0,
        y = 0,
        w = 100,
        h = 50;

      if (!isNaN(x1) && !isNaN(x2) && !isNaN(y1) && !isNaN(y2)) {
        sx = Math.round(x1);
        sy = Math.round(y1);
        ex = Math.round(x2);
        ey = Math.round(y2);
        x = Math.min(sx, ex);
        y = Math.min(sy, ey);
        w = Math.max(1, Math.abs(ex - sx));
        h = Math.max(1, Math.abs(ey - sy));
      } else {
        const rawX = Number(item["x"]) || 0;
        const rawY = Number(item["y"]) || 0;
        const rawW = Number(item["width"] ?? item["w"]) || 100;
        const rawH = Number(item["height"] ?? item["h"]) || 2;
        x = Math.max(0, Math.round(rawX));
        y = Math.max(0, Math.round(rawY));
        w = Math.max(1, Math.round(rawW));
        h = Math.max(1, Math.round(rawH));
        sx = x;
        sy = y;
        ex = x + w;
        ey = y + h;
      }

      const stroke =
        typeof item["stroke"] === "string" && item["stroke"].startsWith("#")
          ? item["stroke"]
          : typeof item["color"] === "string" && item["color"].startsWith("#")
            ? item["color"]
            : "#1E1E1E";

      const strokeW = Math.max(
        1,
        Math.min(16, Math.round(Number(item["strokeWidth"] ?? item["width"]) || 2)),
      );

      convertedObjects.push({
        id: genId(),
        type: targetType,
        x,
        y,
        w,
        h,
        rotation: 0,
        color: stroke,
        width: strokeW,
        startX: sx,
        startY: sy,
        endX: ex,
        endY: ey,
      });
    }
    // 6. Tables -> Deconstructed into real editable cell boxes and cell text elements!
    else if (rawType === "table") {
      const rawX = Number(item["x"]) || 0;
      const rawY = Number(item["y"]) || 0;
      const rawW = Number(item["width"] ?? item["w"]) || 400;
      const rawH = Number(item["height"] ?? item["h"]) || 200;

      const tableX = Math.max(0, Math.min(width - 10, Math.round(rawX)));
      const tableY = Math.max(0, Math.min(height - 10, Math.round(rawY)));
      const tableW = Math.max(60, Math.min(width, Math.round(rawW)));
      const tableH = Math.max(40, Math.min(height, Math.round(rawH)));

      const cells = Array.isArray(item["cells"])
        ? (item["cells"] as Array<Record<string, unknown>>)
        : [];

      if (cells.length > 0) {
        // Emit individual cell shapes & text elements
        for (const cell of cells) {
          const cx = Math.round(Number(cell["x"]) || tableX);
          const cy = Math.round(Number(cell["y"]) || tableY);
          const cw = Math.max(20, Math.round(Number(cell["width"] ?? cell["w"]) || 100));
          const ch = Math.max(16, Math.round(Number(cell["height"] ?? cell["h"]) || 30));

          const isHeader = Boolean(cell["isHeader"]);
          const cellBg =
            typeof cell["bg"] === "string" && cell["bg"].startsWith("#")
              ? cell["bg"]
              : isHeader
                ? "#f1f5f9"
                : "#ffffff";

          const cellStroke =
            typeof cell["stroke"] === "string" && cell["stroke"].startsWith("#")
              ? cell["stroke"]
              : "#cbd5e1";

          // 1. Cell rectangle
          convertedObjects.push({
            id: genId(),
            type: "rect",
            x: cx,
            y: cy,
            w: cw,
            h: ch,
            rotation: 0,
            color: cellStroke,
            fill: true,
            bg: cellBg,
            width: 1,
          });

          // 2. Cell text
          const cText = cell["text"] != null ? String(cell["text"]).trim() : "";
          if (cText) {
            const textColor =
              typeof cell["color"] === "string" && cell["color"].startsWith("#")
                ? cell["color"]
                : isHeader
                  ? "#0f172a"
                  : "#334155";
            const cFontSize = Math.max(
              11,
              Math.min(24, Math.round(Number(cell["fontSize"]) || (isHeader ? 14 : 13))),
            );

            convertedObjects.push({
              id: genId(),
              type: "text",
              x: cx + 6,
              y: cy + Math.max(2, Math.round(ch / 2 - 9)),
              w: Math.max(20, cw - 12),
              h: Math.max(16, ch - 4),
              rotation: 0,
              text: cText,
              color: textColor,
              size: cFontSize,
              font: "Segoe UI,Inter,system-ui,sans-serif",
              bold: isHeader || Boolean(cell["bold"]),
              italic: Boolean(cell["italic"]),
              underline: false,
              isPlaceholder: false,
            });
          }
        }
      } else {
        // Fallback: table outline rectangle
        convertedObjects.push({
          id: genId(),
          type: "rect",
          x: tableX,
          y: tableY,
          w: tableW,
          h: tableH,
          rotation: 0,
          color: "#94a3b8",
          fill: false,
          width: 2,
        });
      }
    }
    // 7. Freehand drawings & ink marks (pen or highlighter)
    else if (rawType === "drawing" || rawType === "pen" || rawType === "highlighter") {
      const isHighlighter = rawType === "highlighter" || Boolean(item["isHighlighter"]);
      const points = Array.isArray(item["points"])
        ? (item["points"] as Array<Record<string, unknown>>)
            .map((p) => ({
              x: Math.round(Number(p["x"]) || 0),
              y: Math.round(Number(p["y"]) || 0),
            }))
            .filter((p) => p.x >= 0 && p.y >= 0)
        : [];

      const stroke =
        typeof item["stroke"] === "string" && item["stroke"].startsWith("#")
          ? item["stroke"]
          : typeof item["color"] === "string" && item["color"].startsWith("#")
            ? item["color"]
            : isHighlighter
              ? "#fde047"
              : "#1E1E1E";

      const strokeW = Math.max(
        1,
        Math.min(
          28,
          Math.round(Number(item["strokeWidth"] ?? item["width"]) || (isHighlighter ? 12 : 3)),
        ),
      );

      if (points.length >= 2) {
        convertedObjects.push({
          id: genId(),
          type: isHighlighter ? "highlighter" : "pen",
          x: points[0]?.x ?? 0,
          y: points[0]?.y ?? 0,
          w: 100,
          h: 50,
          rotation: 0,
          color: stroke,
          width: strokeW,
          points,
          opacity: isHighlighter ? 0.45 : 1,
        });
      }
    }
    // 8. Embedded Photos & Raster Artwork
    else if (rawType === "image" || rawType === "embeddedimage") {
      const rawX = Number(item["x"]) || 0;
      const rawY = Number(item["y"]) || 0;
      const rawW = Number(item["width"] ?? item["w"]) || 150;
      const rawH = Number(item["height"] ?? item["h"]) || 120;

      const x = Math.max(0, Math.min(width - 10, Math.round(rawX)));
      const y = Math.max(0, Math.min(height - 10, Math.round(rawY)));
      const w = Math.max(20, Math.min(width, Math.round(rawW)));
      const h = Math.max(20, Math.min(height, Math.round(rawH)));

      convertedObjects.push({
        id: genId(),
        type: "image",
        x,
        y,
        w,
        h,
        rotation: 0,
        src: typeof item["src"] === "string" ? item["src"] : undefined,
      });
    }
  }

  return convertedObjects;
}

/**
 * Calls server-side Gemini Vision API using @google/genai.
 * Exactly one request per conversion.
 */
async function callGeminiVision(
  rawBase64: string,
  mimeType: string,
  width: number,
  height: number,
): Promise<{ title: string; objects: WhiteboardCanvasObject[] }> {
  const apiKey = process.env["GEMINI_API_KEY"];
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is missing.");
  }

  const ai = new GoogleGenAI({
    apiKey,
    httpOptions: { headers: { "User-Agent": "aistudio-build" } },
  });

  const base64Data = cleanBase64(rawBase64);
  const cleanMime =
    mimeType.includes("jpeg") || mimeType.includes("jpg")
      ? "image/jpeg"
      : mimeType.includes("webp")
        ? "image/webp"
        : "image/png";

  const prompt = `You are an expert AI vision system that deconstructs whiteboard photos, document screenshots, architecture diagrams, flowcharts, tables, mind maps, notes, and sketches into exact native vector elements.

Analyze this image (${width}x${height}px).
Extract and reconstruct ALL visible visual elements as structured vector elements:

1. TEXT:
   - Extract titles, headings, body text, bullet points ("• ..."), numbered items, labels, and captions.
   - For HANDWRITING: transcribe readable handwriting into clear editable text with its original color and size. If handwriting or doodle is uncertain, do not hallucinate fake words—emit it as ink drawing.
   - Detect: text, x, y, width, height, fontSize (estimated px size on this canvas), fontWeight ("bold" | "normal"), fontStyle ("italic" | "normal"), color (real hex color), textAlign ("left" | "center" | "right").

2. STICKY NOTES:
   - Identify sticky notes / post-its.
   - Detect: x, y, width, height, bg (exact hex background color, e.g. yellow #fef08a, orange #fed7aa, green #bbf7d0, cyan #bae6fd, pink #fbcfe8, purple #e9d5ff), text, color (text color, e.g. #422006), fontSize.

3. SHAPES:
   - Identify geometric shapes: "rect", "roundRect", "circle", "ellipse", "triangle", "diamond", "star", "hexagon", "heart".
   - Detect: type, x, y, width, height, stroke (hex border color), strokeWidth (px), fill (true/false), bg (hex fill color if filled).
   - If text is written inside the shape, include "text" property with the inside text.

4. CONNECTORS & ARROWS:
   - Identify lines and arrows connecting elements.
   - Detect: type ("line" | "arrow" | "doubleArrow" | "dashed"), startX, startY, endX, endY, stroke (hex color), strokeWidth (px).

5. TABLES:
   - If there is a table or data grid, deconstruct it into:
     type: "table", x, y, width, height, rows, cols, cells: [ { text, x, y, width, height, isHeader, bg, color, fontSize } ]

6. DRAWINGS / INK:
   - Freehand pen doodles, circles, arrows drawn by hand, or highlighter strokes.
   - Detect: type ("pen" | "highlighter"), points: [{x, y}], stroke (hex color), strokeWidth.

7. PHOTOS & COMPLEX ARTWORK:
   - If there is an embedded photograph, screenshot, or complex raster artwork, mark as type: "image", x, y, width, height.

CRITICAL RULES:
- Coordinates (x, y, width, height) MUST match the image dimensions (${width}x${height}px).
- PRESERVE REAL COLORS: Do NOT turn everything black! Use real hex colors seen in the image (#ef4444 red, #2563eb blue, #16a34a green, #f59e0b amber, #fef08a yellow, #8b5cf6 purple).
- PRESERVE LAYOUT: Keep positions, hierarchy, and connections accurate.
- Output raw JSON ONLY matching the schema. No markdown, no commentary.

SCHEMA:
{
  "title": "Descriptive Diagram / Whiteboard Title",
  "canvasWidth": ${width},
  "canvasHeight": ${height},
  "elements": [
    { "type": "text", "text": "...", "x": 10, "y": 20, "width": 200, "height": 30, "fontSize": 18, "fontWeight": "normal", "color": "#1e293b" },
    { "type": "sticky", "text": "...", "x": 100, "y": 100, "width": 180, "height": 140, "bg": "#fef08a", "color": "#422006", "fontSize": 16 },
    { "type": "rect", "x": 50, "y": 50, "width": 120, "height": 80, "fill": false, "stroke": "#2563eb", "strokeWidth": 2, "text": "Label" },
    { "type": "circle", "x": 200, "y": 50, "width": 80, "height": 80, "fill": true, "bg": "#dbeafe", "stroke": "#2563eb", "strokeWidth": 2 },
    { "type": "arrow", "startX": 170, "startY": 90, "endX": 200, "endY": 90, "stroke": "#64748b", "strokeWidth": 2 }
  ]
}`;

  const candidateModels = [
    DEFAULT_GEMINI_VISION_MODEL,
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-flash-latest",
  ];

  let rawText = "";
  let lastError: Error | null = null;

  for (const modelName of candidateModels) {
    try {
      console.log(`[ImageConverter] Calling Gemini Vision model: ${modelName}`);
      const startTime = Date.now();

      const response = await ai.models.generateContent({
        model: modelName,
        contents: [
          {
            role: "user",
            parts: [
              { text: prompt },
              {
                inlineData: {
                  data: base64Data,
                  mimeType: cleanMime,
                },
              },
            ],
          },
        ],
        config: {
          responseMimeType: "application/json",
        },
      });

      const duration = Date.now() - startTime;
      rawText = response.text || "";
      console.log(
        `[ImageConverter] Gemini Vision (${modelName}) responded in ${duration}ms, response length: ${rawText.length}`,
      );

      if (rawText.trim()) {
        break;
      }
    } catch (err: unknown) {
      lastError = err instanceof Error ? err : new Error(String(err));
      console.warn(`[ImageConverter] Model ${modelName} failed:`, lastError.message);
    }
  }

  if (!rawText.trim()) {
    if (
      lastError &&
      (lastError.message.includes("429") ||
        lastError.message.includes("RESOURCE_EXHAUSTED") ||
        lastError.message.includes("Quota exceeded"))
    ) {
      throw new Error("Gemini API rate limit reached. Please wait a few seconds and try again.");
    }
    throw lastError || new Error("Gemini returned an empty response.");
  }

  const parsed = extractAndParseJson(rawText);
  if (!parsed) {
    throw new Error("Failed to parse JSON response from Gemini.");
  }

  const rawElements = Array.isArray(parsed["elements"])
    ? (parsed["elements"] as Array<Record<string, unknown>>)
    : Array.isArray(parsed["objects"])
      ? (parsed["objects"] as Array<Record<string, unknown>>)
      : [];

  console.log(`[ImageConverter] Parsed ${rawElements.length} raw elements from Gemini Vision`);

  const convertedObjects = processRawAiObjects(rawElements, width, height);

  if (convertedObjects.length === 0) {
    throw new Error("No editable whiteboard elements could be detected in the uploaded image.");
  }

  return {
    title: typeof parsed["title"] === "string" ? parsed["title"] : "Imported Whiteboard",
    objects: convertedObjects,
  };
}

export function compileConversionResponse(
  title: string,
  width: number,
  height: number,
  validObjects: WhiteboardCanvasObject[],
): ConversionResponse {
  let textCount = 0;
  let stickyCount = 0;
  let shapesCount = 0;
  let connectorsCount = 0;
  let drawingsCount = 0;
  let imagesCount = 0;

  for (const obj of validObjects) {
    if (obj.type === "text") textCount++;
    else if (obj.type === "sticky") stickyCount++;
    else if (
      obj.type === "rect" ||
      obj.type === "roundRect" ||
      obj.type === "circle" ||
      obj.type === "ellipse" ||
      obj.type === "triangle" ||
      obj.type === "diamond" ||
      obj.type === "star" ||
      obj.type === "hexagon" ||
      obj.type === "heart"
    )
      shapesCount++;
    else if (
      obj.type === "line" ||
      obj.type === "arrow" ||
      obj.type === "doubleArrow" ||
      obj.type === "dashed"
    )
      connectorsCount++;
    else if (obj.type === "pen" || obj.type === "highlighter") drawingsCount++;
    else if (obj.type === "image") imagesCount++;
  }

  return {
    title: title || "Imported Whiteboard",
    imageWidth: width,
    imageHeight: height,
    objects: validObjects,
    source: "gemini-vision",
    counts: {
      text: textCount,
      sticky: stickyCount,
      shapes: shapesCount,
      connectors: connectorsCount,
      drawings: drawingsCount,
      embeddedImages: imagesCount,
      total: validObjects.length,
    },
  };
}

/**
 * Converts a whiteboard image / screenshot / diagram / document page into native, editable whiteboard objects.
 */
export async function convertWhiteboardImage(
  req: ConvertImageRequest,
): Promise<ConversionResponse> {
  const { width, height, imageBase64, mimeType } = req;

  console.log(`[ImageConverter] ========================================`);
  console.log(`[ImageConverter] Conversion started (${width}x${height})`);

  const geminiResult = await callGeminiVision(imageBase64, mimeType, width, height);

  console.log(
    `[ImageConverter] Successfully extracted ${geminiResult.objects.length} editable objects via Gemini`,
  );

  return compileConversionResponse(geminiResult.title, width, height, geminiResult.objects);
}
