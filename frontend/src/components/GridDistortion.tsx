"use client";

import { useEffect, useRef } from "react";

interface GridDistortionProps {
  image?: string;
  strength?: number;
  className?: string;
}

const vertexShader = `
attribute vec2 uv;
attribute vec2 position;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const fragmentShader = `
precision highp float;
uniform sampler2D uTexture;
uniform vec2 uQuadSize;
uniform vec2 uMouse;
uniform float uTime;
uniform float uStrength;
varying vec2 vUv;

void main() {
  vec2 uv = vUv;
  vec2 quadDistance = abs(uv - 0.5) * uQuadSize;
  float velocity = uStrength;
  float mouseDistance = length(quadDistance - uMouse * uQuadSize);
  float zoom = 1.0 - 0.5 * velocity;
  float zoomRadius = 0.5 + 0.5 * velocity;
  float mouseRadius = min(zoomRadius, mouseDistance);
  zoom += smoothstep(0.0, zoomRadius, mouseRadius) * 0.5 * velocity;
  uv = (uv - 0.5) * zoom + 0.5;
  float wave = sin(mouseDistance * 10.0 - uTime * 4.0) * 0.03 * velocity;
  uv += normalize(quadDistance + 0.0001) * wave;
  gl_FragColor = texture2D(uTexture, uv);
}
`;

function makeGradientCanvas(): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d")!;

  const base = ctx.createLinearGradient(0, 0, 512, 512);
  base.addColorStop(0, "#1e1e2e");
  base.addColorStop(0.5, "#181825");
  base.addColorStop(1, "#11111b");
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, 512, 512);

  const glow = ctx.createRadialGradient(256, 256, 0, 256, 256, 360);
  glow.addColorStop(0, "rgba(250, 179, 135, 0.22)");
  glow.addColorStop(1, "rgba(250, 179, 135, 0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, 512, 512);

  ctx.strokeStyle = "rgba(250, 179, 135, 0.09)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 512; i += 32) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 512);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }
  return canvas;
}

export default function GridDistortion({
  image,
  strength = 0.35,
  className = "",
}: GridDistortionProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let animationFrameId = 0;
    let cleanupFn: (() => void) | undefined;
    let disposed = false;

    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };

    function onMouseMove(e: MouseEvent) {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      mouse.tx = (e.clientX - rect.left) / rect.width;
      mouse.ty = 1 - (e.clientY - rect.top) / rect.height;
    }
    window.addEventListener("mousemove", onMouseMove);

    import("ogl").then(({ Renderer, Program, Mesh, Triangle, Texture }) => {
      const renderer = new Renderer();
      const gl = renderer.gl;
      gl.clearColor(0, 0, 0, 0);

      const program = new Program(gl, {
        vertex: vertexShader,
        fragment: fragmentShader,
        uniforms: {
          uTexture: { value: null },
          uQuadSize: { value: [1, 1] },
          uMouse: { value: [0.5, 0.5] },
          uTime: { value: 0 },
          uStrength: { value: strength },
        },
      });

      function setTexture(img: HTMLCanvasElement | HTMLImageElement) {
        program.uniforms.uTexture.value = new Texture(gl, { image: img });
      }

      if (image) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => setTexture(img);
        img.src = image;
      } else {
        setTexture(makeGradientCanvas());
      }

      const mesh = new Mesh(gl, { geometry: new Triangle(gl), program });

      function resize() {
        const el = containerRef.current;
        if (!el) return;
        renderer.setSize(el.offsetWidth, el.offsetHeight);
        program.uniforms.uQuadSize.value = [
          el.offsetWidth / el.offsetHeight,
          1,
        ];
      }
      window.addEventListener("resize", resize);
      resize();

      function update(time: number) {
        animationFrameId = requestAnimationFrame(update);
        mouse.x += (mouse.tx - mouse.x) * 0.1;
        mouse.y += (mouse.ty - mouse.y) * 0.1;
        program.uniforms.uMouse.value = [mouse.x, mouse.y];
        program.uniforms.uTime.value = time * 0.001;
        renderer.render({ scene: mesh });
      }
      animationFrameId = requestAnimationFrame(update);
      const canvas = gl.canvas;
      container.appendChild(canvas);

      cleanupFn = () => {
        cancelAnimationFrame(animationFrameId);
        window.removeEventListener("resize", resize);
        window.removeEventListener("mousemove", onMouseMove);
        if (canvas.parentElement === container) {
          container.removeChild(canvas);
        }
        gl.getExtension("WEBGL_lose_context")?.loseContext();
      };

      // If the effect was cleaned up before the async import resolved,
      // tear down immediately to avoid an orphaned canvas (dev strict mode).
      if (disposed) cleanupFn();
    });

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      disposed = true;
      cleanupFn?.();
    };
  }, [image, strength]);

  return (
    <div
      ref={containerRef}
      className={`pointer-events-none ${className}`}
      style={{ position: "absolute", inset: 0, overflow: "hidden" }}
    />
  );
}
