import React, { useState, useEffect, useRef, Suspense, useMemo } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Stars, Float, Sparkles } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// --- Configuration ---
const TREE_HEIGHT = 18;
const TREE_RADIUS = 7.5;
// Increased speeds for snappier reaction
const EXPLOSION_SPEED = 0.12;
const RETURN_SPEED = 0.10;

// Particle Counts
const COUNT_LEAVES = 4500;
const COUNT_RIBBON = 600;    
const COUNT_SNOW = 1000;
const COUNT_ORNAMENTS = 300; 

// Emojis list
const HOLIDAY_EMOJIS = ["🎁", "🎄", "🧦", "🦌", "🔔", "⛄", "🍭", "❄️", "🎅", "🍪", "🐶", "🕯️", "🎀", "🌟", "🎈"];
// Increased to 30 per type -> 30 * 15 = 450 total (~10% of 4500 leaves)
const EMOJI_COUNT_PER_TYPE = 30; 

// --- Static Colors (Defined outside component to prevent re-renders) ---
const LEAF_COLORS = ["#16a34a", "#22c55e", "#15803d", "#4ade80"];
const ORNAMENT_COLORS = ["#fbbf24", "#f59e0b", "#2dd4bf"];
const RIBBON_COLORS = ["#fbbf24", "#fcd34d"];

// --- Types ---
type InteractionState = "tree" | "exploded";

// --- Helper Math ---
const randomRange = (min: number, max: number) => Math.random() * (max - min) + min;

// --- Global Singletons for MediaPipe ---
let visionTasksPromise: Promise<any> | null = null;
const getVisionTasks = () => {
    if (!visionTasksPromise) {
        visionTasksPromise = FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
        ).catch((err) => {
            console.error("Failed to load Vision Tasks WASM", err);
            visionTasksPromise = null;
            throw err;
        });
    }
    return visionTasksPromise;
};

// --- Components ---

const HandTracker = ({ 
    onUpdateInteraction 
}: { 
    onUpdateInteraction: (isOpen: boolean, handX: number) => void 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [loading, setLoading] = useState(false);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const animationFrameRef = useRef<number>(0);
  const lastVideoTimeRef = useRef<number>(-1);
  const initRef = useRef(false);

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  const enableCam = async () => {
    if (loading || initRef.current) return;
    setLoading(true);
    initRef.current = true;
    try {
        const vision = await getVisionTasks();
        if (!landmarkerRef.current) {
            landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numHands: 1
            });
        }
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.addEventListener("loadeddata", predictWebcam);
            setPermissionGranted(true);
        }
    } catch (err) {
      console.error("Camera Error:", err);
      setLoading(false);
      initRef.current = false;
    }
  };

  const predictWebcam = () => {
    if (!landmarkerRef.current || !videoRef.current) return;
    if (videoRef.current.videoWidth === 0) {
      animationFrameRef.current = requestAnimationFrame(predictWebcam);
      return;
    }
    let startTimeMs = performance.now();
    if (videoRef.current.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = videoRef.current.currentTime;
      try {
        const results = landmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
        if (results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0];
          const wrist = landmarks[0];
          const tips = [4, 8, 12, 16, 20];
          let totalDist = 0;
          tips.forEach(tipIdx => {
            const tip = landmarks[tipIdx];
            totalDist += Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
          });
          const isOpen = (totalDist / 5) > 0.35; 
          onUpdateInteraction(isOpen, wrist.x);
        } else {
          onUpdateInteraction(false, 0.5);
        }
      } catch (e) { console.warn(e); }
    }
    animationFrameRef.current = requestAnimationFrame(predictWebcam);
  };

  return (
    <div className="absolute bottom-4 left-4 z-50 flex flex-col items-start gap-2">
      <video ref={videoRef} className="w-24 h-16 rounded-lg border border-slate-700 bg-black opacity-60" autoPlay playsInline muted style={{ transform: 'scaleX(-1)' }} />
      {!permissionGranted && (
        <button onClick={enableCam} disabled={loading} className="bg-emerald-950/80 text-amber-200 px-6 py-2 rounded-none border border-amber-500 hover:bg-emerald-900 transition-all font-serif uppercase tracking-widest text-xs shadow-[0_0_15px_rgba(251,191,36,0.3)]">
          {loading ? "INITIALIZING..." : "ENABLE HAND CONTROL"}
        </button>
      )}
    </div>
  );
};

// --- Texture Generator for Emojis ---
function useEmojiTexture(emoji: string) {
  return useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d')!;
    ctx.font = '100px serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'white';
    ctx.fillText(emoji, 64, 64);
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
  }, [emoji]);
}

// --- Particle System Logic ---

const GenericParticleSystem = ({
    count,
    color,
    geometryType,
    distributionType,
    scale = 1,
    targetState,
    rotationTarget,
    emissiveColor = "black", // Default to no emission
    emissiveIntensity = 0,
    metalness = 0.1,
    roughness = 0.8
}: {
    count: number;
    color: string | string[];
    geometryType: "tetra" | "sphere" | "box" | "octahedron";
    distributionType: "foliage" | "snow-tips" | "ornament" | "ribbon";
    scale?: number;
    targetState: InteractionState;
    rotationTarget: number;
    emissiveColor?: string;
    emissiveIntensity?: number;
    metalness?: number;
    roughness?: number;
}) => {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);
    
    // IMPORTANT: Ensure stable reference for colors to prevent re-generation loop
    const colors = useMemo(() => Array.isArray(color) ? color : [color], [color]); 
    const colorArray = useMemo(() => new Float32Array(count * 3), [count, colors]);

    // Generate positions
    const { posTree, posUniv, randomRotations } = useMemo(() => {
        const tree = new Float32Array(count * 3);
        const univ = new Float32Array(count * 3);
        const rRot = new Float32Array(count * 3);
        const _color = new THREE.Color();

        for (let i = 0; i < count; i++) {
            // Color Assigment
            _color.set(colors[Math.floor(Math.random() * colors.length)]);
            colorArray[i * 3] = _color.r;
            colorArray[i * 3 + 1] = _color.g;
            colorArray[i * 3 + 2] = _color.b;

            // --- Tree Position Logic ---
            let tx, ty, tz;
            const h = Math.random() * TREE_HEIGHT; 
            const normH = h / TREE_HEIGHT;
            const maxR = TREE_RADIUS * (1 - normH);
            const layerMod = 0.8 + 0.25 * Math.sin(h * 2.5); 
            const layerR = maxR * layerMod;

            if (distributionType === "ribbon") {
                // Spiral logic
                const ribH = (i / count) * TREE_HEIGHT;
                const ribNormH = ribH / TREE_HEIGHT;
                const ribMaxR = TREE_RADIUS * (1 - ribNormH) * 1.05; 
                
                // Tighter Spiral
                const turns = 7;
                const angle = ribNormH * Math.PI * 2 * turns;
                
                ty = ribH - (TREE_HEIGHT / 2);
                tx = Math.cos(angle) * ribMaxR;
                tz = Math.sin(angle) * ribMaxR;

                // Minimal scatter for clean spiral
                tx += (Math.random() - 0.5) * 0.2;
                tz += (Math.random() - 0.5) * 0.2;
                ty += (Math.random() - 0.5) * 0.1;

            } else if (distributionType === "snow-tips") {
                ty = h - (TREE_HEIGHT / 2);
                const theta = Math.random() * Math.PI * 2;
                const r = layerR * randomRange(0.85, 1.1); 
                tx = Math.cos(theta) * r;
                tz = Math.sin(theta) * r;

            } else if (distributionType === "ornament") {
                ty = h - (TREE_HEIGHT / 2);
                const theta = Math.random() * Math.PI * 2;
                const r = layerR * randomRange(0.8, 1.0);
                tx = Math.cos(theta) * r;
                tz = Math.sin(theta) * r;

            } else {
                // Foliage
                ty = h - (TREE_HEIGHT / 2);
                const theta = Math.random() * Math.PI * 2;
                const r = Math.pow(Math.random(), 0.5) * layerR;
                tx = Math.cos(theta) * r;
                tz = Math.sin(theta) * r;
            }

            tree[i * 3] = tx;
            tree[i * 3 + 1] = ty;
            tree[i * 3 + 2] = tz;

            // --- Universe Position Logic ---
            const rUni = randomRange(15, 60);
            const uTheta = Math.random() * Math.PI * 2;
            const uPhi = Math.acos(2 * Math.random() - 1);
            univ[i * 3] = rUni * Math.sin(uPhi) * Math.cos(uTheta);
            univ[i * 3 + 1] = rUni * Math.sin(uPhi) * Math.sin(uTheta);
            univ[i * 3 + 2] = rUni * Math.cos(uPhi);

            rRot[i * 3] = Math.random();
            rRot[i * 3 + 1] = Math.random();
            rRot[i * 3 + 2] = Math.random();
        }
        return { posTree: tree, posUniv: univ, randomRotations: rRot };
    }, [count, distributionType, colors, colorArray]);

    const currentPos = useRef(new Float32Array(posTree));

    useFrame((state) => {
        if (!meshRef.current) return;
        const time = state.clock.elapsedTime;
        const step = targetState === "exploded" ? EXPLOSION_SPEED : RETURN_SPEED;
        const rotY = (rotationTarget - 0.5) * Math.PI * 4;
        
        for (let i = 0; i < count; i++) {
            const ix = i * 3;
            const tx = targetState === "exploded" ? posUniv[ix] : posTree[ix];
            const ty = targetState === "exploded" ? posUniv[ix+1] : posTree[ix+1];
            const tz = targetState === "exploded" ? posUniv[ix+2] : posTree[ix+2];

            // Lerp Position
            currentPos.current[ix] += (tx - currentPos.current[ix]) * step;
            currentPos.current[ix+1] += (ty - currentPos.current[ix+1]) * step;
            currentPos.current[ix+2] += (tz - currentPos.current[ix+2]) * step;

            let x = currentPos.current[ix];
            let y = currentPos.current[ix+1];
            let z = currentPos.current[ix+2];

            // Add slight ambient float if exploded
            if (targetState === "exploded") {
                x += Math.sin(time * 0.5 + i) * 0.05;
                y += Math.cos(time * 0.3 + i) * 0.05;
            }

            // Apply Hand Rotation regardless of state (Tree OR Exploded)
            const cosR = Math.cos(rotY);
            const sinR = Math.sin(rotY);
            const rx = x * cosR - z * sinR;
            const rz = x * sinR + z * cosR;
            x = rx;
            z = rz;

            dummy.position.set(x, y, z);
            dummy.rotation.set(
                time * randomRotations[ix] + i,
                time * randomRotations[ix+1] + i,
                time * randomRotations[ix+2] + i
            );
            
            // Pulse effect for emissive objects
            let s = scale;
            if (emissiveIntensity > 0) s = scale * (0.9 + Math.sin(time * 3 + i) * 0.2);
            dummy.scale.setScalar(s);

            dummy.updateMatrix();
            meshRef.current.setMatrixAt(i, dummy.matrix);
        }
        meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
            {geometryType === "tetra" && <tetrahedronGeometry args={[1, 0]} >
                <instancedBufferAttribute attach="attributes-color" args={[colorArray, 3]} />
            </tetrahedronGeometry>}
            {geometryType === "sphere" && <sphereGeometry args={[0.5, 16, 16]} />}
            {geometryType === "box" && <boxGeometry args={[0.8, 0.8, 0.8]} />}
            {geometryType === "octahedron" && <octahedronGeometry args={[1, 0]} />}
            
            <meshStandardMaterial 
                color={Array.isArray(color) ? "white" : color} // If using vertex colors, base must be white
                vertexColors={Array.isArray(color)}
                roughness={roughness}
                metalness={metalness}
                emissive={emissiveColor} // Fix: Use custom emissive color, not hardcoded white
                emissiveIntensity={emissiveIntensity}
                toneMapped={false}
            />
        </instancedMesh>
    );
};

// --- New Emoji Particle Component ---
const EmojiParticleSystem: React.FC<{
    emoji: string;
    targetState: InteractionState;
    rotationTarget: number;
}> = ({ emoji, targetState, rotationTarget }) => {
    const count = EMOJI_COUNT_PER_TYPE;
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const texture = useEmojiTexture(emoji);
    const dummy = useMemo(() => new THREE.Object3D(), []);

    // Generate positions
    const { posTree, posUniv, randomRotations } = useMemo(() => {
        const tree = new Float32Array(count * 3);
        const univ = new Float32Array(count * 3);
        const rRot = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
            // Scattered inside tree foliage
            const h = Math.random() * TREE_HEIGHT;
            const normH = h / TREE_HEIGHT;
            const maxR = TREE_RADIUS * (1 - normH);
            const layerMod = 0.8 + 0.25 * Math.sin(h * 2.5);
            const layerR = maxR * layerMod;
            
            const ty = h - (TREE_HEIGHT / 2);
            const theta = Math.random() * Math.PI * 2;
            const r = layerR * randomRange(0.4, 0.95);
            const tx = Math.cos(theta) * r;
            const tz = Math.sin(theta) * r;

            tree[i * 3] = tx;
            tree[i * 3 + 1] = ty;
            tree[i * 3 + 2] = tz;

            // Universe
            const rUni = randomRange(15, 60);
            const uTheta = Math.random() * Math.PI * 2;
            const uPhi = Math.acos(2 * Math.random() - 1);
            univ[i * 3] = rUni * Math.sin(uPhi) * Math.cos(uTheta);
            univ[i * 3 + 1] = rUni * Math.sin(uPhi) * Math.sin(uTheta);
            univ[i * 3 + 2] = rUni * Math.cos(uPhi);
            
            rRot[i * 3] = Math.random();
            rRot[i * 3 + 1] = Math.random();
            rRot[i * 3 + 2] = Math.random();
        }
        return { posTree: tree, posUniv: univ, randomRotations: rRot };
    }, [count]);

    const currentPos = useRef(new Float32Array(posTree));

    useFrame((state) => {
        if (!meshRef.current) return;
        const time = state.clock.elapsedTime;
        const step = targetState === "exploded" ? EXPLOSION_SPEED : RETURN_SPEED;
        const rotY = (rotationTarget - 0.5) * Math.PI * 4;

        for (let i = 0; i < count; i++) {
            const ix = i * 3;
            // Lerp logic
            const tx = targetState === "exploded" ? posUniv[ix] : posTree[ix];
            const ty = targetState === "exploded" ? posUniv[ix+1] : posTree[ix+1];
            const tz = targetState === "exploded" ? posUniv[ix+2] : posTree[ix+2];

            currentPos.current[ix] += (tx - currentPos.current[ix]) * step;
            currentPos.current[ix+1] += (ty - currentPos.current[ix+1]) * step;
            currentPos.current[ix+2] += (tz - currentPos.current[ix+2]) * step;

            let x = currentPos.current[ix];
            let y = currentPos.current[ix+1];
            let z = currentPos.current[ix+2];

            if (targetState === "exploded") {
                 x += Math.sin(time * 0.5 + i) * 0.05;
                 y += Math.cos(time * 0.3 + i) * 0.05;
            }

            // Apply Hand Rotation regardless of state
            const cosR = Math.cos(rotY);
            const sinR = Math.sin(rotY);
            const rx = x * cosR - z * sinR;
            const rz = x * sinR + z * cosR;
            x = rx;
            z = rz;

            dummy.position.set(x, y, z);
            // Billboard effect: look at camera manually? Or just simple rotation
            // For emoji, we often want them facing generally out or spinning
            dummy.rotation.set(
                time * 0.5 + randomRotations[ix], 
                time * 0.5 + randomRotations[ix+1], 
                0
            );
            
            dummy.scale.setScalar(0.6); // Scale of emoji
            dummy.updateMatrix();
            meshRef.current.setMatrixAt(i, dummy.matrix);
        }
        meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
            <planeGeometry args={[1, 1]} />
            <meshBasicMaterial map={texture} transparent alphaTest={0.5} side={THREE.DoubleSide} />
        </instancedMesh>
    );
};


const TopStar = ({ targetState, rotationTarget }: { targetState: InteractionState, rotationTarget: number }) => {
    const ref = useRef<THREE.Group>(null);
    useFrame((state) => {
        if(!ref.current) return;
        const time = state.clock.elapsedTime;
        const rotY = (rotationTarget - 0.5) * Math.PI * 4;
        
        ref.current.position.y = 9.5 + Math.sin(time * 2) * 0.2;
        
        // Rotation for both states
        ref.current.rotation.y = rotY + time * 0.5;

        if (targetState === "tree") {
            ref.current.scale.lerp(new THREE.Vector3(1.8, 1.8, 1.8), 0.1);
        } else {
            ref.current.scale.lerp(new THREE.Vector3(0, 0, 0), 0.1);
        }
    });

    return (
        <group ref={ref}>
            <mesh>
                {/* Changed to 0.7 radius for smaller, sharper star look */}
                <octahedronGeometry args={[0.7, 0]} />
                <meshStandardMaterial emissive="#fbbf24" emissiveIntensity={2} color="#fbbf24" toneMapped={false} />
            </mesh>
            <Sparkles count={40} scale={5} size={8} speed={0.4} opacity={1} color="#fbbf24" />
        </group>
    );
};

const FallingSnow = () => {
    const count = 400;
    const mesh = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);
    const particles = useMemo(() => {
        return new Array(count).fill(0).map(() => ({
            x: randomRange(-25, 25),
            y: randomRange(0, 40),
            z: randomRange(-25, 25),
            speed: randomRange(0.02, 0.08),
            sway: randomRange(0, Math.PI * 2)
        }));
    }, []);

    useFrame((state) => {
        if (!mesh.current) return;
        const time = state.clock.getElapsedTime();
        particles.forEach((p, i) => {
            p.y -= p.speed;
            p.x += Math.sin(time + p.sway) * 0.01;
            if (p.y < -15) p.y = 30;
            dummy.position.set(p.x, p.y, p.z);
            dummy.scale.setScalar(0.12);
            dummy.updateMatrix();
            mesh.current!.setMatrixAt(i, dummy.matrix);
        });
        mesh.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh ref={mesh} args={[undefined, undefined, count]}>
            <sphereGeometry args={[0.5, 6, 6]} />
            <meshBasicMaterial color="#ffffff" transparent opacity={0.6} />
        </instancedMesh>
    );
};

const Scene = ({ interactionState, handX }: { interactionState: InteractionState; handX: number }) => {
  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 2, 38]} fov={45} />
      <OrbitControls enableZoom={false} enablePan={false} maxPolarAngle={Math.PI/1.8} minPolarAngle={Math.PI/2.5} />
      
      {/* Lights */}
      <ambientLight intensity={0.6} color="#34d399" /> 
      <pointLight position={[15, 20, 15]} intensity={2} color="#fbbf24" distance={50} decay={2} /> 
      <pointLight position={[-15, 10, -5]} intensity={1.5} color="#4ade80" distance={50} /> 
      <spotLight position={[0, 40, -10]} intensity={3} angle={0.6} penumbra={0.5} color="#86efac" />

      <Stars radius={100} depth={50} count={2500} factor={4} saturation={0} fade speed={0.5} />
      <FallingSnow />

      <Float speed={1} rotationIntensity={0.1} floatIntensity={0.2}>
        <group position={[0, -2, 0]}>
            
            {/* 1. Main Foliage - Vibrant Green */}
            <GenericParticleSystem 
                count={COUNT_LEAVES} 
                // Use static color constant
                color={LEAF_COLORS} 
                geometryType="tetra" 
                distributionType="foliage" 
                scale={0.55} 
                targetState={interactionState} 
                rotationTarget={handX} 
                roughness={0.7}
                metalness={0.0}
                // Fix: Use dark green emissive, not white, to allow color to show through
                emissiveColor="#052e16" 
                emissiveIntensity={0.5}
            />

            {/* 2. Snow on Tips */}
            <GenericParticleSystem 
                count={COUNT_SNOW} 
                color="#e2e8f0" 
                geometryType="sphere" 
                distributionType="snow-tips" 
                scale={0.3} 
                targetState={interactionState} 
                rotationTarget={handX} 
                emissiveColor="#ffffff"
                emissiveIntensity={0.8}
            />

            {/* 3. New Emoji Particles (~10% of tree) */}
            {HOLIDAY_EMOJIS.map((emoji) => (
                <EmojiParticleSystem 
                    key={emoji} 
                    emoji={emoji} 
                    targetState={interactionState} 
                    rotationTarget={handX} 
                />
            ))}

            {/* 4. Ornaments - Gold & Teal */}
             <GenericParticleSystem 
                count={COUNT_ORNAMENTS} 
                // Use static color constant
                color={ORNAMENT_COLORS} 
                geometryType="sphere" 
                distributionType="ornament" 
                scale={0.45} 
                targetState={interactionState} 
                rotationTarget={handX} 
                metalness={0.9}
                roughness={0.1}
            />

            {/* 5. Golden Spiral Ribbon - 3D Stars (Octahedron), Small */}
            <GenericParticleSystem 
                count={COUNT_RIBBON} 
                // Use static color constant
                color={RIBBON_COLORS} 
                geometryType="octahedron" 
                distributionType="ribbon" 
                scale={0.12} 
                targetState={interactionState} 
                rotationTarget={handX} 
                emissiveColor="#fbbf24"
                emissiveIntensity={1}
                metalness={1}
                roughness={0}
            />

            {/* Top Star */}
            <TopStar targetState={interactionState} rotationTarget={handX} />
        </group>
      </Float>

      <EffectComposer enableNormalPass={false}>
        <Bloom luminanceThreshold={0.7} mipmapBlur intensity={1.5} radius={0.5} />
        <Vignette eskil={false} offset={0.1} darkness={0.6} />
      </EffectComposer>
    </>
  );
};

const App = () => {
  const [interactionState, setInteractionState] = useState<InteractionState>("tree");
  const [handX, setHandX] = useState(0.5);

  return (
    <div className="relative w-full h-full bg-[#020617]">
      <HandTracker onUpdateInteraction={(isOpen, x) => {
          setInteractionState(isOpen ? "exploded" : "tree");
          setHandX(x);
      }} />

      <Canvas gl={{ antialias: true, toneMapping: THREE.ReinhardToneMapping, toneMappingExposure: 1.2 }} dpr={[1, 2]}>
        <Suspense fallback={null}>
            <Scene interactionState={interactionState} handX={handX} />
        </Suspense>
      </Canvas>

      <div className="absolute top-12 w-full flex flex-col items-center pointer-events-none select-none z-10 px-4">
        <h1 className="text-6xl md:text-8xl gold-text text-center drop-shadow-2xl" style={{ fontFamily: "'Pinyon Script', cursive" }}>
          Merry Christmas
        </h1>
        <p className="text-amber-200/80 text-sm md:text-base tracking-[0.5em] mt-2 font-serif uppercase">
            wish u a nice day
        </p>
      </div>
    </div>
  );
};

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
