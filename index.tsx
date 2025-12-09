
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
const EXPLOSION_SPEED = 0.08;
const RETURN_SPEED = 0.06;

// Particle Counts
const COUNT_LEAVES = 4000;
const COUNT_SNOW = 1200;
const COUNT_GIFTS = 150;     // Small boxes
const COUNT_ORNAMENTS = 150; // Round balls

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
        <button onClick={enableCam} disabled={loading} className="bg-emerald-900/90 text-amber-100 px-4 py-2 rounded-full text-xs font-bold shadow-lg border border-amber-500/30 transition-all hover:bg-emerald-800">
          {loading ? "Loading AI..." : "Enable Hand Control"}
        </button>
      )}
    </div>
  );
};

// --- Particle System Logic ---

const GenericParticleSystem = ({
    count,
    color,
    geometryType,
    distributionType,
    scale = 1,
    targetState,
    rotationTarget,
    emissive = false
}: {
    count: number;
    color: string | string[];
    geometryType: "tetra" | "sphere" | "box";
    distributionType: "foliage" | "snow-tips" | "ornament" | "gifts";
    scale?: number;
    targetState: InteractionState;
    rotationTarget: number;
    emissive?: boolean;
}) => {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);
    const colors = Array.isArray(color) ? color : [color];
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
            // Normalized height 0 (bottom) to 1 (top)
            const normH = h / TREE_HEIGHT;
            
            // Base Cone Radius
            const maxR = TREE_RADIUS * (1 - normH);

            // Layer/Branch Logic: Sine wave creates "tiers"
            // freq = 5 layers
            const layerMod = 0.8 + 0.25 * Math.sin(h * 2.5); 
            const layerR = maxR * layerMod;

            if (distributionType === "snow-tips") {
                // Snow sits on the top/outer edges of the layers
                ty = h - (TREE_HEIGHT / 2);
                const theta = Math.random() * Math.PI * 2;
                // Bias towards outer edge
                const r = layerR * randomRange(0.85, 1.1); 
                tx = Math.cos(theta) * r;
                tz = Math.sin(theta) * r;

            } else if (distributionType === "gifts") {
                // Gifts scattered inside or on surface
                ty = h - (TREE_HEIGHT / 2);
                const theta = Math.random() * Math.PI * 2;
                const r = layerR * randomRange(0.2, 0.9);
                tx = Math.cos(theta) * r;
                tz = Math.sin(theta) * r;

            } else if (distributionType === "ornament") {
                // Ornaments on tips
                ty = h - (TREE_HEIGHT / 2);
                const theta = Math.random() * Math.PI * 2;
                const r = layerR * randomRange(0.9, 1.0);
                tx = Math.cos(theta) * r;
                tz = Math.sin(theta) * r;

            } else {
                // Foliage: Volume filling
                ty = h - (TREE_HEIGHT / 2);
                const theta = Math.random() * Math.PI * 2;
                const r = Math.sqrt(Math.random()) * layerR;
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

            // Random rotation speed/axis
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

            if (targetState === "tree") {
                // Apply tree rotation
                const cosR = Math.cos(rotY);
                const sinR = Math.sin(rotY);
                const rx = x * cosR - z * sinR;
                const rz = x * sinR + z * cosR;
                x = rx;
                z = rz;
            } else {
                // Floating noise
                x += Math.sin(time * 0.5 + i) * 0.05;
                y += Math.cos(time * 0.3 + i) * 0.05;
            }

            dummy.position.set(x, y, z);
            
            // Random particle rotation
            dummy.rotation.x = time * randomRotations[ix] + i;
            dummy.rotation.y = time * randomRotations[ix+1] + i;
            dummy.rotation.z = time * randomRotations[ix+2] + i;

            // Breathing scale for magic feel
            let s = scale;
            if (emissive) s = scale * (0.8 + Math.sin(time * 3 + i) * 0.4);
            dummy.scale.setScalar(s);

            dummy.updateMatrix();
            meshRef.current.setMatrixAt(i, dummy.matrix);
        }
        meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
            {geometryType === "tetra" && <tetrahedronGeometry args={[1, 0]} />}
            {geometryType === "sphere" && <sphereGeometry args={[0.5, 8, 8]} />}
            {geometryType === "box" && <boxGeometry args={[0.8, 0.8, 0.8]} />}
            
            <meshStandardMaterial 
                color={Array.isArray(color) ? undefined : color}
                vertexColors={Array.isArray(color)}
                roughness={0.8}
                metalness={0.1}
                emissive={emissive ? "white" : "black"}
                emissiveIntensity={emissive ? 1 : 0}
                toneMapped={false}
            />
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
        
        if (targetState === "tree") {
            ref.current.rotation.y = rotY + time;
            ref.current.scale.lerp(new THREE.Vector3(1.5, 1.5, 1.5), 0.1);
        } else {
            ref.current.scale.lerp(new THREE.Vector3(0, 0, 0), 0.1);
        }
    });

    return (
        <group ref={ref}>
            <mesh>
                <octahedronGeometry args={[1, 0]} />
                <meshStandardMaterial emissive="#fbbf24" emissiveIntensity={3} color="#fbbf24" toneMapped={false} />
            </mesh>
            <Sparkles count={20} scale={4} size={6} speed={0.4} opacity={1} color="#fbbf24" />
        </group>
    );
};

const FallingSnow = () => {
    const count = 300;
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
            <meshBasicMaterial color="#fff" transparent opacity={0.4} />
        </instancedMesh>
    );
};

const Scene = ({ interactionState, handX }: { interactionState: InteractionState; handX: number }) => {
  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 2, 35]} fov={45} />
      <OrbitControls enableZoom={false} enablePan={false} maxPolarAngle={Math.PI/1.8} minPolarAngle={Math.PI/2.5} />
      
      {/* Lighting Setup */}
      <ambientLight intensity={0.4} color="#a5f3fc" /> {/* Cool ambient light */}
      <pointLight position={[10, 15, 10]} intensity={1.5} color="#fbbf24" /> {/* Warm Sun */}
      <pointLight position={[-10, 5, -10]} intensity={1} color="#3b82f6" /> {/* Blue Backlight */}
      <spotLight position={[0, 30, 0]} intensity={1} angle={0.5} penumbra={1} color="#fff" />

      <Stars radius={100} depth={50} count={2000} factor={4} saturation={0} fade speed={1} />
      <FallingSnow />

      <Float speed={1} rotationIntensity={0.1} floatIntensity={0.2}>
        <group position={[0, -2, 0]}>
            
            {/* 1. Main Green Foliage - Layered */}
            <GenericParticleSystem 
                count={COUNT_LEAVES} 
                color={["#14532d", "#15803d", "#166534", "#064e3b"]} // Deep rich greens
                geometryType="tetra" 
                distributionType="foliage" 
                scale={0.55} 
                targetState={interactionState} 
                rotationTarget={handX} 
            />

            {/* 2. Snow on Tips - White, clustered on layer edges */}
            <GenericParticleSystem 
                count={COUNT_SNOW} 
                color="#f8fafc" 
                geometryType="sphere" 
                distributionType="snow-tips" 
                scale={0.35} 
                targetState={interactionState} 
                rotationTarget={handX} 
            />

            {/* 3. Gift Boxes - Scattered inside */}
            <GenericParticleSystem 
                count={COUNT_GIFTS} 
                color={["#ef4444", "#fbbf24", "#3b82f6", "#ffffff"]} // Red, Gold, Blue, White wrapping
                geometryType="box" 
                distributionType="gifts" 
                scale={0.45} 
                targetState={interactionState} 
                rotationTarget={handX} 
            />

             {/* 4. Ornaments - Shiny balls */}
             <GenericParticleSystem 
                count={COUNT_ORNAMENTS} 
                color={["#dc2626", "#fbbf24"]} 
                geometryType="sphere" 
                distributionType="ornament" 
                scale={0.4} 
                targetState={interactionState} 
                rotationTarget={handX} 
                emissive={true}
            />

            {/* Top Star */}
            <TopStar targetState={interactionState} rotationTarget={handX} />
        </group>
      </Float>

      <EffectComposer disableNormalPass>
        <Bloom luminanceThreshold={0.8} mipmapBlur intensity={1.2} radius={0.4} />
        <Vignette eskil={false} offset={0.1} darkness={0.5} />
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

      <Canvas gl={{ antialias: true, toneMapping: THREE.ReinhardToneMapping, toneMappingExposure: 1.5 }} dpr={[1, 2]}>
        <Suspense fallback={null}>
            <Scene interactionState={interactionState} handX={handX} />
        </Suspense>
      </Canvas>

      <div className="absolute top-10 w-full flex flex-col items-center pointer-events-none select-none z-10 px-4">
        <h1 className="text-6xl md:text-8xl gold-text text-center drop-shadow-2xl" style={{ fontFamily: "'Vladimir Script', 'Pinyon Script', cursive" }}>
          Merry Christmas
        </h1>
        <p className="text-amber-100/70 text-base md:text-lg tracking-[0.3em] mt-2 font-serif italic">
            Season's Greetings
        </p>
      </div>
    </div>
  );
};

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
